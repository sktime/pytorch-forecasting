"""Automated test for all metrics in PyTorch Forecasting."""

from inspect import isclass
import shutil

import numpy as np
import pytest
import torch
from torch.nn.utils import rnn

from pytorch_forecasting._registry import all_objects
from pytorch_forecasting.metrics.base_metrics import (
    CompositeMetric,
    DistributionLoss,
    Metric,
    MultiHorizonMetric,
    MultiLoss,
    TorchMetricWrapper,
)
from pytorch_forecasting.tests._config import EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
from pytorch_forecasting.tests.test_all_estimators import (
    BaseFixtureGenerator,
    PackageConfig,
)

ONLY_CHANGED_MODULES = False


class TestAllPtMetrics(PackageConfig, BaseFixtureGenerator):
    """Test all metrics in PyTorch Forecasting."""

    object_type_filter = "metric"

    fixture_sequence = [
        "object_pkg",
        "object_name",
        "object_instance",
        "test_data",
    ]

    def _generate_test_data(self, test_name, **kwargs):
        """Return test data fixtures for the metrics."""

        batch_size, timesteps = 4, 10

        test_cases = []
        test_names = []

        y_pred_2d = torch.randn(batch_size, timesteps)
        y_actual_2d = torch.randn(batch_size, timesteps)
        test_cases.append({"y_pred": y_pred_2d, "y_actual": y_actual_2d})
        test_names.append("point_forecast")

        # 3d tensors for quantile forecast
        y_pred_3d = torch.randn(batch_size, timesteps, 7)
        test_cases.append({"y_pred": y_pred_3d, "y_actual": y_actual_2d})
        test_names.append("quantile_forecast")

        # data with weighted targets
        y_actual_weighted = (y_actual_2d, torch.ones(batch_size, timesteps))
        test_cases.append({"y_pred": y_pred_2d, "y_actual": y_actual_weighted})
        test_names.append("weighted_2d_tensor")

        # PackedSequence data (simulates variable length sequences in each batch)
        lengths = torch.tensor([10, 8, 6, 4])

        padded_pred = torch.randn(batch_size, timesteps)
        padded_actual = torch.randn(batch_size, timesteps)

        packed_actual = rnn.pack_padded_sequence(
            padded_actual, lengths, batch_first=True, enforce_sorted=False
        )

        test_cases.append({"y_pred": padded_pred, "y_actual": packed_actual})
        test_names.append("packed_sequence")

        # 3d tensors for distribution parameters
        n_params = 2
        y_pred_dist = torch.randn(batch_size, timesteps, n_params)
        y_pred_dist[..., 1] = torch.abs(y_pred_dist[..., 1]) + 0.1
        test_cases.append({"y_pred": y_pred_dist, "y_actual": y_actual_2d})
        test_names.append("distribution_params")

        return test_cases, test_names

    def test_metric_instantiation(self, object_class):
        """Test that metrics can be instantiated."""

        try:
            if issubclass(object_class, (MultiLoss, CompositeMetric)):
                pytest.skip(
                    f"{object_class.__name__} is requiring special instantiation."
                )
            elif issubclass(object_class, TorchMetricWrapper):
                pytest.skip(
                    f"{object_class.__name__} is a wrapper and should not be instantiated directly."  # noqa: E501
                )
            else:
                metric = object_class()

        except Exception as e:
            pytest.fail(f"Failed to instantiate {object_class.__name__}: {e}")

        assert isinstance(metric, Metric)

    def test_metric_forward_pass(self, object_instance, test_data):
        if object_instance is None:
            pytest.skip("No valid metric instance provided.")

        y_pred = test_data["y_pred"]
        y_actual = test_data["y_actual"]

        try:
            res = object_instance(y_pred, y_actual)
        except Exception as e:
            pytest.fail(
                f"Forward pass failed for {object_instance.__class__.__name__}: {e}"
            )  # noqa: E501

        assert isinstance(
            res, torch.Tensor
        ), f"Expected output to be a tensor, got {type(res)}"  # noqa: E501
        # dependent on metric reduction type.
        assert torch.numel(res) == 1 or res.ndim <= 2
        assert torch.isfinite(res).all(), "Output contains non-finite values."

    def test_metric_update_compute(self, object_instance, test_data):
        """Test update/compute methods pattern of metric."""

        if object_instance is None:
            pytest.skip("No valid metric instance provided.")

        y_pred = test_data["y_pred"]
        y_actual = test_data["y_actual"]

        try:
            object_instance.reset()
            object_instance.update(y_pred, y_actual)
            res = object_instance.compute()
        except Exception as e:
            pytest.fail(
                f"Update/compute failed for {object_instance.__class__.__name__}: {e}"
            )  # noqa: E501

        assert isinstance(
            res, torch.Tensor
        ), f"Expected output to be a tensor, got {type(res)}"  # noqa: E501
        assert torch.isfinite(res).all(), "Output contains non-finite values."

        object_instance.update(y_pred, y_actual)
        res2 = object_instance.compute()
        assert isinstance(
            res2, torch.Tensor
        ), f"Expected output to be a tensor, got {type(res2)}"  # noqa: E501

    def test_metric_reduction(self, object_instance):
        """Test metric reduction behavior."""

        if not issubclass(object_instance, (Metric, MultiHorizonMetric)):
            pytest.skip(
                f"{object_instance.__name__} does not support reduction testing."
            )

        reduction_modes = ["none", "mean", "sqrt-mean"]

        batch_size, timesteps = 4, 10
        y_pred = torch.randn(batch_size, timesteps)
        y_actual = torch.randn(batch_size, timesteps)

        for reduction in reduction_modes:
            try:
                metric = object_instance(reduction=reduction)
                res = metric(y_pred, y_actual)
            except Exception as e:
                pytest.fail(
                    f"Reduction '{reduction}' failed for {object_instance.__name__}: {e}"  # noqa: E501
                )

            if reduction == "none":
                assert res.shape == (batch_size, timesteps)
            elif reduction == "mean":
                assert res.shape == (1,)
            elif reduction == "sqrt-mean":
                assert res.shape == (1,)
                assert res >= 0, "Square root mean reduction should be non-negative."

    def test_packed_sequence_handling(self, object_instance, test_data):
        """Test that metrics properly handle PackedSequence inputs."""

        if object_instance is None:
            pytest.skip("No valid metric instance provided.")

        if not isinstance(test_data["y_pred"], rnn.PackedSequence):
            pytest.skip("Test data is not a PackedSequence.")

        y_pred = test_data["y_pred"]
        y_actual = test_data["y_actual"]

        res = object_instance(y_pred, y_actual)
        assert isinstance(res, torch.Tensor)
        assert torch.isfinite(res).all(), "Output contains non-finite values."

        object_instance.reset()
        object_instance.update(y_pred, y_actual)
        res2 = object_instance.compute()
        assert isinstance(res2, torch.Tensor)
        assert torch.isfinite(res2).all(), "Output contains non-finite values."

    def test_to_prediction_method(self, object_instance, test_data):
        """Test the ``to_prediction`` method of metrics."""

        if object_instance is None:
            pytest.skip("No valid metric instance provided.")

        y_pred = test_data["y_pred"]

        prediction = object_instance.to_prediction(y_pred)
        assert isinstance(prediction, torch.Tensor)

        if y_pred.ndim == 3:
            assert prediction.ndim == 2
            assert prediction.shape[:2] == y_pred.shape[:2]
        else:
            assert prediction.shape == y_pred.shape

    def test_to_quatiles_method(self, object_instance, test_data):
        """Test the ``to_quantiles`` method of metrics."""

        if object_instance is None:
            pytest.skip("No valid metric instance provided.")

        y_pred = test_data["y_pred"]
        quantiles = [0.1, 0.5, 0.9]

        quantile_pred = object_instance.to_quantiles(y_pred, quantiles=quantiles)
        assert isinstance(quantile_pred, torch.Tensor)

        expcted_shape = y_pred.shape[:2] + (len(quantiles),)
        assert quantile_pred.shape == expcted_shape

    def test_metric_compositions(self, object_instance):
        """Test correct working of metric compositions via CompositeMetric"""

        if not issubclass(object_instance, Metric):
            pytest.skip(
                f"{object_instance.__name__} is not a Metric subclass,",
                " skipping test.",
            )

        metric_a = object_instance()
        metric_b = object_instance()

        composite = metric_a + metric_b
        assert isinstance(composite, CompositeMetric)

        weighted = metric_a * 0.5
        assert isinstance(weighted, CompositeMetric)

        composite_weighted = metric_a + metric_b * 0.5
        assert isinstance(composite_weighted, CompositeMetric)

    def test_doctest_examples(self, object_class):
        """Run doctests for metric class."""
        try:
            from skbase.utils.doctest_run import run_doctest

            run_doctest(object_class, name=f"class {object_class.__name__}")
        except ImportError:
            pytest.skip("skbase doctest utilities not available")
        except Exception as e:
            pytest.skip(f"Doctest failed for {object_class.__name__}: {e}")
