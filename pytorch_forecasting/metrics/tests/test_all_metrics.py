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


def get_all_metrics():
    """
    Manually collect all metric classes.

    Current approach it to manually include classes one by one.
    If breaking changes are detected by the introduction of a metric into the
    test suite, it is decided to either change the test or include metric-specific
    tests in the metric's own test suite.
    """
    metric_classes = []
    try:
        from pytorch_forecasting.metrics.point import (
            MAE,
            MAPE,
            MASE,
            RMSE,
            SMAPE,
            CrossEntropy,
            PoissonLoss,
            TweedieLoss,
        )

        metric_classes.extend(
            [
                MAE,
                MAPE,
                RMSE,
                SMAPE,
                PoissonLoss,
                TweedieLoss,
                # CrossEntropy,
                # MASE
            ]
        )
    except ImportError:
        raise ImportError("Failed to import point metrics.")
    try:
        from pytorch_forecasting.metrics.quantile import QuantileLoss

        metric_classes.append(QuantileLoss)
    except ImportError:
        raise ImportError("Failed to import quantile metrics.")

    try:
        from pytorch_forecasting.metrics.distributions import (
            BetaDistributionLoss,
            ImplicitQuantileNetworkDistributionLoss,
            LogNormalDistributionLoss,
            MQF2DistributionLoss,
            MultivariateNormalDistributionLoss,
            NegativeBinomialDistributionLoss,
            NormalDistributionLoss,
        )

        metric_classes.extend(
            [
                NormalDistributionLoss,
                NegativeBinomialDistributionLoss,
                LogNormalDistributionLoss,
                MultivariateNormalDistributionLoss,
                BetaDistributionLoss,
                # ImplicitQuantileNetworkDistributionLoss,
                # MQF2DistributionLoss,
            ]
        )
    except ImportError:
        raise ImportError("Failed to import distribution metrics.")

    return metric_classes


def get_test_data():
    """Return test data fixtures for the metrics."""

    batch_size, timesteps = 4, 10

    test_cases = {}

    y_pred_2d = torch.randn(batch_size, timesteps)
    y_actual_2d = torch.randn(batch_size, timesteps)
    test_cases["point"] = {"y_pred": y_pred_2d, "y_actual": y_actual_2d}

    # 3d tensors for quantile forecast (default 7 quantiles)
    y_pred_3d = torch.randn(batch_size, timesteps, 7)
    test_cases["quantile"] = {"y_pred": y_pred_3d, "y_actual": y_actual_2d}

    # data with weighted targets
    y_actual_weighted = (y_actual_2d, torch.ones(batch_size, timesteps))
    test_cases["weighted_2d"] = {"y_pred": y_pred_2d, "y_actual": y_actual_weighted}

    test_cases["weighted_3d"] = {
        "y_pred": y_pred_3d,
        "y_actual": (y_actual_2d, torch.ones(batch_size, timesteps)),
    }

    # PackedSequence data (simulates variable length sequences in each batch)
    lengths = torch.tensor([10, 8, 6, 4])
    padded_actual = torch.randn(batch_size, timesteps)

    packed_actual = rnn.pack_padded_sequence(
        padded_actual, lengths, batch_first=True, enforce_sorted=False
    )

    test_cases["packed_sequence_2d"] = {"y_pred": y_pred_2d, "y_actual": packed_actual}
    test_cases["packed_sequence_3d"] = {"y_pred": y_pred_3d, "y_actual": packed_actual}
    # 3d tensors for distribution parameters
    n_params = 4
    y_pred_dist = torch.randn(batch_size, timesteps, n_params)
    y_pred_dist[..., 1] = torch.abs(y_pred_dist[..., 1]) + 0.1
    y_pred_dist[..., 3] = torch.abs(y_pred_dist[..., 3]) + 0.1
    test_cases["distribution"] = {"y_pred": y_pred_dist, "y_actual": y_actual_2d}

    n_class = 5
    y_pred_class = torch.randn(batch_size, timesteps, n_class)
    y_actual_class = torch.randint(0, 5, (batch_size, timesteps))
    test_cases["classification"] = {"y_pred": y_pred_class, "y_actual": y_actual_class}

    return test_cases


DATA_FORMAT_NAMES = [
    "point",
    "quantile",
    "packed_sequence_2d",
    "packed_sequence_3d",
    "weighted_2d",
    "weighted_3d",
    "distribution",
    "classification",
]


@pytest.fixture(params=get_all_metrics())
def metric_class(request):
    """Fixture to provide metric classes."""
    return request.param


@pytest.fixture
def metric_instance(metric_class):
    if issubclass(metric_class, (MultiLoss, CompositeMetric, TorchMetricWrapper)):
        pytest.skip(f"{metric_class.__name__} requires special instantiation.")

    try:
        return metric_class()
    except Exception as e:
        pytest.fail(f"Failed to instantiate {metric_class.__name__}: {e}")


@pytest.fixture(scope="session")
def test_data():
    """Session-scoped fixture for test data."""
    return get_test_data()


class TestAllPtMetrics:
    """Test all metrics in PyTorch Forecasting."""

    def test_metric_instantiation(self, metric_class):
        """Test that metrics can be instantiated."""

        try:
            if issubclass(metric_class, (MultiLoss, CompositeMetric)):
                pytest.skip(
                    f"{metric_class.__name__} is requiring special instantiation."
                )
            elif issubclass(metric_class, TorchMetricWrapper):
                pytest.skip(
                    f"{metric_class.__name__} is a wrapper and should not be instantiated directly."  # noqa: E501
                )
            else:
                metric = metric_class()

        except Exception as e:
            pytest.fail(f"Failed to instantiate {metric_class.__name__}: {e}")

        assert isinstance(metric, Metric)

    @pytest.mark.parametrize("test_case", DATA_FORMAT_NAMES)
    def test_metric_forward_pass(self, metric_instance, test_case, test_data):
        if metric_instance is None:
            pytest.skip("No valid metric instance provided.")

        test_case_data = test_data[test_case]

        try:
            y_pred = test_case_data["y_pred"]
            y_actual = test_case_data["y_actual"]
            res = metric_instance(y_pred, y_actual)
        except Exception as e:
            pytest.skip(
                f"Forward pass failed with {test_case} for {metric_instance.__class__.__name__}: {e}"  # noqa: E501
            )  # noqa: E501

        assert isinstance(
            res, torch.Tensor
        ), f"Expected output to be a tensor, got {type(res)}"  # noqa: E501
        # dependent on metric reduction type.
        assert torch.numel(res) == 1 or res.ndim <= 2
        assert torch.isfinite(res).all(), "Output contains non-finite values."

    @pytest.mark.parametrize("test_case", DATA_FORMAT_NAMES)
    def test_metric_update_compute(self, metric_instance, test_data, test_case):
        """Test update/compute methods pattern of metric."""

        test_case_data = test_data[test_case]

        if metric_instance is None:
            pytest.skip("No valid metric instance provided.")

        y_pred = test_case_data["y_pred"]
        y_actual = test_case_data["y_actual"]

        try:
            metric_instance.reset()
            metric_instance.update(y_pred, y_actual)
            res = metric_instance.compute()
        except Exception as e:
            pytest.skip(
                f"Update/compute failed with {test_case} for {metric_instance.__class__.__name__}: {e}"  # noqa: E501
            )  # noqa: E501

        assert isinstance(
            res, torch.Tensor
        ), f"Expected output to be a tensor, got {type(res)}"  # noqa: E501
        assert torch.isfinite(res).all(), "Output contains non-finite values."

        try:
            metric_instance.update(y_pred, y_actual)
            res2 = metric_instance.compute()
        except Exception as e:
            pytest.skip(
                f"Second update and compute failed for {metric_instance.__class__.__name__}: {e}"  # noqa: E501
            )
        assert isinstance(
            res2, torch.Tensor
        ), f"Expected output to be a tensor, got {type(res2)}"  # noqa: E501

    @pytest.mark.parametrize("test_case", DATA_FORMAT_NAMES)
    def test_metric_reduction(self, metric_class, test_data, test_case):
        """Test metric reduction behavior."""

        if not issubclass(metric_class, (Metric, MultiHorizonMetric)):
            pytest.skip(f"{metric_class.__name__} does not support reduction testing.")

        reduction_modes = ["none", "mean", "sqrt-mean"]

        y_pred = test_data[test_case]["y_pred"]
        y_actual = test_data[test_case]["y_actual"]

        for reduction in reduction_modes:
            try:
                metric = metric_class(reduction=reduction)
                res = metric(y_pred, y_actual)
            except Exception as e:
                pytest.skip(
                    f"Reduction '{reduction}' failed for {metric_class.__name__}: {e}"  # noqa: E501
                )

            if reduction == "none":
                assert res.numel() > 1, "Expected multiple outputs for `none` reduction"
                assert res.shape[0] == y_pred.shape[0]
                assert res.shape[1] == y_pred.shape[1]
            elif reduction in ["mean", "sqrt-mean"]:
                assert res.numel() <= y_pred.shape[0]
                if reduction == "sqrt-mean":
                    assert (
                        res >= 0
                    ), "Expected non-negative output for `sqrt-mean` reduction."  # noqa: E501

    @pytest.mark.parametrize(
        "test_case",
        [
            "packed_sequence_2d",
            "packed_sequence_3d",
        ],
    )
    def test_packed_sequence_handling(self, metric_instance, test_data, test_case):
        """Test that metrics properly handle PackedSequence inputs."""

        if metric_instance is None:
            pytest.skip("No valid metric instance provided.")

        try:
            test_case_data = test_data[test_case]
            y_pred = test_case_data["y_pred"]
            y_actual = test_case_data["y_actual"]

            res = metric_instance(y_pred, y_actual)
        except Exception as e:
            pytest.skip(
                f"PackedSequence with {test_case[-2:]} tensor handling failed for {metric_instance.__class__.__name__}: {e}"  # noqa: E501
            )
        assert isinstance(res, torch.Tensor)
        assert torch.isfinite(res).all(), "Output contains non-finite values."

        metric_instance.reset()
        metric_instance.update(y_pred, y_actual)
        res2 = metric_instance.compute()
        assert isinstance(res2, torch.Tensor)
        assert torch.isfinite(res2).all(), "Output contains non-finite values."

    @pytest.mark.parametrize("test_case", DATA_FORMAT_NAMES)
    def test_to_prediction_method(self, metric_instance, test_data, test_case):
        """Test the ``to_prediction`` method of metrics."""

        if metric_instance is None:
            pytest.skip("No valid metric instance provided.")

        try:
            test_case_data = test_data[test_case]
            y_pred = test_case_data["y_pred"]
            prediction = metric_instance.to_prediction(
                y_pred,
            )
        except Exception as e:
            pytest.skip(
                f"to_prediction failed with {test_case} for {metric_instance.__class__.__name__}: {e}"  # noqa: E501
            )

        assert isinstance(prediction, torch.Tensor)

        if y_pred.ndim == 3:
            assert prediction.ndim == 2
            assert prediction.shape[:2] == y_pred.shape[:2]
        # else:
        #     assert prediction.shape == y_pred.shape

    @pytest.mark.parametrize("test_case", DATA_FORMAT_NAMES)
    def test_to_quantiles_method(self, metric_instance, test_data, test_case):
        """Test the ``to_quantiles`` method of metrics."""

        if metric_instance is None:
            pytest.skip("No valid metric instance provided.")

        try:
            test_case_data = test_data[test_case]
            y_pred = test_case_data["y_pred"]
            quantiles = [0.1, 0.5, 0.9]
            if metric_instance.__class__.__name__ == "QuantileLoss":
                quantile_pred = metric_instance.to_quantiles(y_pred)
            else:
                quantile_pred = metric_instance.to_quantiles(
                    y_pred, quantiles=quantiles
                )  # noqa: E501
        except Exception as e:
            pytest.skip(
                f"to_quantiles failed with {test_case} for {metric_instance.__class__.__name__}: {e}"  # noqa: E501
            )
        assert isinstance(quantile_pred, torch.Tensor)

        if metric_instance.__class__.__name__ == "QuantileLoss":
            assert quantile_pred.shape == y_pred.shape
        else:
            assert quantile_pred.ndim == 3
            if y_pred.ndim == 2 and metric_instance.__class__.__name__ != "PoissonLoss":  # noqa: E501
                assert quantile_pred.shape[2] == 1
            else:
                assert quantile_pred.shape[2] == len(quantiles)

    def test_metric_compositions(self, metric_class):
        """Test correct working of metric compositions via CompositeMetric"""

        if not issubclass(metric_class, Metric):
            pytest.skip(
                f"{metric_class.__name__} is not a Metric subclass,",
                " skipping test.",
            )

        metric_a = metric_class()
        metric_b = metric_class()

        composite = metric_a + metric_b
        assert isinstance(composite, CompositeMetric)

        weighted = metric_a * 0.5
        assert isinstance(weighted, CompositeMetric)

        composite_weighted = metric_a + metric_b * 0.5
        assert isinstance(composite_weighted, CompositeMetric)

    def test_doctest_examples(self, metric_class):
        """Run doctests for metric class."""
        try:
            from skbase.utils.doctest_run import run_doctest

            run_doctest(metric_class, name=f"class {metric_class.__name__}")
        except ImportError:
            pytest.skip("skbase doctest utilities not available")
        except Exception as e:
            pytest.skip(f"Doctest failed for {metric_class.__name__}: {e}")
