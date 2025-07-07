"""Automated test for all metrics in PyTorch Forecasting."""

from inspect import isclass
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn.utils import rnn

from pytorch_forecasting._registry import all_objects
from pytorch_forecasting.tests.test_all_estimators import (
    BaseFixtureGenerator,
    PackageConfig,
)


class TestAllPtMetrics(PackageConfig, BaseFixtureGenerator):
    """Test suite for all metrics in PyTorch Forecasting."""

    object_type_filter = "metric"

    fixture_sequence = ["object_pkg", "object_class"]

    def _test_integration_metrics(self, metric, y_pred, y):
        from pytorch_forecasting.metrics import PoissonLoss, QuantileLoss

        metric.reset()
        metric.update(y_pred, y)
        res = metric.compute()

        assert isinstance(res, torch.Tensor)
        assert torch.isfinite(res).all(), "Non-finite values in metric result."

        point_pred = metric.to_prediction(y_pred)
        assert isinstance(point_pred, torch.Tensor), "Prediction should be a tensor."
        assert (
            point_pred.shape[:2] == y_pred.shape[:2]
        ), "Prediction shape mismatch with y_pred."  # noqa: E501
        assert point_pred.ndim == 2

        quantiles = [0.1, 0.5, 0.9]
        if isinstance(metric, QuantileLoss):
            quantile_pred = metric.to_quantiles(y_pred)
            assert isinstance(
                quantile_pred, torch.Tensor
            ), "Quantile prediction should be a tensor."  # noqa: E501
            assert quantile_pred.shape == y_pred.shape
        else:
            quantile_pred = metric.to_quantiles(y_pred, quantiles=quantiles)
            assert isinstance(
                quantile_pred, torch.Tensor
            ), "Quantile prediction should be a tensor."  # noqa: E501
            if y_pred.ndim == 3 or isinstance(metric, PoissonLoss):
                assert quantile_pred.shape == (
                    y_pred.shape[0],
                    y_pred.shape[1],
                    len(quantiles),
                ), f"`to_quantiles()`, fails for metric {metric}()."
            elif y_pred.ndim == 2:
                assert quantile_pred.shape == (
                    y_pred.shape[0],
                    y_pred.shape[1],
                    1,
                ), f"`to_quantiles()`, fails for metric {metric}()."

            # testing composite metrics
            composite = metric + metric
            weighted = metric * 0.5

            composite_result = composite(y_pred, y)
            weighted_result = weighted(y_pred, y)

            assert isinstance(composite_result, torch.Tensor)
            assert isinstance(weighted_result, torch.Tensor)
            assert torch.isfinite(composite_result).all()
            assert torch.isfinite(weighted_result).all()

    def _test_reduction_modes(self, metric_class, y_pred, y):
        """Test that all metrics support different reduction modes."""

        for reduction in ["mean", "sqrt-mean", "none"]:
            metric = metric_class(reduction=reduction)
            if metric_class.__name__ == "NormalDistributionLoss":
                metric._transformation = None
            result = metric(y_pred, y)
            assert isinstance(result, torch.Tensor)

            if reduction == "none":
                if isinstance(y, rnn.PackedSequence):
                    assert (
                        result.shape[0] == y_pred.shape[0]
                    )  # only check across batch.
                else:
                    assert result.shape == y_pred.shape
            else:
                assert result.ndim == 0
                if reduction == "sqrt-mean":
                    assert (
                        result >= 0
                    ), "Result should be non-negative for sqrt-mean reduction."  # noqa: E501

    @pytest.mark.parametrize("target_type", ["standard", "packed", "weighted"])
    def test_metric_functionality(self, object_class, request, target_type):
        """Test metric functionality with appropriate test data."""
        # Skip abstract classes
        if object_class.__name__ in [
            "Metric",
            "MultiHorizonMetric",
            "DistributionLoss",
        ]:
            pytest.skip(f"Skipping abstract class {object_class.__name__}")

        prepare_data_fixture_name = object_class.requires_data_type()
        test_case = request.getfixturevalue(prepare_data_fixture_name)["test_cases"][
            target_type
        ]  # noqa: E501

        y_pred, y, kwargs = object_class.prepare_test_inputs(test_case)
        test_params = object_class.get_test_params()

        metric = object_class(**test_params, **kwargs)

        self._test_integration_metrics(metric, y_pred, y)
        self._test_reduction_modes(object_class, y_pred, y)
