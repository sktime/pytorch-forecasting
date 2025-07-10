"""Automated test for all metrics in PyTorch Forecasting."""

import pytest
import torch

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

    def _test_reduction_modes(self, metric_class, y_pred, y, test_params):
        """Test that all metrics support different reduction modes."""

        for reduction in ["mean", "none"]:
            metric = metric_class(**test_params)
            if metric_class.__name__ in [
                "NormalDistributionLoss",
                "MultivariateNormalDistributionLoss",
                "ImplicitQuantileNetworkDistributionLoss",
            ]:  # noqa: E501
                metric._transformation = None
            metric.update(y_pred, y)
            result = metric.reduce_loss(
                metric.losses, metric.lengths, reduction=reduction
            )  # noqa: E501
            assert isinstance(result, torch.Tensor)

            if reduction == "none":
                result.shape == metric.losses
            else:
                assert result.ndim == 0
                if reduction == "sqrt-mean":
                    assert (
                        result >= 0
                    ), "Result should be non-negative for sqrt-mean reduction."  # noqa: E501

    @pytest.mark.parametrize("target_type", ["standard", "packed", "weighted"])
    def test_metric_functionality(self, object_pkg, object_class, request, target_type):
        """Test metric functionality with appropriate test data."""

        prepare_data_fixture_name = object_pkg.requires_data_type()
        data = request.getfixturevalue(prepare_data_fixture_name)

        if target_type not in data["test_cases"]:
            return None

        test_case = data["test_cases"][target_type]
        y_pred, y = object_pkg.prepare_test_inputs(test_case)

        test_params = object_pkg.get_test_params()

        metric = object_class(**test_params)

        torch_encoder = object_pkg.get_encoder()
        y_pred = metric.rescale_parameters(
            parameters=y_pred,
            target_scale=test_case["x"]["target_scale"],
            encoder=torch_encoder,
        )

        self._test_integration_metrics(metric, y_pred, y)
        self._test_reduction_modes(object_class, y_pred, y, test_params)
