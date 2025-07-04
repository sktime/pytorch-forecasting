"""Automated test for all metrics in PyTorch Forecasting."""

from inspect import isclass
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn.utils import rnn

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, TorchNormalizer
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    RMSE,
    SMAPE,
    NormalDistributionLoss,
    PoissonLoss,
    QuantileLoss,
    TweedieLoss,
)


@pytest.fixture(scope="module")
def prepare_point_forecast():
    """Prepare a point forecast dataset, for testing point forecast metrics."""

    batch_size, timesteps = 4, 20
    prediction_length = timesteps // 2

    df = pd.DataFrame(
        {
            "time_idx": np.tile(np.arange(timesteps), batch_size),
            "group_id": np.repeat(np.arange(batch_size), timesteps),
            "target": np.random.rand(batch_size * timesteps),
        }
    )

    for i in range(3):
        df[f"feature_{i}"] = np.random.randn(batch_size * timesteps)

    dataset = TimeSeriesDataSet(
        data=df,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=timesteps // 2,
        max_prediction_length=prediction_length,
        target_normalizer=GroupNormalizer(groups=["group_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    dataloader = dataset.to_dataloader(batch_size=batch_size, shuffle=True)
    x, y = next(iter(dataloader))
    y_pred = torch.randn(batch_size, prediction_length)

    test_cases = {}

    test_cases["standard"] = {
        "x": x,
        "y": y,
        "y_pred": y_pred,
    }
    # each batch has a different prediction length
    lengths = torch.tensor(
        [
            prediction_length,
            prediction_length - 2,
            prediction_length - 4,
            prediction_length - 6,
        ]
    )

    y_packed = rnn.pack_padded_sequence(
        y[0], lengths, batch_first=True, enforce_sorted=False
    )

    test_cases["packed"] = {
        "x": x,
        "y": y_packed,
        "y_pred": y_pred,
    }

    weights = torch.ones_like(y_pred)
    y_weighted = (y[0], weights)
    test_cases["weighted"] = {
        "x": x,
        "y": y_weighted,
        "y_pred": y_pred,
    }

    return {"dataset": dataset, "test_cases": test_cases}


@pytest.fixture(scope="module")
def prepare_quantile_forecast():
    """Prepare a quantile forecast dataset, for testing quantile metrics."""

    batch_size, timesteps = 4, 20
    prediction_length = timesteps // 2

    df = pd.DataFrame(
        {
            "time_idx": np.tile(np.arange(timesteps), batch_size),
            "group_id": np.repeat(np.arange(batch_size), timesteps),
            "target": np.random.rand(batch_size * timesteps),
        }
    )

    for i in range(3):
        df[f"feature_{i}"] = np.random.randn(batch_size * timesteps)

    dataset = TimeSeriesDataSet(
        data=df,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=timesteps // 2,
        max_prediction_length=prediction_length,
        target_normalizer=GroupNormalizer(groups=["group_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    dataloader = dataset.to_dataloader(batch_size=batch_size, shuffle=True)
    x, y = next(iter(dataloader))

    quantiles = [0.1, 0.5, 0.9]  # Example quantiles
    y_pred = torch.randn(batch_size, prediction_length, len(quantiles))

    y_pred, _ = torch.sort(
        y_pred, dim=-1
    )  # sort along quantile dimension, to simulate perfect quantile distribution.  # noqa: E501

    test_cases = {}
    test_cases["standard"] = {"x": x, "y": y, "y_pred": y_pred, "quantiles": quantiles}

    lengths = torch.tensor(
        [
            prediction_length,
            prediction_length - 2,
            prediction_length - 4,
            prediction_length - 6,
        ]
    )
    packed_y = rnn.pack_padded_sequence(
        y[0], lengths, batch_first=True, enforce_sorted=False
    )
    test_cases["packed"] = {
        "x": x,
        "y": packed_y,
        "y_pred": y_pred,
        "quantiles": quantiles,
    }  # noqa: E501

    weights = torch.ones_like(y[0])
    weighted_y = (y[0], weights)
    test_cases["weighted"] = {
        "x": x,
        "y": weighted_y,
        "y_pred": y_pred,
        "quantiles": quantiles,
    }  # noqa: E501

    return {"dataset": dataset, "test_cases": test_cases}


@pytest.fixture(scope="module")
def prepare_normal_distribution_forecast():
    """Prepare data for normal distribution loss metrics."""

    batch_size, timesteps = 4, 20
    prediction_length = timesteps // 2

    mean = 1.0
    std = 0.1
    normal_dist = torch.distributions.Normal(loc=mean, scale=std)
    normal_target = normal_dist.sample((batch_size, timesteps)).numpy()

    df = pd.DataFrame(
        {
            "group_id": np.repeat(np.arange(batch_size), timesteps),
            "time_idx": np.tile(np.arange(timesteps), batch_size),
            "target": normal_target.flatten(),
        }
    )

    for i in range(3):
        df[f"feature_{i}"] = np.random.randn(batch_size * timesteps)

    dataset = TimeSeriesDataSet(
        data=df,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=prediction_length,
        max_prediction_length=prediction_length,
        target_normalizer=TorchNormalizer(),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    dataloader = dataset.to_dataloader(batch_size=batch_size, shuffle=True)
    x, y = next(iter(dataloader))

    # create random prediction paired with a normal distribution.
    y_pred = torch.stack(
        [
            torch.randn(batch_size, prediction_length),
            torch.abs(torch.randn(batch_size, prediction_length)) + 0.1,
        ],
        dim=-1,
    )

    test_cases = {}
    test_cases["standard"] = {"x": x, "y": y, "y_pred": y_pred}
    lengths = torch.tensor(
        [
            prediction_length,
            prediction_length - 2,
            prediction_length - 4,
            prediction_length - 6,
        ]
    )
    y_packed = rnn.pack_padded_sequence(
        y[0], lengths, batch_first=True, enforce_sorted=False
    )

    test_cases["packed"] = {"x": x, "y": y_packed, "y_pred": y_pred}

    weights = torch.ones_like(y[0])
    y_weighted = (y[0], weights)
    test_cases["weighted"] = {"x": x, "y": y_weighted, "y_pred": y_pred}

    return {"dataset": dataset, "test_cases": test_cases}


@pytest.fixture(params=["standard", "packed", "weighted"])
def point_forecast_case(request, prepare_point_forecast):
    """Parametrized fixture for point forecast test cases."""
    return prepare_point_forecast["test_cases"][request.param]


@pytest.fixture(params=["standard", "packed", "weighted"])
def quantile_forecast_case(request, prepare_quantile_forecast):
    """Parametrized fixture for quantile forecast test cases."""
    return prepare_quantile_forecast["test_cases"][request.param]


@pytest.fixture(params=["standard", "packed", "weighted"])
def normal_distribution_case(request, prepare_normal_distribution_forecast):
    """Parametrized fixture for normal distribution test cases."""
    return prepare_normal_distribution_forecast["test_cases"][request.param]


class TestAllPtMetrics:
    """Test suite for all metrics in PyTorch Forecasting."""

    def _test_integration_metrics(self, metric, y_pred, y):
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
            if isinstance(metric_class, NormalDistributionLoss):
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

    def test_point_forecast_metrics(self, point_forecast_case):
        """Test point forecast metrics (MAE, MAPE, RMSE, SMAPE, PoissonLoss, TweedieLoss)."""  # noqa: E501
        point_metrics = [MAE, MAPE, RMSE, SMAPE, PoissonLoss, TweedieLoss]

        for metric_class in point_metrics:
            metric = metric_class()
            self._test_integration_metrics(
                metric, point_forecast_case["y_pred"], point_forecast_case["y"]
            )  # noqa: E501
            self._test_reduction_modes(
                metric_class, point_forecast_case["y_pred"], point_forecast_case["y"]
            )  # noqa: E501

    def test_quantile_forecast_metrics(self, quantile_forecast_case):
        """Test quantile forecast metrics (QuantileLoss)."""
        metric = QuantileLoss(quantiles=quantile_forecast_case["quantiles"])
        self._test_integration_metrics(
            metric, quantile_forecast_case["y_pred"], quantile_forecast_case["y"]
        )  # noqa: E501

    def test_normal_distribution_metrics(self, normal_distribution_case):
        """Test normal distribution metrics (NormalDistributionLoss)."""
        metric = NormalDistributionLoss()
        # necessary step to get the dimensions for map_x_to_distribution
        normal_distribution_case["y_pred"] = metric.rescale_parameters(
            normal_distribution_case["y_pred"],
            target_scale=normal_distribution_case["x"]["target_scale"],
            encoder=TorchNormalizer(),
        )
        self._test_integration_metrics(
            metric, normal_distribution_case["y_pred"], normal_distribution_case["y"]
        )  # noqa: E501
