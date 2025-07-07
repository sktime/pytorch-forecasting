from inspect import isclass
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn.utils import rnn

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, TorchNormalizer


@pytest.fixture(scope="module")
def prepare_point_forecast():
    """Prepare a point forecast dataset, for testing point forecast metrics."""

    torch.manual_seed(42)
    np.random.seed(42)
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

    torch.manual_seed(42)
    np.random.seed(42)

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

    torch.manual_seed(42)
    np.random.seed(42)

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
