import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn.utils import rnn

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, TorchNormalizer


@pytest.fixture(scope="module")
def point_forecast():
    """Prepare a point forecast dataset, for testing point forecast metrics."""

    torch.manual_seed(42)
    np.random.seed(42)
    batch_size, timesteps = 4, 20
    prediction_length = timesteps // 2

    x = {
        "target_scale": torch.randn(batch_size, 2),
    }
    y = (torch.randn(batch_size, prediction_length), None)
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

    weights = torch.ones_like(y[0])
    y_weighted = (y[0], weights)
    test_cases["weighted"] = {
        "x": x,
        "y": y_weighted,
        "y_pred": y_pred,
    }

    return {"test_cases": test_cases}


@pytest.fixture(scope="module")
def quantile_forecast():
    """Prepare a quantile forecast dataset, for testing quantile metrics."""

    torch.manual_seed(42)
    np.random.seed(42)

    batch_size, timesteps = 4, 20
    prediction_length = timesteps // 2

    x = {
        "target_scale": torch.randn(batch_size, 2),
    }

    y = (torch.randn(batch_size, prediction_length), None)

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

    return {"test_cases": test_cases}


@pytest.fixture(scope="module")
def normal_distribution_forecast():
    """Prepare data for normal distribution loss metrics."""

    torch.manual_seed(42)
    np.random.seed(42)

    batch_size, timesteps = 4, 20
    prediction_length = timesteps // 2

    mean = 1.0
    std = 0.1
    normal_dist = torch.distributions.Normal(loc=mean, scale=std)
    y_actual = normal_dist.sample((batch_size, prediction_length))

    x = {
        "target_scale": torch.randn(batch_size, 2),
    }

    y = (y_actual, None)

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

    return {"test_cases": test_cases}


@pytest.fixture(scope="module")
def multivariate_normal_distribution_forecast():
    """Prepare data for multivariate normal distribution loss metrics."""

    torch.manual_seed(42)
    np.random.seed(42)

    batch_size, timesteps = 4, 20
    prediction_length = timesteps // 2

    n_targets = 2

    mean = torch.tensor([1.0, 1.0])
    std = torch.tensor([0.2, 0.1])
    cov_factor = torch.tensor([[0.0], [0.0]])

    multivar_normal_dist = torch.distributions.LowRankMultivariateNormal(
        loc=mean, cov_diag=std**2, cov_factor=cov_factor
    )

    x = {
        "target_scale": torch.randn(batch_size, 2),
    }

    multivar_normal_dist = multivar_normal_dist.sample((batch_size, prediction_length))[
        :, :, 0
    ]

    y = (multivar_normal_dist, None)

    mean = torch.randn(batch_size, prediction_length, n_targets)
    diag_vars = torch.abs(torch.randn(batch_size, prediction_length, n_targets)) + 0.1
    cov_factors = torch.randn(batch_size, prediction_length, n_targets * 1)  # rank = 1

    y_pred = torch.cat([mean, diag_vars, cov_factors], dim=-1)

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

    return {"test_cases": test_cases}


@pytest.fixture(scope="module")
def negative_binomial_distribution_forecast():
    """Prepare data for negative binomial distribution loss metric."""

    torch.manual_seed(42)
    np.random.seed(42)

    batch_size, timesteps = 4, 20
    prediction_length = timesteps // 2

    mean = 100.0
    shape = 1.0

    total_count = 1.0 / shape
    probs = mean / (mean + total_count)

    neg_bin_dist = torch.distributions.NegativeBinomial(
        total_count=total_count, probs=probs
    )

    neg_bin_target = neg_bin_dist.sample((batch_size, prediction_length))
    target_mean = neg_bin_target.mean(dim=1, keepdim=True)
    target_std = neg_bin_target.std(dim=1, keepdim=True)

    x = {"target_scale": torch.cat([target_mean, target_std], dim=1)}

    y = (neg_bin_target, None)

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

    return {"test_cases": test_cases}


@pytest.fixture(scope="module")
def log_normal_distribution_forecast():
    """Prepare data for log normal distribution loss metrics"""

    torch.manual_seed(42)
    np.random.seed(42)

    batch_size, timesteps = 4, 20
    prediction_length = timesteps // 2

    mean = 2.0
    std = 0.2
    log_normal_dist = torch.distributions.LogNormal(mean, std)
    log_normal_target = log_normal_dist.sample((batch_size, timesteps)).numpy()

    df = pd.DataFrame(
        {
            "group_id": np.repeat(np.arange(batch_size), timesteps),
            "time_idx": np.tile(np.arange(timesteps), batch_size),
            "target": log_normal_target.flatten(),
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

    # create random prediction paired with a log normal distribution.
    y_pred = torch.stack(
        [
            torch.randn(batch_size, prediction_length),
            torch.abs(torch.randn(batch_size, prediction_length)) + 0.1,
        ],
        dim=-1,
    )

    test_cases = {}
    test_cases["standard"] = {"x": x, "y": y, "y_pred": y_pred}

    weights = torch.ones_like(y[0])
    y_weighted = (y[0], weights)
    test_cases["weighted"] = {"x": x, "y": y_weighted, "y_pred": y_pred}

    return {"dataset": dataset, "test_cases": test_cases}


@pytest.fixture(scope="module")
def beta_distribution_forecast():
    """Prepare data for beta distribution loss metrics."""

    torch.manual_seed(42)
    np.random.seed(42)

    batch_size, timesteps = 4, 20
    prediction_length = timesteps // 2

    initial_mean = torch.tensor([0.1])
    initial_shape = torch.tensor([10])
    beta_dist = torch.distributions.Beta(initial_mean, initial_shape)
    beta_target = beta_dist.sample((batch_size, timesteps)).numpy()

    df = pd.DataFrame(
        {
            "group_id": np.repeat(np.arange(batch_size), timesteps),
            "time_idx": np.tile(np.arange(timesteps), batch_size),
            "target": beta_target.flatten(),
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

    # create random prediction paired with a beta distribution.
    y_pred = torch.stack(
        [
            torch.abs(torch.randn(batch_size, prediction_length)) + 0.1,
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


@pytest.fixture(scope="module")
def implicit_quantile_network_distribution_forecast():
    """Prepare data for implicit quantile network distribution loss metrics."""

    torch.manual_seed(42)
    np.random.seed(42)

    batch_size, timesteps = 4, 20
    prediction_length = timesteps // 2

    target = torch.randn(batch_size, timesteps)

    df = pd.DataFrame(
        {
            "group_id": np.repeat(np.arange(batch_size), timesteps),
            "time_idx": np.tile(np.arange(timesteps), batch_size),
            "target": target.flatten(),
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

    output_size = 5

    y_pred = torch.randn(batch_size, prediction_length, output_size)

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


@pytest.fixture(scope="module")
def mqf2_distribution_forecast():
    """Prepare data for MQF2 distribution loss metrics."""

    torch.manual_seed(42)
    np.random.seed(42)

    batch_size, timesteps = 4, 20
    prediction_length = timesteps // 2

    hidden_size = 4  # default hidden size for MQF2.

    target = torch.randn(batch_size, timesteps)

    df = pd.DataFrame(
        {
            "group_id": np.repeat(np.arange(batch_size), timesteps),
            "time_idx": np.tile(np.arange(timesteps), batch_size),
            "target": target.flatten(),
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

    y_pred = torch.randn(batch_size, prediction_length, hidden_size)

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


@pytest.fixture(scope="module")
def classification_forecast():
    """Prepare data for classification forecast metrics."""

    torch.manual_seed(42)
    np.random.seed(42)

    batch_size, timesteps = 4, 20

    n_classes = 3

    class_logits = np.random.randint(0, n_classes, size=(batch_size * timesteps))

    df = pd.DataFrame(
        {
            "group_id": np.repeat(np.arange(batch_size), timesteps),
            "time_idx": np.tile(np.arange(timesteps), batch_size),
            "target": class_logits,  # Integer class labels
        }
    )

    from pytorch_forecasting.data.encoders import NaNLabelEncoder

    dataset = TimeSeriesDataSet(
        data=df,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=timesteps // 2,
        max_prediction_length=timesteps // 2,
        target_normalizer=NaNLabelEncoder(),  # Use label encoder for categorical target
        add_relative_time_idx=True,
        add_encoder_length=True,
    )

    dataloader = dataset.to_dataloader(batch_size=batch_size, shuffle=True)
    x, y = next(iter(dataloader))

    prediction_length = timesteps // 2

    y_pred = torch.rand(batch_size, prediction_length, n_classes)  # noqa: E501

    test_cases = {}

    test_cases["standard"] = {"x": x, "y": y, "y_pred": y_pred}

    lengths = torch.tensor(
        [
            prediction_length,
            (prediction_length) - 2,
            (prediction_length) - 4,
            (prediction_length) - 6,
        ]
    )

    y_packed = rnn.pack_padded_sequence(
        y[0], lengths, batch_first=True, enforce_sorted=False
    )
    test_cases["packed"] = {
        "x": x,
        "y": (y_packed, y[1]),
        "y_pred": y_pred,
    }

    # Weighted case
    weights = torch.ones_like(y[0])
    y_weighted = (y[0], weights)
    test_cases["weighted"] = {
        "x": x,
        "y": y_weighted,
        "y_pred": y_pred,
    }

    return {"dataset": dataset, "test_cases": test_cases}
