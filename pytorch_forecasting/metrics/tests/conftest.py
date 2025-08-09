import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn.utils import rnn

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, TorchNormalizer

BATCH_SIZE = 4
PREDICTION_LENGTH = 10


@pytest.fixture(scope="module")
def point_forecast():
    """Prepare a point forecast dataset, for testing point forecast metrics."""

    torch.manual_seed(42)
    np.random.seed(42)

    x = {
        "target_scale": torch.randn(BATCH_SIZE, 2),
    }
    y = (torch.randn(BATCH_SIZE, PREDICTION_LENGTH), None)
    y_pred = torch.randn(BATCH_SIZE, PREDICTION_LENGTH)

    test_cases = {}

    test_cases["standard"] = {
        "x": x,
        "y": y,
        "y_pred": y_pred,
    }
    # each batch has a different prediction length
    lengths = torch.tensor(
        [
            PREDICTION_LENGTH,
            PREDICTION_LENGTH - 2,
            PREDICTION_LENGTH - 4,
            PREDICTION_LENGTH - 6,
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

    return test_cases


@pytest.fixture(scope="module")
def quantile_forecast():
    """Prepare a quantile forecast dataset, for testing quantile metrics."""

    torch.manual_seed(42)
    np.random.seed(42)

    x = {
        "target_scale": torch.randn(BATCH_SIZE, 2),
    }

    y = (torch.randn(BATCH_SIZE, PREDICTION_LENGTH), None)

    quantiles = [0.1, 0.5, 0.9]  # Example quantiles
    y_pred = torch.randn(BATCH_SIZE, PREDICTION_LENGTH, len(quantiles))

    y_pred, _ = torch.sort(
        y_pred, dim=-1
    )  # sort along quantile dimension, to simulate perfect quantile distribution.  # noqa: E501

    test_cases = {}
    test_cases["standard"] = {"x": x, "y": y, "y_pred": y_pred, "quantiles": quantiles}

    lengths = torch.tensor(
        [
            PREDICTION_LENGTH,
            PREDICTION_LENGTH - 2,
            PREDICTION_LENGTH - 4,
            PREDICTION_LENGTH - 6,
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

    return test_cases


@pytest.fixture(scope="module")
def normal_distribution_forecast():
    """Prepare data for normal distribution loss metrics."""

    torch.manual_seed(42)
    np.random.seed(42)

    mean = 1.0
    std = 0.1
    normal_dist = torch.distributions.Normal(loc=mean, scale=std)
    y_actual = normal_dist.sample((BATCH_SIZE, PREDICTION_LENGTH))

    x = {
        "target_scale": torch.randn(BATCH_SIZE, 2),
    }

    y = (y_actual, None)

    # create random prediction paired with a normal distribution.
    y_pred = torch.stack(
        [
            torch.randn(BATCH_SIZE, PREDICTION_LENGTH),
            torch.abs(torch.randn(BATCH_SIZE, PREDICTION_LENGTH)) + 0.1,
        ],
        dim=-1,
    )

    test_cases = {}
    test_cases["standard"] = {"x": x, "y": y, "y_pred": y_pred}
    lengths = torch.tensor(
        [
            PREDICTION_LENGTH,
            PREDICTION_LENGTH - 2,
            PREDICTION_LENGTH - 4,
            PREDICTION_LENGTH - 6,
        ]
    )
    y_packed = rnn.pack_padded_sequence(
        y[0], lengths, batch_first=True, enforce_sorted=False
    )

    test_cases["packed"] = {"x": x, "y": y_packed, "y_pred": y_pred}

    weights = torch.ones_like(y[0])
    y_weighted = (y[0], weights)
    test_cases["weighted"] = {"x": x, "y": y_weighted, "y_pred": y_pred}

    return test_cases


@pytest.fixture(scope="module")
def multivariate_normal_distribution_forecast():
    """Prepare data for multivariate normal distribution loss metrics."""

    torch.manual_seed(42)
    np.random.seed(42)

    n_targets = 2

    mean = torch.tensor([1.0, 1.0])
    std = torch.tensor([0.2, 0.1])
    cov_factor = torch.tensor([[0.0], [0.0]])

    multivar_normal_dist = torch.distributions.LowRankMultivariateNormal(
        loc=mean, cov_diag=std**2, cov_factor=cov_factor
    )

    multivar_normal_dist = multivar_normal_dist.sample((BATCH_SIZE, PREDICTION_LENGTH))[
        :, :, 0
    ]
    target_mean = multivar_normal_dist.mean(dim=1)
    target_std = multivar_normal_dist.std(dim=1)

    x = {
        "target_scale": torch.stack([target_mean, target_std], dim=1),
    }

    y = (multivar_normal_dist, None)

    mean = torch.randn(BATCH_SIZE, PREDICTION_LENGTH, n_targets)
    diag_vars = torch.abs(torch.randn(BATCH_SIZE, PREDICTION_LENGTH, n_targets)) + 0.1
    cov_factors = torch.randn(BATCH_SIZE, PREDICTION_LENGTH, n_targets * 1)  # rank = 1

    y_pred = torch.cat([mean, diag_vars, cov_factors], dim=-1)

    test_cases = {}
    test_cases["standard"] = {"x": x, "y": y, "y_pred": y_pred}
    lengths = torch.tensor(
        [
            PREDICTION_LENGTH,
            PREDICTION_LENGTH - 2,
            PREDICTION_LENGTH - 4,
            PREDICTION_LENGTH - 6,
        ]
    )
    y_packed = rnn.pack_padded_sequence(
        y[0], lengths, batch_first=True, enforce_sorted=False
    )

    test_cases["packed"] = {"x": x, "y": y_packed, "y_pred": y_pred}

    return test_cases


@pytest.fixture(scope="module")
def negative_binomial_distribution_forecast():
    """Prepare data for negative binomial distribution loss metric."""

    torch.manual_seed(42)
    np.random.seed(42)

    mean = 100.0
    shape = 1.0

    total_count = 1.0 / shape
    probs = mean / (mean + total_count)

    neg_bin_dist = torch.distributions.NegativeBinomial(
        total_count=total_count, probs=probs
    )

    neg_bin_target = neg_bin_dist.sample((BATCH_SIZE, PREDICTION_LENGTH))
    target_mean = neg_bin_target.mean(dim=1)
    target_std = neg_bin_target.std(dim=1)

    x = {"target_scale": torch.stack([target_mean, target_std], dim=1)}

    y = (neg_bin_target, None)

    y_pred = torch.stack(
        [
            torch.randn(BATCH_SIZE, PREDICTION_LENGTH),
            torch.abs(torch.randn(BATCH_SIZE, PREDICTION_LENGTH)) + 0.1,
        ],
        dim=-1,
    )

    test_cases = {}
    test_cases["standard"] = {"x": x, "y": y, "y_pred": y_pred}
    lengths = torch.tensor(
        [
            PREDICTION_LENGTH,
            PREDICTION_LENGTH - 2,
            PREDICTION_LENGTH - 4,
            PREDICTION_LENGTH - 6,
        ]
    )
    y_packed = rnn.pack_padded_sequence(
        y[0], lengths, batch_first=True, enforce_sorted=False
    )

    test_cases["packed"] = {"x": x, "y": y_packed, "y_pred": y_pred}

    weights = torch.ones_like(y[0])
    y_weighted = (y[0], weights)
    test_cases["weighted"] = {"x": x, "y": y_weighted, "y_pred": y_pred}

    return test_cases


@pytest.fixture(scope="module")
def log_normal_distribution_forecast():
    """Prepare data for log normal distribution loss metrics"""

    torch.manual_seed(42)
    np.random.seed(42)

    mean = 2.0
    std = 0.2

    log_normal_dist = torch.distributions.LogNormal(mean, std)
    log_normal_target = log_normal_dist.sample((BATCH_SIZE, PREDICTION_LENGTH))

    target_mean = log_normal_target.mean(dim=1)
    target_std = log_normal_target.std(dim=1)

    x = {"target_scale": torch.stack([target_mean, target_std], dim=1)}

    y = (log_normal_target, None)

    # create random prediction paired with a log normal distribution.
    y_pred = torch.stack(
        [
            torch.randn(BATCH_SIZE, PREDICTION_LENGTH),
            torch.abs(torch.randn(BATCH_SIZE, PREDICTION_LENGTH)) + 0.1,
        ],
        dim=-1,
    )

    test_cases = {}
    test_cases["standard"] = {"x": x, "y": y, "y_pred": y_pred}

    weights = torch.ones_like(y[0])
    y_weighted = (y[0], weights)
    test_cases["weighted"] = {"x": x, "y": y_weighted, "y_pred": y_pred}

    return test_cases


@pytest.fixture(scope="module")
def beta_distribution_forecast():
    """Prepare data for beta distribution loss metrics."""

    torch.manual_seed(42)
    np.random.seed(42)

    initial_mean = 2.0
    initial_shape = 5.0

    beta_dist = torch.distributions.Beta(initial_mean, initial_shape)
    beta_target = beta_dist.sample((BATCH_SIZE, PREDICTION_LENGTH))

    target_mean = beta_target.mean(dim=1)
    target_std = beta_target.std(dim=1)

    x = {"target_scale": torch.stack([target_mean, target_std], dim=1)}

    y = (beta_target, None)

    # create random prediction paired with a beta distribution.
    y_pred = torch.stack(
        [
            torch.abs(torch.randn(BATCH_SIZE, PREDICTION_LENGTH)) + 0.1,
            torch.abs(torch.randn(BATCH_SIZE, PREDICTION_LENGTH)) + 0.1,
        ],
        dim=-1,
    )

    test_cases = {}
    test_cases["standard"] = {"x": x, "y": y, "y_pred": y_pred}

    lengths = torch.tensor(
        [
            PREDICTION_LENGTH,
            PREDICTION_LENGTH - 2,
            PREDICTION_LENGTH - 4,
            PREDICTION_LENGTH - 6,
        ]
    )
    y_packed = rnn.pack_padded_sequence(
        y[0], lengths, batch_first=True, enforce_sorted=False
    )
    test_cases["packed"] = {"x": x, "y": y_packed, "y_pred": y_pred}
    weights = torch.ones_like(y[0])
    y_weighted = (y[0], weights)
    test_cases["weighted"] = {"x": x, "y": y_weighted, "y_pred": y_pred}

    return test_cases


@pytest.fixture(scope="module")
def implicit_quantile_network_distribution_forecast():
    """Prepare data for implicit quantile network distribution loss metrics."""

    torch.manual_seed(42)
    np.random.seed(42)

    target = torch.randn(BATCH_SIZE, PREDICTION_LENGTH)
    target_mean = target.mean(dim=1)
    target_std = target.std(dim=1)

    x = {"target_scale": torch.stack([target_mean, target_std], dim=1)}

    y = (target, None)

    output_size = 5

    y_pred = torch.randn(BATCH_SIZE, PREDICTION_LENGTH, output_size)

    test_cases = {}
    test_cases["standard"] = {"x": x, "y": y, "y_pred": y_pred}
    lengths = torch.tensor(
        [
            PREDICTION_LENGTH,
            PREDICTION_LENGTH - 2,
            PREDICTION_LENGTH - 4,
            PREDICTION_LENGTH - 6,
        ]
    )
    y_packed = rnn.pack_padded_sequence(
        y[0], lengths, batch_first=True, enforce_sorted=False
    )
    test_cases["packed"] = {"x": x, "y": y_packed, "y_pred": y_pred}

    weights = torch.ones_like(y[0])
    y_weighted = (y[0], weights)
    test_cases["weighted"] = {"x": x, "y": y_weighted, "y_pred": y_pred}

    return test_cases


@pytest.fixture(scope="module")
def mqf2_distribution_forecast():
    """Prepare data for MQF2 distribution loss metrics."""

    torch.manual_seed(42)
    np.random.seed(42)

    hidden_size = 4  # default hidden size for MQF2.

    target = torch.randn(BATCH_SIZE, PREDICTION_LENGTH)
    target_mean = target.mean(dim=1)
    target_std = target.std(dim=1)

    x = {"target_scale": torch.stack([target_mean, target_std], dim=1)}

    y = (target, None)

    y_pred = torch.randn(BATCH_SIZE, PREDICTION_LENGTH, hidden_size)

    test_cases = {}
    test_cases["standard"] = {"x": x, "y": y, "y_pred": y_pred}
    lengths = torch.tensor(
        [
            PREDICTION_LENGTH,
            PREDICTION_LENGTH - 2,
            PREDICTION_LENGTH - 4,
            PREDICTION_LENGTH - 6,
        ]
    )
    y_packed = rnn.pack_padded_sequence(
        y[0], lengths, batch_first=True, enforce_sorted=False
    )
    test_cases["packed"] = {"x": x, "y": y_packed, "y_pred": y_pred}

    weights = torch.ones_like(y[0])
    y_weighted = (y[0], weights)
    test_cases["weighted"] = {"x": x, "y": y_weighted, "y_pred": y_pred}

    return test_cases


@pytest.fixture(scope="module")
def classification_forecast():
    """Prepare data for classification forecast metrics."""

    torch.manual_seed(42)
    np.random.seed(42)

    n_classes = 3

    target = torch.randint(0, n_classes, size=(BATCH_SIZE, PREDICTION_LENGTH))

    y_pred = torch.rand(BATCH_SIZE, PREDICTION_LENGTH, n_classes)

    x = {}

    y = (target, None)

    y_pred = torch.rand(BATCH_SIZE, PREDICTION_LENGTH, n_classes)  # noqa: E501

    test_cases = {}

    test_cases["standard"] = {"x": x, "y": y, "y_pred": y_pred}

    lengths = torch.tensor(
        [
            PREDICTION_LENGTH,
            (PREDICTION_LENGTH) - 2,
            (PREDICTION_LENGTH) - 4,
            (PREDICTION_LENGTH) - 6,
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

    return test_cases
