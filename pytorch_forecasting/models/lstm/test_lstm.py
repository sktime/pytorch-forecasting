"""Tests for LSTMModel - univariate and multivariate forecasting."""

import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    MultiNormalizer,
    TorchNormalizer,
)
from pytorch_forecasting.metrics import MAE, MultiLoss
from pytorch_forecasting.models.lstm import LSTMModel


def _make_data(n_series=3, timesteps=10):
    return pd.DataFrame(
        dict(
            value=np.random.rand(n_series * timesteps),
            group=np.repeat(np.arange(n_series), timesteps),
            time_idx=np.tile(np.arange(timesteps), n_series),
        )
    )


def _make_multi_target_data(n_series=3, timesteps=10):
    return pd.DataFrame(
        dict(
            target1=np.random.rand(n_series * timesteps),
            target2=np.random.rand(n_series * timesteps),
            group=np.repeat(np.arange(n_series), timesteps),
            time_idx=np.tile(np.arange(timesteps), n_series),
        )
    )


def _get_batch(dataset, train=True):
    return next(iter(dataset.to_dataloader(train=train, batch_size=6, num_workers=0)))


@pytest.fixture
def univariate_dataset():
    data = _make_data()
    return TimeSeriesDataSet(
        data,
        group_ids=["group"],
        target="value",
        time_idx="time_idx",
        min_encoder_length=5,
        max_encoder_length=5,
        min_prediction_length=2,
        max_prediction_length=2,
        time_varying_unknown_reals=["value"],
        target_normalizer=EncoderNormalizer(),
    )


@pytest.fixture
def multivariate_dataset():
    data = _make_multi_target_data()
    return TimeSeriesDataSet(
        data,
        group_ids=["group"],
        target=["target1", "target2"],
        time_idx="time_idx",
        min_encoder_length=5,
        max_encoder_length=5,
        min_prediction_length=2,
        max_prediction_length=2,
        time_varying_unknown_reals=["target1", "target2"],
        target_normalizer=MultiNormalizer([EncoderNormalizer(), TorchNormalizer()]),
    )


def test_univariate_train_shape(univariate_dataset):
    model = LSTMModel.from_dataset(univariate_dataset, n_layers=1, hidden_size=8)
    x, _ = _get_batch(univariate_dataset, train=True)
    model.train()
    assert model(x)["prediction"].shape == torch.Size([6, 2, 1])


def test_univariate_eval_shape(univariate_dataset):
    model = LSTMModel.from_dataset(univariate_dataset, n_layers=1, hidden_size=8)
    x, _ = _get_batch(univariate_dataset, train=False)
    model.eval()
    with torch.no_grad():
        assert model(x)["prediction"].shape == torch.Size([6, 2, 1])


def test_multivariate_train_shape(multivariate_dataset):
    """Regression test for #1582 - multi-target caused a shape mismatch."""
    model = LSTMModel.from_dataset(
        multivariate_dataset,
        n_layers=1,
        hidden_size=8,
        loss=MultiLoss([MAE(), MAE()]),
    )
    x, _ = _get_batch(multivariate_dataset, train=True)
    model.train()
    out = model(x)["prediction"]
    assert len(out) == 2
    assert out[0].shape == torch.Size([6, 2, 1])
    assert out[1].shape == torch.Size([6, 2, 1])


def test_multivariate_eval_shape(multivariate_dataset):
    """Autoregressive eval mode should work for multiple targets."""
    model = LSTMModel.from_dataset(
        multivariate_dataset,
        n_layers=1,
        hidden_size=8,
        loss=MultiLoss([MAE(), MAE()]),
    )
    x, _ = _get_batch(multivariate_dataset, train=False)
    model.eval()
    with torch.no_grad():
        out = model(x)["prediction"]
    assert len(out) == 2
    assert out[0].shape == torch.Size([6, 2, 1])


def test_from_dataset_infers_multiloss(multivariate_dataset):
    model = LSTMModel.from_dataset(multivariate_dataset, n_layers=1, hidden_size=8)
    assert isinstance(model.loss, MultiLoss)
    assert len(model.loss) == 2
