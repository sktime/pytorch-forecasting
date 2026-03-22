"""Tests for DecoderMLP v2 implementation."""

import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data._tslib_data_module import TslibDataModule
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
from pytorch_forecasting.models.mlp._decodermlp_v2 import DecoderMLP


@pytest.fixture
def sample_dataset():
    n_samples = 100
    n_series = 3
    time_idx = np.arange(n_samples)
    series_data = []
    for i in range(n_series):
        trend = 0.1 * time_idx
        seasonality = 10 * np.sin(2 * np.pi * time_idx / 20)
        noise = np.random.normal(0, 1, n_samples)
        values = trend + seasonality + noise
        series = pd.DataFrame(
            {
                "time_idx": time_idx,
                "series_id": i,
                "value": values,
                "feat1": np.random.normal(0, 1, n_samples),
                "feat2": np.random.normal(0, 1, n_samples),
            }
        )
        series_data.append(series)
    data = pd.concat(series_data).reset_index(drop=True)
    ts = TimeSeries(
        data,
        time="time_idx",
        group=["series_id"],
        target=["value"],
        num=["feat1", "feat2"],
        cat=[],
        known=["time_idx"],
        unknown=["value", "feat1", "feat2"],
    )
    dm = TslibDataModule(ts, context_length=16, prediction_length=4, batch_size=4)
    dm.setup()
    return {"data_module": dm, "time_series": ts}


def test_decodermlp_init(sample_dataset):
    dm = sample_dataset["data_module"]
    model = DecoderMLP(
        loss=MAE(), hidden_size=64, n_hidden_layers=2, metadata=dm.metadata
    )
    assert model.hidden_size == 64
    assert model.n_hidden_layers == 2
    assert model.n_quantiles is None


def test_decodermlp_forward(sample_dataset):
    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]
    model = DecoderMLP(loss=MAE(), metadata=dm.metadata)
    with torch.no_grad():
        output = model(batch)
    assert "prediction" in output
    assert output["prediction"].shape[0] == dm.batch_size
    assert output["prediction"].shape[1] == dm.metadata["prediction_length"]


def test_decodermlp_quantile_loss(sample_dataset):
    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]
    quantiles = [0.1, 0.5, 0.9]
    model = DecoderMLP(loss=QuantileLoss(quantiles=quantiles), metadata=dm.metadata)
    with torch.no_grad():
        output = model(batch)
    assert output["prediction"].shape[-1] == len(quantiles)
    assert output["prediction"].shape[1] == dm.metadata["prediction_length"]


def test_decodermlp_no_nan(sample_dataset):
    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]
    model = DecoderMLP(loss=MAE(), metadata=dm.metadata)
    with torch.no_grad():
        output = model(batch)
    assert not torch.isnan(output["prediction"]).any()


def test_decodermlp_univariate():
    n_samples = 100
    time_idx = np.arange(n_samples)
    values = np.sin(2 * np.pi * time_idx / 20) + np.random.normal(0, 0.1, n_samples)
    series = pd.DataFrame({"time_idx": time_idx, "series_id": 0, "value": values})
    ts = TimeSeries(
        series,
        time="time_idx",
        group=["series_id"],
        target=["value"],
        num=[],
        cat=[],
        known=["time_idx"],
        unknown=["value"],
    )
    dm = TslibDataModule(ts, context_length=16, prediction_length=4, batch_size=4)
    dm.setup()
    model = DecoderMLP(loss=MAE(), metadata=dm.metadata)
    batch = next(iter(dm.train_dataloader()))[0]
    with torch.no_grad():
        output = model(batch)
    assert "prediction" in output
    assert output["prediction"].shape[0] == dm.batch_size
    assert output["prediction"].shape[1] == dm.metadata["prediction_length"]
