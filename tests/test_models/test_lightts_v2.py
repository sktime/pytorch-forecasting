import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data._tslib_data_module import TslibDataModule
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
from pytorch_forecasting.models.lightts._lightts_v2 import LightTS


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing using v2."""
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


@pytest.mark.parametrize(
    "d_model, chunk_size",
    [
        (64, 2),
        (128, 4),
    ],
)
def test_lightts_init(d_model, chunk_size, sample_dataset):
    """Test LightTS initialization."""

    dm = sample_dataset["data_module"]
    metadata = dm.metadata

    model = LightTS(
        loss=MAE(), d_model=d_model, chunk_size=chunk_size, metadata=metadata
    )

    assert model.d_model == d_model
    assert model.chunk_size == chunk_size
    assert model.n_quantiles is None


def test_lightts_forward(sample_dataset):
    """Test forward pass of LightTS."""

    dm = sample_dataset["data_module"]
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]
    metadata = dm.metadata

    model = LightTS(loss=MAE(), d_model=64, chunk_size=4, metadata=metadata)

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    assert output["prediction"].shape[0] == dm.batch_size
    assert output["prediction"].shape[1] == metadata["prediction_length"]


def test_quantile_loss_output(sample_dataset):
    """Test LightTS output shape with quantile loss."""

    dm = sample_dataset["data_module"]
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]
    metadata = dm.metadata

    quantiles = [0.1, 0.5, 0.9]
    model = LightTS(
        loss=QuantileLoss(quantiles=quantiles),
        d_model=128,
        chunk_size=4,
        logging_metrics=[SMAPE(), MAE()],
        metadata=metadata,
    )

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    pred = output["prediction"]
    assert pred.ndim == 3
    assert pred.shape[-1] == len(quantiles)
    assert pred.shape[1] == metadata["prediction_length"]


def test_univariate_forecast():
    """Test univariate forecasting with LightTS."""

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
    metadata = dm.metadata

    model = LightTS(loss=MAE(), d_model=64, chunk_size=4, metadata=metadata)

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    assert output["prediction"].shape[0] == dm.batch_size
    assert output["prediction"].shape[1] == metadata["prediction_length"]
