import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data._tslib_data_module import TslibDataModule
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
from pytorch_forecasting.models.dlinear._dlinear_v2 import DLinear


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
    "moving_average, individual",
    [
        (5, False),
        (25, True),
    ],
)
def test_dlinear_init(moving_average, individual, sample_dataset):
    """Test DLinear initialization."""

    dm = sample_dataset["data_module"]

    metadata = dm.metadata
    loss = MAE()
    model = DLinear(
        loss=loss, moving_avg=moving_average, individual=individual, metadata=metadata
    )

    assert model.moving_avg == moving_average
    assert model.individual == individual
    assert model.n_quantiles is None


def test_dlinear_forward(sample_dataset):
    """Test forward pass of DLinear."""

    dm = sample_dataset["data_module"]

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    metadata = dm.metadata

    model = DLinear(loss=MAE(), moving_avg=5, individual=True, metadata=metadata)

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    assert output["prediction"].shape[0] == dm.batch_size
    assert output["prediction"].shape[1] == metadata["prediction_length"]


def test_quantile_loss_output(sample_dataset):
    """Test DLinear output shape with quantile loss."""

    dm = sample_dataset["data_module"]

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    metadata = dm.metadata

    quantiles = [0.1, 0.5, 0.9]

    model = DLinear(
        loss=QuantileLoss(quantiles=quantiles),
        moving_avg=5,
        individual=True,
        logging_metrics=[SMAPE(), MAE()],
        metadata=metadata,
    )

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    pred = output["prediction"]
    assert pred.ndim == 4
    assert pred.shape[-1] == len(quantiles)
    assert pred.shape[1] == metadata["prediction_length"]


def test_univariate_forecast():
    """Test univariate forecasting with DLinear."""

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

    model = DLinear(loss=MAE(), moving_avg=5, individual=False, metadata=metadata)

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    assert output["prediction"].shape[0] == dm.batch_size
    assert output["prediction"].shape[1] == metadata["prediction_length"]
