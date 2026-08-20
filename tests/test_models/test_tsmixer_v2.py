import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import TslibDataModule
from pytorch_forecasting.metrics import MAE, QuantileLoss, SMAPE
from pytorch_forecasting.models.tsmixer._tsmixer_v2 import TSMixer


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


@pytest.fixture
def model_with_logging_metrics(sample_dataset):
    """TSMixer instance used to test BaseModel logging_metrics registration."""
    dm = sample_dataset["data_module"]
    with pytest.warns(UserWarning):
        model = TSMixer(
            loss=MAE(),
            logging_metrics=[SMAPE(), MAE()],
            metadata=dm.metadata,
        )
    return model


@pytest.mark.parametrize(
    "d_model, e_layers, dropout",
    [
        (32, 1, 0.0),
        (64, 2, 0.1),
    ],
)
def test_tsmixer_init(d_model, e_layers, dropout, sample_dataset):
    """Test TSMixer initialization."""

    dm = sample_dataset["data_module"]

    model = TSMixer(
        loss=MAE(),
        d_model=d_model,
        e_layers=e_layers,
        dropout=dropout,
        metadata=dm.metadata,
    )

    assert model.d_model == d_model
    assert model.e_layers == e_layers
    assert model.dropout == dropout
    assert model.n_quantiles is None


def test_tsmixer_forward(sample_dataset):
    """Test forward pass of TSMixer."""

    dm = sample_dataset["data_module"]

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    metadata = dm.metadata

    model = TSMixer(
        loss=MAE(),
        d_model=32,
        e_layers=2,
        dropout=0.1,
        metadata=metadata,
    )

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    assert output["prediction"].shape[0] == dm.batch_size
    assert output["prediction"].shape[1] == metadata["prediction_length"]


def test_quantile_loss_output(sample_dataset):
    """Test TSMixer output shape with quantile loss."""

    dm = sample_dataset["data_module"]

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    metadata = dm.metadata

    quantiles = [0.1, 0.5, 0.9]

    model = TSMixer(
        loss=QuantileLoss(quantiles=quantiles),
        d_model=32,
        e_layers=2,
        dropout=0.1,
        logging_metrics=[SMAPE(), MAE()],
        metadata=metadata,
    )

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    pred = output["prediction"]
    assert pred.shape == (
        dm.batch_size,
        metadata["prediction_length"],
        len(quantiles),
    )


def test_univariate_forecast():
    """Test univariate forecasting with TSMixer."""

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

    model = TSMixer(
        loss=MAE(),
        d_model=32,
        e_layers=1,
        dropout=0.1,
        metadata=metadata,
    )

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    assert output["prediction"].shape == (
        dm.batch_size,
        metadata["prediction_length"],
        1,
    )


def test_logging_metrics_is_module_list(model_with_logging_metrics):
    """logging_metrics must be registered as nn.ModuleList so .to() propagates."""
    assert isinstance(model_with_logging_metrics.logging_metrics, nn.ModuleList)


def test_logging_metrics_device_propagation(model_with_logging_metrics):
    """Metric state tensors must follow the model when moved to a different device."""
    model_with_logging_metrics.to("meta")
    for metric in model_with_logging_metrics.logging_metrics:
        for state_name in metric._defaults:
            val = getattr(metric, state_name)
            if isinstance(val, torch.Tensor):
                assert val.device.type == "meta"
