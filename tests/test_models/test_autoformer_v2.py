import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import TslibDataModule
from pytorch_forecasting.metrics import MAE, SMAPE
from pytorch_forecasting.models.autoformer._autoformer_v2 import Autoformer


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
    """Autoformer instance used to test BaseModel logging_metrics registration."""
    dm = sample_dataset["data_module"]
    with pytest.warns(UserWarning):
        model = Autoformer(
            loss=MAE(),
            logging_metrics=[SMAPE(), MAE()],
            metadata=dm.metadata,
            hidden_size=16,
            n_heads=2,
            e_layers=1,
            d_layers=1,
            d_ff=32,
        )
    return model


@pytest.mark.parametrize(
    "hidden_size, n_heads, e_layers, d_layers",
    [
        (16, 2, 1, 1),
        (32, 4, 2, 1),
    ],
)
def test_autoformer_init(hidden_size, n_heads, e_layers, d_layers, sample_dataset):
    """Test Autoformer initialization."""
    dm = sample_dataset["data_module"]
    metadata = dm.metadata
    loss = MAE()
    model = Autoformer(
        loss=loss,
        hidden_size=hidden_size,
        n_heads=n_heads,
        e_layers=e_layers,
        d_layers=d_layers,
        metadata=metadata,
    )

    assert model.hidden_size == hidden_size
    assert model.n_heads == n_heads
    assert model.e_layers == e_layers
    assert model.d_layers == d_layers
    assert model.n_quantiles is None


def test_univariate_forecast():
    """Test univariate forecasting with Autoformer."""
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

    model = Autoformer(
        loss=MAE(),
        hidden_size=16,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        d_ff=32,
        metadata=metadata,
    )

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    assert output["prediction"].shape[0] == dm.batch_size
    assert output["prediction"].shape[1] == metadata["prediction_length"]
    assert output["prediction"].shape[2] == metadata["n_features"]["target"]


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
