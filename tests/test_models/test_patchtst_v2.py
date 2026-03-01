import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data._tslib_data_module import TslibDataModule
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
from pytorch_forecasting.models.patchtst._patchtst_v2 import PatchTST


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
    "d_model, n_heads, e_layers, patch_len, stride",
    [
        (16, 2, 1, 8, 4),  # Small valid config
        (32, 4, 2, 16, 8),  # Medium config
    ],
)
def test_patchtst_init(d_model, n_heads, e_layers, patch_len, stride, sample_dataset):
    """Test PatchTST initialization with varying depths and patch sizes."""
    dm = sample_dataset["data_module"]
    metadata = dm.metadata
    loss = MAE()

    model = PatchTST(
        loss=loss,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        patch_len=patch_len,
        stride=stride,
        metadata=metadata,
    )

    assert model.d_model == d_model
    assert model.patch_len == patch_len
    assert model.n_quantiles is None


def test_patchtst_forward(sample_dataset):
    """Test standard point-forecasting forward pass of PatchTST."""
    dm = sample_dataset["data_module"]
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    model = PatchTST(
        loss=MAE(),
        d_model=16,
        n_heads=2,
        e_layers=1,
        patch_len=8,
        stride=4,
        metadata=dm.metadata,
    )

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    assert output["prediction"].shape[0] == dm.batch_size
    assert output["prediction"].shape[1] == dm.metadata["prediction_length"]
    assert output["prediction"].shape[2] == len(sample_dataset["time_series"].target)


def test_quantile_loss_output(sample_dataset):
    """Test PatchTST output shape specifically when using quantile loss."""
    dm = sample_dataset["data_module"]
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    quantiles = [0.1, 0.5, 0.9]

    model = PatchTST(
        loss=QuantileLoss(quantiles=quantiles),
        d_model=16,
        n_heads=2,
        e_layers=1,
        patch_len=8,
        stride=4,
        logging_metrics=[SMAPE(), MAE()],
        metadata=dm.metadata,
    )

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    pred = output["prediction"]

    # Expected shape for quantile: (batch_size, prediction_length, n_quantiles)
    # when target_dim is 1
    assert pred.ndim == 3
    assert pred.shape[-1] == len(quantiles)
    assert pred.shape[1] == dm.metadata["prediction_length"]
    assert pred.shape[0] == dm.batch_size


def test_univariate_forecast():
    """Test univariate forecasting specifically (no covariates) with PatchTST."""
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

    model = PatchTST(
        loss=MAE(),
        d_model=16,
        n_heads=2,
        e_layers=1,
        patch_len=8,
        stride=4,
        metadata=dm.metadata,
    )

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    assert output["prediction"].shape[0] == dm.batch_size
    assert output["prediction"].shape[1] == dm.metadata["prediction_length"]
    assert output["prediction"].shape[2] == len(ts.target)
