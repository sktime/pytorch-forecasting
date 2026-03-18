import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data._tslib_data_module import TslibDataModule
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_forecasting.models.nhits._nhits_v2 import NHiTS


@pytest.fixture
def sample_dataset():
    """Create a compact synthetic dataset for NHiTS v2 tests."""
    n_samples = 120
    n_series = 3

    time_idx = np.arange(n_samples)
    series_data = []
    for series_id in range(n_series):
        trend = 0.03 * time_idx
        seasonality = np.sin(2 * np.pi * time_idx / 24)
        noise = np.random.normal(0, 0.05, n_samples)
        target = trend + seasonality + noise

        series_data.append(
            pd.DataFrame(
                {
                    "time_idx": time_idx,
                    "series_id": series_id,
                    "target": target,
                    "known_1": np.cos(2 * np.pi * time_idx / 24),
                    "known_2": (time_idx % 12) / 12.0,
                    "unknown_1": np.random.normal(0, 1, n_samples),
                }
            )
        )

    data = pd.concat(series_data).reset_index(drop=True)

    ts = TimeSeries(
        data=data,
        time="time_idx",
        group=["series_id"],
        target=["target"],
        num=["known_1", "known_2", "unknown_1"],
        cat=[],
        known=["known_1", "known_2"],
        unknown=["unknown_1"],
        static=["series_id"],
    )

    dm = TslibDataModule(
        time_series_dataset=ts,
        context_length=16,
        prediction_length=4,
        batch_size=4,
    )
    dm.setup()

    return {"data_module": dm}


def test_nhits_v2_init(sample_dataset):
    """Model initializes with expected derived configuration."""
    dm = sample_dataset["data_module"]
    model = NHiTS(loss=MAE(), metadata=dm.metadata)

    assert model.context_length == dm.metadata["context_length"]
    assert model.prediction_length == dm.metadata["prediction_length"]
    assert model.n_stacks == 3
    assert model.n_quantiles is None


def test_nhits_v2_forward_shape(sample_dataset):
    """Forward pass returns correctly shaped predictions."""
    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))
    x, _ = batch

    model = NHiTS(loss=MAE(), hidden_size=32, n_blocks=[1, 1], metadata=dm.metadata)
    model.eval()

    with torch.no_grad():
        out = model(x)

    assert "prediction" in out
    prediction = out["prediction"]
    assert prediction.ndim == 3
    assert prediction.shape[0] == dm.batch_size
    assert prediction.shape[1] == dm.metadata["prediction_length"]
    assert prediction.shape[2] == 1


def test_nhits_v2_quantile_shape(sample_dataset):
    """Quantile loss produces [batch, prediction_length, n_quantiles] output."""
    dm = sample_dataset["data_module"]
    x, _ = next(iter(dm.train_dataloader()))

    quantiles = [0.1, 0.5, 0.9]
    model = NHiTS(
        loss=QuantileLoss(quantiles=quantiles),
        hidden_size=24,
        n_blocks=[1, 1],
        metadata=dm.metadata,
    )
    model.eval()

    with torch.no_grad():
        out = model(x)

    pred = out["prediction"]
    assert pred.ndim == 3
    assert pred.shape[1] == dm.metadata["prediction_length"]
    assert pred.shape[2] == len(quantiles)


def test_nhits_v2_step_train(sample_dataset):
    """Custom step returns training loss dictionary."""
    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))

    model = NHiTS(
        loss=MAE(),
        hidden_size=16,
        n_blocks=[1, 1],
        backcast_loss_ratio=0.2,
        metadata=dm.metadata,
    )
    model.train()

    result = model.step(batch, batch_idx=0, stage="train")

    assert "loss" in result
    assert torch.is_tensor(result["loss"])
