import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data._tslib_data_module import TslibDataModule
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_forecasting.models.nlinear._nlinear_v2 import NLinear


def _clone_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Clone tensor values in a batch dict for safe mutation in tests."""
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@pytest.fixture
def sample_dataset():
    """Create a target-only sample dataset for testing NLinear v2."""
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
            }
        )
        series_data.append(series)

    data = pd.concat(series_data).reset_index(drop=True)

    ts = TimeSeries(
        data,
        time="time_idx",
        group=["series_id"],
        target=["value"],
        num=[],
        cat=[],
        known=[],
        unknown=["value"],
        static=[],
    )

    dm = TslibDataModule(ts, context_length=16, prediction_length=4, batch_size=4)
    dm.setup()

    return {"data_module": dm, "time_series": ts}


def test_nlinear_init(sample_dataset):
    """Test NLinear initialization."""
    dm = sample_dataset["data_module"]
    metadata = dm.metadata

    model = NLinear(loss=MAE(), metadata=metadata)

    assert model.n_quantiles is None


def test_nlinear_forward(sample_dataset):
    """Test forward pass of NLinear."""
    dm = sample_dataset["data_module"]
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]
    metadata = dm.metadata

    model = NLinear(loss=MAE(), metadata=metadata)

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    pred = output["prediction"]
    assert pred.ndim == 3
    assert pred.shape[0] == dm.batch_size
    assert pred.shape[1] == metadata["prediction_length"]
    assert pred.shape[-1] == 1

    point_pred = model.to_prediction(output)
    assert point_pred.ndim == 2
    assert point_pred.shape[0] == dm.batch_size
    assert point_pred.shape[1] == metadata["prediction_length"]


def test_quantile_loss_output(sample_dataset):
    """Test NLinear output shape with quantile loss."""
    dm = sample_dataset["data_module"]
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]
    metadata = dm.metadata

    quantiles = [0.1, 0.5, 0.9]

    model = NLinear(
        loss=QuantileLoss(quantiles=quantiles),
        metadata=metadata,
    )

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    pred = output["prediction"]
    assert pred.ndim == 3
    assert pred.shape[-1] == len(quantiles)
    assert pred.shape[1] == metadata["prediction_length"]


def test_nlinear_requires_metadata():
    """Test NLinear requires metadata from a fitted datamodule."""
    with pytest.raises(ValueError, match="metadata"):
        NLinear(loss=MAE(), metadata=None)


def test_nlinear_rejects_multi_target_metadata():
    """Test NLinear rejects multi-target metadata."""
    n_samples = 100
    time_idx = np.arange(n_samples)
    series = pd.DataFrame(
        {
            "time_idx": time_idx,
            "series_id": 0,
            "value": np.sin(time_idx / 3.0),
            "value_2": np.cos(time_idx / 5.0),
        }
    )

    ts = TimeSeries(
        series,
        time="time_idx",
        group=["series_id"],
        target=["value", "value_2"],
        num=[],
        cat=[],
        known=[],
        unknown=["value", "value_2"],
        static=[],
    )
    dm = TslibDataModule(ts, context_length=16, prediction_length=4, batch_size=4)
    dm.setup()

    with pytest.raises(ValueError, match="single target"):
        NLinear(loss=MAE(), metadata=dm.metadata)


@pytest.mark.parametrize(
    ("feature_key", "shape"),
    [
        ("history_cont", (4, 16, 1)),
        ("history_cat", (4, 16, 1)),
        ("future_cont", (4, 4, 1)),
        ("future_cat", (4, 4, 1)),
        ("static_categorical_features", (4, 1, 1)),
        ("static_continuous_features", (4, 1, 1)),
    ],
)
def test_nlinear_rejects_unsupported_feature_tensors(
    feature_key, shape, sample_dataset
):
    """Test NLinear rejects non-empty unsupported feature tensors."""
    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]
    batch = _clone_batch(batch)
    batch[feature_key] = torch.ones(shape)

    model = NLinear(loss=MAE(), metadata=dm.metadata)

    with pytest.raises(ValueError, match="target-history-only input"):
        model(batch)


def test_nlinear_rejects_malformed_history_target_rank(sample_dataset):
    """Test NLinear rejects malformed target tensors."""
    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]
    batch = _clone_batch(batch)
    batch["history_target"] = batch["history_target"].squeeze(-1)

    model = NLinear(loss=MAE(), metadata=dm.metadata)

    with pytest.raises(ValueError, match="shape \\[batch, context_length, 1\\]"):
        model(batch)


def test_nlinear_rejects_context_length_mismatch(sample_dataset):
    """Test NLinear rejects target history with the wrong context length."""
    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]
    batch = _clone_batch(batch)
    batch["history_target"] = batch["history_target"][:, :-1, :]

    model = NLinear(loss=MAE(), metadata=dm.metadata)

    with pytest.raises(ValueError, match="context length"):
        model(batch)


def test_nlinear_applies_target_scale_when_present(sample_dataset):
    """Test NLinear preserves the optional target_scale transform pathway."""
    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]
    metadata = dm.metadata
    model = NLinear(loss=MAE(), metadata=metadata)

    with torch.no_grad():
        raw_output = model(batch)["prediction"]

    scaled_batch = _clone_batch(batch)
    scaled_batch["target_scale"] = {
        "scale": torch.tensor([[2.0]], dtype=raw_output.dtype),
        "center": torch.tensor([[1.0]], dtype=raw_output.dtype),
    }

    with torch.no_grad():
        scaled_output = model(scaled_batch)["prediction"]

    assert torch.allclose(scaled_output, raw_output * 2.0 + 1.0)


def test_univariate_forecast():
    """Test univariate forecasting with NLinear."""
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
        known=[],
        unknown=["value"],
        static=[],
    )

    dm = TslibDataModule(ts, context_length=16, prediction_length=4, batch_size=4)
    dm.setup()

    metadata = dm.metadata
    model = NLinear(loss=MAE(), metadata=metadata)

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    pred = output["prediction"]
    assert pred.ndim == 3
    assert pred.shape[0] == dm.batch_size
    assert pred.shape[1] == metadata["prediction_length"]
    assert pred.shape[-1] == 1
