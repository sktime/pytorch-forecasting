"""Basic test suite for Naive v2 model."""

import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import TslibDataModule
from pytorch_forecasting.metrics import MAE, SMAPE
from pytorch_forecasting.models.baseline import Naive


@pytest.fixture
def basic_tslib_data_module():
    """Create a basic TslibDataModule for testing."""
    # Create a simple time series dataset
    np.random.seed(42)
    series_len = 30
    df = pd.DataFrame(
        {
            "time_idx": np.tile(np.arange(series_len, dtype=np.int64), 2),
            "group_id": np.repeat(["group_0", "group_1"], series_len),
            "value": np.random.rand(series_len * 2).astype(np.float32),
        }
    )
    df["group_id"] = df["group_id"].astype("category")

    ts = TimeSeries(
        data=df,
        time="time_idx",
        target="value",
        group=["group_id"],
        num=["value"],
        cat=[],
    )

    dm = TslibDataModule(
        time_series=ts,
        batch_size=2,
        context_length=8,
        prediction_length=4,
        train_val_test_split=(0.7, 0.15, 0.15),
    )
    dm.setup()
    return dm


@pytest.fixture
def model(basic_tslib_data_module):
    """Initialize a Naive v2 model using real datamodule metadata."""
    return Naive(
        loss=MAE(),
        logging_metrics=[SMAPE()],
        metadata=basic_tslib_data_module.metadata,
    )


def test_basic_model_initialization(model, basic_tslib_data_module):
    """Verify metadata wiring and zero-parameter baseline invariants."""
    metadata = basic_tslib_data_module.metadata

    assert isinstance(model, Naive)
    assert model.context_length == metadata["context_length"]
    assert model.prediction_length == metadata["prediction_length"]
    assert list(model.parameters()) == []
    assert model.configure_optimizers() is None


def test_last_value_repeat_exact(model):
    """Verify forward() repeats the exact last observed target value."""
    # Shape: (batch_size=2, context_length=4, n_targets=1)
    history = torch.tensor(
        [
            [[1.0], [2.0], [3.0], [10.0]],
            [[5.0], [6.0], [7.0], [20.0]],
        ]
    )

    model.eval()
    with torch.no_grad():
        pred = model({"history_target": history})["prediction"]

    expected = torch.tensor(
        [
            [[10.0]] * model.prediction_length,
            [[20.0]] * model.prediction_length,
        ]
    )

    assert pred.shape == (2, model.prediction_length, 1)
    assert torch.allclose(pred, expected)


def test_invalid_history_target_raises(model):
    """Test that the model raises a ValueError
    when 'history_target' is missing or empty."""
    with pytest.raises(ValueError):
        model({})  # Missing 'history_target'

    with pytest.raises(ValueError):
        model({"history_target": torch.empty(2, 0, 1)})  # Empty history_target


def test_multivariate_last_value_repeat(model):
    """Test that the model can handle multivariate data
    and works with TslibDataModule."""
    x = torch.randn(2, 8, 3)
    pred = model({"history_target": x})["prediction"]
    assert pred.shape == (2, model.prediction_length, 3)
    expected = x[:, -1:, :].repeat(1, model.prediction_length, 1)
    assert torch.allclose(pred, expected)
