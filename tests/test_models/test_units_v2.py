"""Tests for UniTS v2 model."""

import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import TslibDataModule
from pytorch_forecasting.metrics import MAE, SMAPE
from pytorch_forecasting.models.units._units_v2 import UniTS


@pytest.fixture
def sample_multivariate_data():
    """Sample multivariate data for testing."""
    np.random.seed(42)
    series_len = 30
    num_groups = 3
    data = []

    for i in range(num_groups):
        time_idx = np.arange(series_len, dtype=np.int64)
        trend = 100 + i * 20 + 0.5 * time_idx
        seasonal = 10 * np.sin(2 * np.pi * time_idx / 12)
        noise = np.random.normal(0, 5, series_len)
        target = trend + seasonal + noise

        temperature = (
            20
            + 15 * np.sin(2 * np.pi * time_idx / 365)
            + np.random.normal(0, 3, series_len)
        )
        humidity = (
            30
            + 20 * np.cos(2 * np.pi * time_idx / 7)
            + np.random.normal(0, 5, series_len)
        )

        df_group = pd.DataFrame(
            {
                "time_idx": time_idx,
                "group_id": f"group_{i}",
                "value": target.astype(np.float32),
                "temperature": temperature.astype(np.float32),
                "humidity": humidity.astype(np.float32),
            }
        )
        data.append(df_group)

    df = pd.concat(data, ignore_index=True)
    df["group_id"] = df["group_id"].astype("category")
    return df


@pytest.fixture
def basic_timeseries_dataset(sample_multivariate_data):
    """Create a basic TimeSeries dataset for testing."""
    return TimeSeries(
        data=sample_multivariate_data,
        time="time_idx",
        target="value",
        group=["group_id"],
        num=["value", "temperature", "humidity"],
        cat=[],
        known=["temperature", "humidity", "time_idx"],
        static=[],
    )


@pytest.fixture
def basic_tslib_data_module(basic_timeseries_dataset):
    """Create a basic TslibDataModule for testing."""
    return TslibDataModule(
        time_series_dataset=basic_timeseries_dataset,
        batch_size=2,
        context_length=16,
        prediction_length=4,
        train_val_test_split=(0.7, 0.15, 0.15),
    )


@pytest.fixture
def basic_metadata(basic_tslib_data_module):
    """Basic metadata from data module for model initialization."""
    basic_tslib_data_module.setup()
    return basic_tslib_data_module.metadata


@pytest.fixture(params=[16, 32], ids=["d_model_16", "d_model_32"])
def model(request, basic_metadata):
    """Initialize a UniTS model for testing."""
    return UniTS(
        loss=MAE(),
        d_model=request.param,
        n_heads=4,
        e_layers=2,
        d_ff=64,
        dropout=0.1,
        patch_len=8,
        stride=4,
        logging_metrics=[SMAPE()],
        optimizer="adam",
        metadata=basic_metadata,
    )


def test_parameter_validation(basic_metadata):
    """Test parameter validation for UniTS."""
    with pytest.raises(ValueError, match="d_model"):
        UniTS(loss=MAE(), metadata=basic_metadata, d_model=33, n_heads=8)

    with pytest.raises(ValueError, match="patch_len"):
        UniTS(loss=MAE(), metadata=basic_metadata, patch_len=32)
