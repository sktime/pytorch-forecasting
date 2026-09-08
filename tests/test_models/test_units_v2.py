"""Tests for UniTS v2 model."""

import numpy as np
import pandas as pd
import pytest

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.models.units._units_v2 import UniTS

BATCH_SIZE = 2
MAX_ENCODER_LENGTH = 16
MAX_PREDICTION_LENGTH = 4
D_MODEL = 16
N_HEADS = 4
PATCH_LEN = 8
STRIDE = 4


@pytest.fixture
def sample_multivariate_data():
    """Synthetic multivariate time series DataFrame."""
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
    """TimeSeries object from sample data."""
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
def basic_data_module(basic_timeseries_dataset):
    """EncoderDecoderTimeSeriesDataModule, not yet set up."""
    return EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=basic_timeseries_dataset,
        batch_size=BATCH_SIZE,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        train_val_test_split=(0.7, 0.15, 0.15),
    )


@pytest.fixture
def basic_metadata(basic_data_module):
    """Metadata dict extracted after DataModule setup."""
    basic_data_module.setup()
    return basic_data_module.metadata



def test_d_model_not_divisible_by_n_heads(basic_metadata):
    """d_model % n_heads != 0 must raise ValueError."""
    with pytest.raises(ValueError, match="d_model"):
        UniTS(
            loss=MAE(),
            d_model=33,
            n_heads=8,
            metadata=basic_metadata,
        )


def test_patch_len_exceeds_context(basic_metadata):
    """patch_len > context_length must raise ValueError."""
    with pytest.raises(ValueError, match="patch_len"):
        UniTS(
            loss=MAE(),
            patch_len=MAX_ENCODER_LENGTH + 1,
            metadata=basic_metadata,
        )
