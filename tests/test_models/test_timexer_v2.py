import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data._tslib_data_module import TslibDataModule
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
from pytorch_forecasting.models.timexer._timexer_v2 import TimeXer


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
        )  # noqa: E501
        humidity = (
            30
            + 20 * np.cos(2 * np.pi * time_idx / 7)
            + np.random.normal(0, 5, series_len)
        )  # noqa: E501
        pressure = (
            1013
            + 10 * np.sin(2 * np.pi * time_idx / 30)
            + np.random.normal(0, 2, series_len)
        )  # noqa: E501

        static_cont_val = np.float32(i * 10.0)
        static_cat_code = np.float32(i % 2)

        df_group = pd.DataFrame(
            {
                "time_idx": time_idx,
                "group_id": f"group_{i}",
                "value": target.astype(np.float32),
                "temperature": temperature.astype(np.float32),
                "humidity": humidity.astype(np.float32),
                "pressure": pressure.astype(np.float32),
                "static_cont_feat": np.full(
                    series_len, static_cont_val, dtype=np.float32
                ),
                "static_cat_feat": np.full(
                    series_len, static_cat_code, dtype=np.float32
                ),
            }
        )
        data.append(df_group)

    df = pd.concat(data, ignore_index=True)
    df["group_id"] = df["group_id"].astype("category")

    return df


@pytest.fixture
def sample_multivariate_multi_series_data():
    """Create sample data for M mode (multiple series) testing."""
    np.random.seed(123)

    series_len = 30
    num_groups = 5
    data = []

    for i in range(num_groups):
        time_idx = np.arange(series_len, dtype=np.int64)
        base_level = 50 + i * 15
        trend_slope = 0.2 + i * 0.1
        seasonal_amp = 5 + i * 2

        # Target variables (multiple targets for M mode)
        target1 = (
            base_level
            + trend_slope * time_idx
            + seasonal_amp * np.sin(2 * np.pi * time_idx / 7)
            + np.random.normal(0, 1, series_len)
        )
        target2 = (
            base_level * 0.8
            + trend_slope * 0.5 * time_idx
            + seasonal_amp * 0.7 * np.cos(2 * np.pi * time_idx / 7)
            + np.random.normal(0, 1.5, series_len)
        )  # noqa: E501

        # Exogenous variables
        temperature = (
            18
            + 12 * np.sin(2 * np.pi * time_idx / 365)
            + np.random.normal(0, 2, series_len)
        )  # noqa: E501
        humidity = (
            45
            + 25 * np.cos(2 * np.pi * time_idx / 7 + i * np.pi / 4)
            + np.random.normal(0, 4, series_len)
        )  # noqa: E501
        pressure = (
            1010
            + 8 * np.sin(2 * np.pi * time_idx / 30)
            + np.random.normal(0, 1.5, series_len)
        )  # noqa: E501
        wind_speed = (
            5
            + 3 * np.sin(2 * np.pi * time_idx / 14)
            + np.random.normal(0, 1, series_len)
        )  # noqa: E501

        df_group = pd.DataFrame(
            {
                "time_idx": time_idx,
                "group_id": f"series_{i}",
                "target1": target1.astype(np.float32),
                "target2": target2.astype(np.float32),
                "temperature": temperature.astype(np.float32),
                "humidity": humidity.astype(np.float32),
                "pressure": pressure.astype(np.float32),
                "wind_speed": wind_speed.astype(np.float32),
            }
        )
        data.append(df_group)

    df = pd.concat(data, ignore_index=True)
    df["group_id"] = df["group_id"].astype("category")

    return df


def basic_timeseries_dataset(sample_multivariate_data):
    """Create a basic TimeSeries dataset for testing."""
    return TimeSeries(
        data=sample_multivariate_data,
        time="time_idx",
        target="value",
        group=["group_id"],
        num=[
            "value",
            "temperature",
            "humidity",
            "pressure",
            "static_cont_feat",
            "static_cat_feat",
        ],
        cat=[],
        known=["temperature", "humidity", "pressure", "time_idx"],
        static=["static_cont_feat", "static_cat_feat"],
    )


@pytest.fixture
def basic_tslib_data_module(basic_timeseries_dataset):
    """Create a basic TslibDataModule for testing."""
    return TslibDataModule(
        time_series_dataset=basic_timeseries_dataset,
        batch_size=4,
        context_length=12,
        prediction_length=8,
        train_val_test_split=(0.6, 0.2, 0.2),
    )


@pytest.fixture
def basic_metadata(basic_tslib_data_module):
    """Basic metadata from data module for model initialization."""
    basic_tslib_data_module.setup()

    # Return the generated metadata
    return basic_tslib_data_module.metadata


@pytest.fixture
def model(basic_metadata):
    """Initialize a TimeXer model for testing."""
    return TimeXer(
        loss=MAE(),
        hidden_size=64,
        n_heads=8,
        e_layers=2,
        d_ff=256,
        droupout=0.1,
        patch_length=4,
        logging_metrics=[SMAPE()],
        optimizer="adam",
        optimizer_params={"lr": 1e-3},
        lr_scheduler="reduce_lr_on_plateau",
        lr_scheduler_params={
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
        },
        metadata=basic_metadata,
    )
