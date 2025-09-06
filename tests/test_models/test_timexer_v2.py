"""
Basic test frameowrk for TimeXer v2 model.
TODO:
- Add tests for testing the scaling of features, once that is implemented in the D1/D2
  level.
- Add tests for the M mode (multiple series) once that is implemented.
"""

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


@pytest.fixture
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
        batch_size=2,
        context_length=12,
        prediction_length=8,
        train_val_test_split=(0.7, 0.15, 0.15),
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
        dropout=0.1,
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


def test_basic_model_initialization(model, basic_metadata):
    """Test the basic model initialization."""

    assert isinstance(model, TimeXer)

    assert model.hidden_size == 64
    assert model.n_heads == 8
    assert model.e_layers == 2
    assert model.d_ff == 256
    assert model.patch_length == 4
    assert model.dropout == 0.1

    assert model.patch_num == 3
    assert model.n_target_vars == 1
    assert model.head_nf == 64 * (3 + 1)

    assert model.context_length == basic_metadata["context_length"]
    assert model.prediction_length == basic_metadata["prediction_length"]
    assert model.cont_dim == basic_metadata["n_features"]["continuous"]
    assert model.cat_dim == basic_metadata["n_features"]["categorical"]
    assert model.target_dim == basic_metadata["n_features"]["target"]
    assert model.features == basic_metadata["features"]


def test_multivariate_single_series(model, basic_tslib_data_module):
    basic_tslib_data_module.setup()
    train_dataloader = basic_tslib_data_module.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    model.eval()
    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    predictions = output["prediction"]

    batch_size = batch["history_cont"].shape[0]
    assert predictions.shape == (batch_size, model.prediction_length, model.target_dim)

    assert not torch.isnan(predictions).any()
    assert not torch.isinf(predictions).any()


def test_quantile_predictions(basic_metadata):
    """Test quantile predictions with TimeXer model."""

    quantiles = [0.1, 0.5, 0.9]

    model = TimeXer(
        loss=QuantileLoss(quantiles=quantiles),
        hidden_size=64,
        n_heads=8,
        e_layers=2,
        d_ff=256,
        dropout=0.1,
        patch_length=4,
        metadata=basic_metadata,
    )

    assert model.n_quantiles == 3

    batch_size = 4

    # sample input data as a substitute for x
    sample_input_data = {
        "history_cont": torch.randn(
            batch_size, 12, basic_metadata["n_features"]["continuous"]
        ),
        "history_target": torch.randn(
            batch_size, 12, basic_metadata["n_features"]["target"]
        ),
        "history_time_idx": torch.arange(12).unsqueeze(0).repeat(batch_size, 1),
    }

    model.eval()
    with torch.no_grad():
        output = model(sample_input_data)

    predictions = output["prediction"]
    assert predictions.shape == (batch_size, 8, 3)


def test_missing_history_target_handling(basic_metadata):
    """Test handling of missing history_target in TimeXer model."""

    model = TimeXer(
        loss=MAE(),
        hidden_size=64,
        n_heads=8,
        e_layers=2,
        d_ff=256,
        dropout=0.1,
        patch_length=4,
        metadata=basic_metadata,
    )

    batch_size = 4
    sample_input = {
        "history_cont": torch.randn(
            batch_size, 12, basic_metadata["n_features"]["continuous"]
        ),  # noqa: E501
        "history_time_idx": torch.arange(12).unsqueeze(0).repeat(batch_size, 1),
    }

    model.eval()
    with torch.no_grad():
        output = model(sample_input)

    predictions = output["prediction"]
    assert predictions.shape == (batch_size, 8, basic_metadata["n_features"]["target"])
    assert not torch.isnan(predictions).any()


def test_endogenous_exogenous_variable_selection(basic_metadata):
    """Test explicit endogenous and exogenous variable selection in TimeXer model."""

    endo_names = basic_metadata["feature_names"]["continuous"][0]
    exog_names = basic_metadata["feature_names"]["continuous"][1]

    model = TimeXer(
        loss=MAE(),
        hidden_size=64,
        n_heads=8,
        endogenous_vars=[endo_names],
        exogenous_vars=[exog_names],
        e_layers=2,
        metadata=basic_metadata,
    )

    batch_size = 4
    sample_input = {
        "history_cont": torch.randn(
            batch_size, 12, basic_metadata["n_features"]["continuous"]
        ),
        "history_target": torch.randn(
            batch_size, 12, basic_metadata["n_features"]["target"]
        ),
        "history_time_idx": torch.arange(12).unsqueeze(0).repeat(batch_size, 1),
    }

    model.eval()
    with torch.no_grad():
        output = model(sample_input)

    predictions = output["prediction"]
    assert predictions.shape == (batch_size, 8, 1)
    assert not torch.isnan(predictions).any()


def test_integration_with_datamodule(model, basic_tslib_data_module):
    """Test integration of TimeXer model with TslibDataModule."""

    basic_tslib_data_module.setup(stage="fit")
    basic_tslib_data_module.setup(stage="test")

    train_loader = basic_tslib_data_module.train_dataloader()
    test_loader = basic_tslib_data_module.test_dataloader()
    val_loader = basic_tslib_data_module.val_dataloader()

    model.eval()
    with torch.no_grad():
        train_batch = next(iter(train_loader))[0]
        train_output = model(train_batch)
        assert train_output["prediction"].shape[1] == model.prediction_length

        # Check if validation and test sets are not empty
        # If they are empty, skip the validation and test checks
        try:
            val_batch = next(iter(val_loader))[0]
            val_output = model(val_batch)
            assert val_output["prediction"].shape[1] == model.prediction_length
        except StopIteration:
            print("Validation set is empty, skipping validation testing")

        try:
            test_batch = next(iter(test_loader))[0]
            test_output = model(test_batch)
            assert test_output["prediction"].shape[1] == model.prediction_length
        except StopIteration:
            print("Test set is empty, skipping test testing")
