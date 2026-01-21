import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data._tslib_data_module import TslibDataModule
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
from pytorch_forecasting.models.itransformer import iTransformer


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


@pytest.fixture(params=[False, True], ids=["einsum_attn", "efficient_attn"])
def model(request, basic_metadata):
    """Initialize a TimeXer model for testing."""
    return iTransformer(
        loss=MAE(),
        hidden_size=64,
        n_heads=8,
        e_layers=2,
        d_ff=256,
        dropout=0.1,
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

    assert isinstance(model, iTransformer)

    assert model.d_model == 512
    assert model.n_heads == 8
    assert model.e_layers == 2
    assert model.d_ff == 256
    assert model.dropout == 0.1

    assert model.context_length == basic_metadata["context_length"]
    assert model.prediction_length == basic_metadata["prediction_length"]
    assert model.cont_dim == basic_metadata["n_features"]["continuous"]
    assert model.cat_dim == basic_metadata["n_features"]["categorical"]
    assert model.target_dim == basic_metadata["n_features"]["target"]
    assert model.features == basic_metadata["features"]


def test_quantile_prediction(basic_metadata):
    """Test quantile predictions with iTransformer model."""
    quantiles = [0.1, 0.5, 0.9]
    model = iTransformer(
        loss=QuantileLoss(quantiles=quantiles),
        d_model=32,
        n_heads=4,
        e_layers=1,
        d_ff=64,
        dropout=0.1,
        metadata=basic_metadata,
    )
    batch_size = 4
    sample_input = {
        "history_cont": torch.randn(
            batch_size,
            basic_metadata["context_length"],
            basic_metadata["n_features"]["continuous"],
        ),
        "history_target": torch.randn(
            batch_size,
            basic_metadata["context_length"],
            basic_metadata["n_features"]["target"],
        ),
    }
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
    prediction = output["prediction"]
    assert prediction.shape == (batch_size, model.prediction_length, len(quantiles))
    assert not torch.isnan(prediction).any()
    assert not torch.isinf(prediction).any()


def test_integration_with_datamodule(model, basic_tslib_data_module):
    """Test integration of iTransformer model with TslibDataModule."""
    basic_tslib_data_module.setup(stage="fit")
    basic_tslib_data_module.setup(stage="test")
    train_loader = basic_tslib_data_module.train_dataloader()
    test_loader = basic_tslib_data_module.test_dataloader()
    val_loader = basic_tslib_data_module.val_dataloader()
    batch = next(iter(train_loader))[0]
    model.eval()
    with torch.no_grad():
        output = model(batch)
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
        prediction = output["prediction"]
        batch_size = batch["history_cont"].shape[0]
        assert prediction.shape[0] == batch_size
        assert prediction.shape[1] == model.prediction_length


def test_attention_output_shape(basic_metadata):
    """Test that attention output is present when output_attention=True."""
    model = iTransformer(
        loss=MAE(),
        d_model=16,
        n_heads=2,
        e_layers=1,
        d_ff=32,
        dropout=0.1,
        output_attention=True,
        metadata=basic_metadata,
    )
    batch_size = 2
    sample_input = {
        "history_cont": torch.randn(
            batch_size,
            basic_metadata["context_length"],
            basic_metadata["n_features"]["continuous"],
        ),
        "history_target": torch.randn(
            batch_size,
            basic_metadata["context_length"],
            basic_metadata["n_features"]["target"],
        ),
    }
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
    assert "attention" in output
    attn = output["attention"]
    assert isinstance(attn, list) or isinstance(attn, tuple)
    assert len(attn) == model.e_layers


def test_arbitrary_num_variables(basic_metadata):
    """Test iTransformer can forecast with arbitrary numbers of variables."""
    for n_vars in [1, 3, 10]:
        meta = dict(basic_metadata)
        meta["n_features"] = dict(meta["n_features"])
        meta["n_features"]["continuous"] = n_vars
        model = iTransformer(
            loss=MAE(),
            d_model=8,
            n_heads=2,
            e_layers=1,
            d_ff=16,
            dropout=0.1,
            metadata=meta,
        )
        batch_size = 2
        sample_input = {
            "history_cont": torch.randn(batch_size, meta["context_length"], n_vars),
            "history_target": torch.randn(
                batch_size, meta["context_length"], meta["n_features"]["target"]
            ),
        }
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        prediction = output["prediction"]
        assert prediction.shape[0] == batch_size
        assert prediction.shape[1] == model.prediction_length


def test_multivariate_benchmark_like(basic_metadata):
    """Test iTransformer on a challenging multivariate-like batch (simulated)."""
    # Simulate a batch with more variables and longer context
    meta = dict(basic_metadata)
    meta["n_features"] = dict(meta["n_features"])
    meta["n_features"]["continuous"] = 8
    meta["context_length"] = 24
    meta["prediction_length"] = 12
    model = iTransformer(
        loss=MAE(),
        d_model=16,
        n_heads=2,
        e_layers=2,
        d_ff=32,
        dropout=0.1,
        metadata=meta,
    )
    batch_size = 3
    sample_input = {
        "history_cont": torch.randn(
            batch_size, meta["context_length"], meta["n_features"]["continuous"]
        ),
        "history_target": torch.randn(
            batch_size, meta["context_length"], meta["n_features"]["target"]
        ),
    }
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
    prediction = output["prediction"]
    assert prediction.shape == (
        batch_size,
        meta["prediction_length"],
        meta["n_features"]["target"],
    )
    assert not torch.isnan(prediction).any()
