"""Tests for UniTS v2 model."""

import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
from pytorch_forecasting.metrics import MAE, SMAPE, NormalDistributionLoss, QuantileLoss
from pytorch_forecasting.models.units._units_v2 import UniTS

BATCH_SIZE = 2
MAX_ENCODER_LENGTH = 16
MAX_PREDICTION_LENGTH = 4
D_MODEL = 16
N_HEADS = 4
E_LAYERS = 1
D_FF = 32
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


def test_basic_attributes(basic_metadata):
    """Model attributes match constructor args."""
    model = UniTS(
        loss=MAE(),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        e_layers=E_LAYERS,
        d_ff=D_FF,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        metadata=basic_metadata,
    )
    assert model.d_model == D_MODEL
    assert model.n_heads == N_HEADS
    assert model.e_layers == E_LAYERS
    assert model.context_length == MAX_ENCODER_LENGTH
    assert model.prediction_length == MAX_PREDICTION_LENGTH


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


def test_hyperparameters_saved(basic_metadata):
    """save_hyperparameters stores model config (not loss/metadata)."""
    model = UniTS(
        loss=MAE(),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        metadata=basic_metadata,
    )
    assert model.hparams["d_model"] == D_MODEL
    assert "loss" not in model.hparams
    assert "metadata" not in model.hparams


def test_output_shape_point_loss(basic_metadata, basic_data_module):
    """Prediction shape is (B, pred_len, target_dim) with point loss."""
    basic_data_module.setup()
    model = UniTS(
        loss=MAE(),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        e_layers=E_LAYERS,
        d_ff=D_FF,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        metadata=basic_metadata,
    )
    model.eval()

    batch_x, _ = next(iter(basic_data_module.train_dataloader()))
    with torch.no_grad():
        output = model(batch_x)

    pred = output["prediction"]
    actual_batch = batch_x["target_past"].shape[0]
    assert pred.shape == (
        actual_batch,
        MAX_PREDICTION_LENGTH,
        basic_metadata["target"],
    )


def test_no_nan_or_inf(basic_metadata, basic_data_module):
    """Output must not contain NaN or Inf values."""
    basic_data_module.setup()
    model = UniTS(
        loss=MAE(),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        metadata=basic_metadata,
    )
    model.eval()

    batch_x, _ = next(iter(basic_data_module.train_dataloader()))
    with torch.no_grad():
        pred = model(batch_x)["prediction"]

    assert not torch.isnan(pred).any(), "Predictions contain NaN"
    assert not torch.isinf(pred).any(), "Predictions contain Inf"


def test_quantile_loss_output_shape(basic_metadata, basic_data_module):
    """QuantileLoss must produce (B, pred_len, target_dim, n_quantiles) output."""
    basic_data_module.setup()
    quantiles = [0.1, 0.5, 0.9]
    model = UniTS(
        loss=QuantileLoss(quantiles=quantiles),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        metadata=basic_metadata,
    )
    model.eval()

    batch_x, _ = next(iter(basic_data_module.train_dataloader()))
    with torch.no_grad():
        pred = model(batch_x)["prediction"]

    actual_batch = batch_x["target_past"].shape[0]
    expected_shape = (
        (actual_batch, MAX_PREDICTION_LENGTH, len(quantiles))
        if basic_metadata["target"] == 1
        else (
            actual_batch,
            MAX_PREDICTION_LENGTH,
            basic_metadata["target"],
            len(quantiles),
        )
    )
    assert pred.shape == expected_shape


def test_quantile_n_quantiles_attribute(basic_metadata):
    """n_quantiles attribute set correctly when using QuantileLoss."""
    model = UniTS(
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        metadata=basic_metadata,
    )
    assert model.n_quantiles == 3


def test_point_loss_n_quantiles_is_none(basic_metadata):
    """n_quantiles is None when using a point loss like MAE."""
    model = UniTS(
        loss=MAE(),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        metadata=basic_metadata,
    )
    assert model.n_quantiles is None


def test_distribution_loss_output_shape(basic_metadata, basic_data_module):
    """DistributionLoss must produce (B, pred_len, [target_dim], n_dist_args) output."""
    basic_data_module.setup()
    loss = NormalDistributionLoss()
    model = UniTS(
        loss=loss,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        metadata=basic_metadata,
    )
    model.eval()

    batch_x, _ = next(iter(basic_data_module.train_dataloader()))
    with torch.no_grad():
        pred = model(batch_x)["prediction"]

    actual_batch = batch_x["target_past"].shape[0]
    expected_shape = (
        (actual_batch, MAX_PREDICTION_LENGTH, len(loss.distribution_arguments))
        if basic_metadata["target"] == 1
        else (
            actual_batch,
            MAX_PREDICTION_LENGTH,
            basic_metadata["target"],
            len(loss.distribution_arguments),
        )
    )
    assert pred.shape == expected_shape


@pytest.mark.parametrize("loss_cls", [MAE, SMAPE])
def test_multiple_point_losses(loss_cls, basic_metadata, basic_data_module):
    """Model produces valid output with various point losses."""
    basic_data_module.setup()
    model = UniTS(
        loss=loss_cls(),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        metadata=basic_metadata,
    )
    model.eval()

    batch_x, _ = next(iter(basic_data_module.train_dataloader()))
    with torch.no_grad():
        pred = model(batch_x)["prediction"]

    assert not torch.isnan(pred).any()


def test_metadata_dimensions(basic_metadata):
    """Metadata contains all keys the model constructor reads."""
    required_keys = [
        "max_encoder_length",
        "max_prediction_length",
        "target",
    ]
    for key in required_keys:
        assert key in basic_metadata, f"Missing metadata key: {key}"


def test_train_batch_roundtrip(basic_metadata, basic_data_module):
    """Model processes a real training batch and returns valid output."""
    basic_data_module.setup()
    model = UniTS(
        loss=MAE(),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        metadata=basic_metadata,
    )
    model.eval()

    batch_x, batch_y = next(iter(basic_data_module.train_dataloader()))
    with torch.no_grad():
        output = model(batch_x)

    pred = output["prediction"]
    assert pred.shape[1] == MAX_PREDICTION_LENGTH
    assert not torch.isnan(pred).any()


def test_loss_computes_on_real_batch(basic_metadata, basic_data_module):
    """Loss function returns a finite scalar on a real batch."""
    basic_data_module.setup()
    loss_fn = MAE()
    model = UniTS(
        loss=loss_fn,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        metadata=basic_metadata,
    )
    model.eval()

    batch_x, batch_y = next(iter(basic_data_module.train_dataloader()))
    with torch.no_grad():
        pred = model(batch_x)["prediction"]
        loss_val = loss_fn(pred, batch_y)

    assert torch.isfinite(loss_val), "Loss is not finite"
