"""
Tests for the CrossFormer v2 model and forecaster.
"""

from lightning.pytorch import Trainer
import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import TslibDataModule
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_forecasting.models.crossformer import (
    CrossFormer,
    CrossFormerForecaster,
)


def _make_dataframe(n_samples=120, n_groups=3, n_covariates=2, n_targets=1, seed=42):
    """Create a synthetic multivariate dataframe."""
    rng = np.random.default_rng(seed)
    frames = []
    time_idx = np.arange(n_samples)

    target_cols = [f"target_{i}" for i in range(n_targets)]
    covar_cols = [f"covar_{i}" for i in range(n_covariates)]

    for g in range(n_groups):
        df = pd.DataFrame({"time_idx": time_idx, "group_id": f"g{g}"})
        for t, col in enumerate(target_cols):
            df[col] = (
                10 * g
                + 0.1 * time_idx
                + 5 * np.sin(2 * np.pi * time_idx / 20 + t)
                + rng.normal(0, 0.5, n_samples)
            )
        for c, col in enumerate(covar_cols):
            df[col] = 2 * np.cos(2 * np.pi * time_idx / (15 + c)) + rng.normal(
                0, 0.5, n_samples
            )
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["group_id"] = df["group_id"].astype("category")
    return df, target_cols, covar_cols


def _make_timeseries(
    n_targets=1, n_covariates=2, context_length=30, prediction_length=6
):
    """Build a TimeSeries + TslibDataModule from synthetic data."""
    df, target_cols, covar_cols = _make_dataframe(
        n_targets=n_targets, n_covariates=n_covariates
    )
    ts = TimeSeries(
        data=df,
        time="time_idx",
        target=target_cols,
        group=["group_id"],
        num=target_cols + covar_cols,
        cat=[],
        known=["time_idx"],
        unknown=target_cols + covar_cols,
    )
    dm = TslibDataModule(
        ts,
        context_length=context_length,
        prediction_length=prediction_length,
        batch_size=4,
        train_val_test_split=(0.7, 0.15, 0.15),
    )
    dm.setup()
    return ts, dm


def _small_model(metadata, **kwargs):
    """Small CrossFormer for fast tests."""
    params = dict(
        loss=MAE(),
        d_model=32,
        n_heads=2,
        d_ff=64,
        e_layers=1,
        seg_len=5,
        win_size=2,
        factor=4,
        dropout=0.0,
        metadata=metadata,
    )
    params.update(kwargs)
    return CrossFormer(**params)


def _batch(dm):
    return next(iter(dm.train_dataloader()))[0]


def test_forward_output_shape():
    """Prediction shape is (batch, prediction_length, target_dim)."""
    _, dm = _make_timeseries()
    model = _small_model(dm.metadata)
    model.eval()

    batch = _batch(dm)
    with torch.no_grad():
        output = model(batch)

    prediction = output["prediction"]
    # single target + point loss -> (batch, prediction_length, n_quantiles=1)
    assert prediction.shape == (
        batch["history_target"].shape[0],
        dm.metadata["prediction_length"],
        1,
    )
    assert not torch.isnan(prediction).any()
    assert not torch.isinf(prediction).any()


def test_quantile_loss_output():
    """QuantileLoss produces a trailing quantile dimension."""
    _, dm = _make_timeseries()
    quantiles = [0.1, 0.5, 0.9]
    model = _small_model(dm.metadata, loss=QuantileLoss(quantiles=quantiles))
    model.eval()

    with torch.no_grad():
        prediction = model(_batch(dm))["prediction"]

    # single target -> (batch, prediction_length, n_quantiles)
    assert prediction.shape == (
        prediction.shape[0],
        dm.metadata["prediction_length"],
        len(quantiles),
    )


@pytest.mark.parametrize(
    "seg_len,d_model,n_heads,e_layers,win_size",
    [
        (5, 32, 2, 1, 2),  # default-ish small config
        (4, 64, 4, 2, 2),  # larger model, more layers
        (10, 16, 2, 1, 2),  # large segments -> few segments per window
        (2, 32, 2, 2, 3),  # many small segments, odd merge window
    ],
)
def test_different_seg_len_and_model_sizes(
    seg_len, d_model, n_heads, e_layers, win_size
):
    """Model works across segment lengths and hierarchical scale configs."""
    _, dm = _make_timeseries()
    model = _small_model(
        dm.metadata,
        seg_len=seg_len,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_model * 2,
        e_layers=e_layers,
        win_size=win_size,
    )
    model.eval()

    with torch.no_grad():
        prediction = model(_batch(dm))["prediction"]

    assert prediction.shape[1] == dm.metadata["prediction_length"]
    assert prediction.shape[2] == dm.metadata["n_features"]["target"]
    assert not torch.isnan(prediction).any()


@pytest.mark.parametrize("seg_len", [4, 5, 7])
def test_history_length_requiring_padding(seg_len):
    """context_length not divisible by seg_len must be padded correctly."""
    _, dm = _make_timeseries(context_length=30, prediction_length=7)
    model = _small_model(dm.metadata, seg_len=seg_len, d_model=32, n_heads=2)
    model.eval()

    with torch.no_grad():
        prediction = model(_batch(dm))["prediction"]

    assert prediction.shape[1] == 7  # prediction_length
    assert not torch.isnan(prediction).any()


def test_univariate_no_covariates():
    """Model works with a single target and no continuous covariates."""
    _, dm = _make_timeseries(n_covariates=0)
    model = _small_model(dm.metadata)
    model.eval()

    assert model.data_dim == 1

    with torch.no_grad():
        prediction = model(_batch(dm))["prediction"]

    assert prediction.shape[1] == dm.metadata["prediction_length"]
    assert prediction.shape[2] == 1
    assert not torch.isnan(prediction).any()


def test_multivariate_targets():
    """Model predicts all target dimensions jointly."""
    _, dm = _make_timeseries(n_targets=2, n_covariates=1)
    model = _small_model(dm.metadata)
    model.eval()

    assert model.target_dim == 2
    # data_dim = targets + continuous covariates
    assert model.data_dim == 2 + dm.metadata["n_features"]["continuous"]

    with torch.no_grad():
        prediction = model(_batch(dm))["prediction"]

    assert prediction.shape[2] == 2
    assert not torch.isnan(prediction).any()


def test_covariates_affect_prediction():
    """Continuous covariates are part of the model input (data_dim > target_dim)."""
    _, dm = _make_timeseries()
    model = _small_model(dm.metadata)
    model.eval()

    batch = _batch(dm)
    batch_zeroed = dict(batch)
    batch_zeroed["history_cont"] = torch.zeros_like(batch["history_cont"])

    with torch.no_grad():
        out_with = model(batch)["prediction"]
        out_without = model(batch_zeroed)["prediction"]

    # zeroing covariates must change the prediction -> covariates are used
    assert not torch.allclose(out_with, out_without)


def test_forecaster_fit_predict():
    """End-to-end workflow: fit a CrossFormerForecaster and predict."""
    ts, _ = _make_timeseries()
    forecaster = CrossFormerForecaster(
        d_model=32,
        n_heads=2,
        d_ff=64,
        e_layers=1,
        seg_len=5,
        win_size=2,
        factor=4,
        dropout=0.0,
        datamodule=TslibDataModule(
            context_length=30,
            prediction_length=6,
            batch_size=4,
            train_val_test_split=(0.7, 0.15, 0.15),
        ),
    )
    forecaster.fit(
        ts,
        trainer=Trainer(
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=1,
            accelerator="cpu",
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        ),
    )

    predictions = forecaster.predict(ts, mode="prediction")

    assert predictions.metadata.is_prediction
    assert isinstance(predictions, TimeSeries)
    assert len(predictions.to_pandas()) > 0
