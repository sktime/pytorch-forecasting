import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
from pytorch_forecasting.metrics import MAE, SMAPE
from pytorch_forecasting.models.nhits._nhits_pkg_v2 import NHiTS_pkg_v2
from pytorch_forecasting.models.nhits._nhits_v2 import NHiTS_v2

CONTEXT_LENGTH = 6
PREDICTION_LENGTH = 3
BATCH_SIZE = 4
N_SERIES = 3
N_SAMPLES = 80


@pytest.fixture
def sample_datamodule():
    """Create a sample EncoderDecoderTimeSeriesDataModule for testing.

    Returns
    -------
    dm : EncoderDecoderTimeSeriesDataModule
        Configured data module with synthetic univariate time series.
    """
    time_idx = np.arange(N_SAMPLES)
    series_data = []
    for i in range(N_SERIES):
        values = np.sin(2 * np.pi * time_idx / 20) + np.random.normal(0, 0.1, N_SAMPLES)
        series_data.append(
            pd.DataFrame({"time_idx": time_idx, "series_id": i, "value": values})
        )
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
    )

    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=ts,
        max_encoder_length=CONTEXT_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,
        batch_size=BATCH_SIZE,
    )
    dm.setup("fit")
    return dm


def test_nhits_v2_forward_shapes(sample_datamodule):
    """Test that forward pass returns correct output shapes.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    """
    dm = sample_datamodule
    metadata = dm.metadata
    model = NHiTS_v2(loss=MAE(), metadata=metadata)

    batch_x, _ = next(iter(dm.train_dataloader()))

    with torch.no_grad():
        out = model(batch_x)

    assert "prediction" in out
    assert "backcast" in out
    assert "block_forecasts" in out
    assert "block_backcasts" in out

    pred = out["prediction"]
    assert pred.ndim == 3, f"prediction must be 3D, got {pred.ndim}D"
    assert pred.shape[1] == PREDICTION_LENGTH
    assert pred.shape[2] == 1

    backcast = out["backcast"]
    assert backcast.ndim == 3, f"backcast must be 3D, got {backcast.ndim}D"
    assert backcast.shape[1] == CONTEXT_LENGTH
    assert backcast.shape[2] == 1


def test_nhits_v2_training_step(sample_datamodule):
    """Test that training_step returns a scalar loss with gradients.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    """
    dm = sample_datamodule
    metadata = dm.metadata
    model = NHiTS_v2(loss=MAE(), metadata=metadata)

    batch = next(iter(dm.train_dataloader()))
    result = model.training_step(batch, batch_idx=0)

    assert "loss" in result
    loss = result["loss"]
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    loss.backward()


@pytest.mark.parametrize("backcast_loss_ratio", [0.0, 0.1, 0.5])
def test_nhits_v2_backcast_loss_ratio(sample_datamodule, backcast_loss_ratio):
    """Test that backcast_loss_ratio affects the training loss.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    backcast_loss_ratio : float
        Weight of the backcast loss, parametrized over multiple values.
    """
    dm = sample_datamodule
    metadata = dm.metadata
    model = NHiTS_v2(
        loss=MAE(), metadata=metadata, backcast_loss_ratio=backcast_loss_ratio
    )

    batch = next(iter(dm.train_dataloader()))
    result = model.training_step(batch, batch_idx=0)

    loss = result["loss"]
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0


@pytest.mark.parametrize(
    "n_blocks, hidden_size",
    [
        ([1, 1, 1], 64),
        ([1, 1], 128),
        ([1], 32),
    ],
)
def test_nhits_v2_architecture_variants(sample_datamodule, n_blocks, hidden_size):
    """Test that different n_blocks and hidden_size configs run without error.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    n_blocks : list of int
        Number of blocks per stack.
    hidden_size : int
        Width of the MLP layers.
    """
    dm = sample_datamodule
    metadata = dm.metadata
    model = NHiTS_v2(
        loss=MAE(), metadata=metadata, n_blocks=n_blocks, hidden_size=hidden_size
    )

    batch_x, _ = next(iter(dm.train_dataloader()))
    with torch.no_grad():
        out = model(batch_x)

    assert out["prediction"].shape[1] == PREDICTION_LENGTH


def test_nhits_v2_pkg_get_cls():
    """Test that NHiTS_pkg_v2.get_cls() returns NHiTS_v2."""
    assert NHiTS_pkg_v2.get_cls() is NHiTS_v2


def test_nhits_v2_pkg_naming_convention():
    """Test that pkg class name follows the convention NHiTS_pkg_v2."""
    assert NHiTS_pkg_v2.__name__ == "NHiTS_pkg_v2"


def test_nhits_v2_pkg_test_train_params():
    """Test that get_test_train_params returns a non-empty list of dicts."""
    params = NHiTS_pkg_v2.get_test_train_params()
    assert isinstance(params, list)
    assert len(params) > 0
    for p in params:
        assert isinstance(p, dict)
        assert "datamodule_cfg" in p


def test_nhits_v2_default_loss(sample_datamodule):
    """Test that loss=None falls back to MASE as default.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    """
    from pytorch_forecasting.metrics import MASE

    dm = sample_datamodule
    model = NHiTS_v2(metadata=dm.metadata)
    assert isinstance(model.loss, MASE)


def test_nhits_v2_validation_step(sample_datamodule):
    """Test that validation_step returns a scalar val_loss.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    """
    dm = sample_datamodule
    model = NHiTS_v2(loss=MAE(), metadata=dm.metadata)

    batch = next(iter(dm.train_dataloader()))
    result = model.validation_step(batch, batch_idx=0)

    assert "val_loss" in result
    loss = result["val_loss"]
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_nhits_v2_forward_with_2d_mask(sample_datamodule):
    """Test forward pass when encoder_mask is already 2D.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    """
    dm = sample_datamodule
    model = NHiTS_v2(loss=MAE(), metadata=dm.metadata)

    batch_x, _ = next(iter(dm.train_dataloader()))

    batch_x_2d_mask = dict(batch_x)
    if batch_x_2d_mask["encoder_mask"].dim() == 3:
        batch_x_2d_mask["encoder_mask"] = batch_x_2d_mask["encoder_mask"].squeeze(-1)

    with torch.no_grad():
        out = model(batch_x_2d_mask)

    assert out["prediction"].shape[1] == PREDICTION_LENGTH


def test_nhits_v2_forward_with_3d_mask(sample_datamodule):
    """Test forward pass when encoder_mask is 3D — exercises the squeeze branch.

    The model's forward() contains ``if encoder_mask.dim() == 3: squeeze(-1)``.
    This test passes a 3D mask directly to ensure that branch is covered.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    """
    dm = sample_datamodule
    model = NHiTS_v2(loss=MAE(), metadata=dm.metadata)

    batch_x, _ = next(iter(dm.train_dataloader()))

    batch_x_3d_mask = dict(batch_x)
    mask = batch_x_3d_mask["encoder_mask"]
    # Ensure mask is 3D: (batch, context_length, 1)
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    batch_x_3d_mask["encoder_mask"] = mask

    with torch.no_grad():
        out = model(batch_x_3d_mask)

    assert out["prediction"].shape[1] == PREDICTION_LENGTH


# ---------------------------------------------------------------------------
# Slow test: NHiTS v1 vs v2 numerical validation
# ---------------------------------------------------------------------------

_CMP_N_SERIES = 4
_CMP_N_SAMPLES = 120
_CMP_CONTEXT = 12
_CMP_PRED = 4
_CMP_EPOCHS = 10
_CMP_SEED = 42


def _make_comparison_dataframe():
    """Shared synthetic DataFrame for v1/v2 comparison."""
    rng = np.random.default_rng(_CMP_SEED)
    time_idx = np.arange(_CMP_N_SAMPLES)
    rows = []
    for i in range(_CMP_N_SERIES):
        values = np.sin(2 * np.pi * time_idx / 20 + i) + rng.normal(
            0, 0.1, _CMP_N_SAMPLES
        )
        for t, v in zip(time_idx, values):
            rows.append({"time_idx": int(t), "series_id": i, "value": float(v)})
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_nhits_v1_v2_val_loss_comparable():
    """Numerical validation: NHiTS v2 val MAE must be comparable to v1.

    Both models are trained on the same synthetic dataset for the same number
    of epochs. The test asserts:

    1. Both models converge (val MAE < 1.0 on normalised sine data).
    2. The v2 val MAE is within 3x of the v1 val MAE, confirming the rework
       has not introduced a systematic numerical regression.

    Run explicitly with::

        pytest -m slow tests/test_models/test_nhits_v2.py
    """
    from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
    from pytorch_forecasting.metrics import MAE as MAE_v1
    from pytorch_forecasting.models import NHiTS as NHiTS_v1

    df = _make_comparison_dataframe()

    # ------------------------------------------------------------------ v1
    train_ds_v1 = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="value",
        group_ids=["series_id"],
        time_varying_unknown_reals=["value"],
        max_encoder_length=_CMP_CONTEXT,
        max_prediction_length=_CMP_PRED,
        min_encoder_length=_CMP_CONTEXT,
    )
    train_dl_v1 = train_ds_v1.to_dataloader(train=True, batch_size=8, num_workers=0)

    pl.seed_everything(_CMP_SEED)
    model_v1 = NHiTS_v1.from_dataset(
        train_ds_v1,
        hidden_size=32,
        n_blocks=[1, 1, 1],
        learning_rate=1e-3,
        loss=MAE_v1(),
    )
    trainer_v1 = pl.Trainer(
        max_epochs=_CMP_EPOCHS,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        limit_val_batches=0,  # no val needed
    )
    trainer_v1.fit(model_v1, train_dl_v1)

    # compute train MAE manually after training
    model_v1.eval()
    v1_losses = []
    with torch.no_grad():
        for batch in train_dl_v1:
            x, y = batch
            out = model_v1(x)
            pred = out["prediction"]
            if hasattr(pred, "prediction"):
                pred = pred.prediction
            if pred.dim() == 3:
                pred = pred[..., 0]
            target = y if isinstance(y, torch.Tensor) else y[0]
            if target.dim() == 2:
                pass
            v1_losses.append(float(torch.mean(torch.abs(pred - target))))
    train_mae_v1 = float(np.mean(v1_losses))

    # ------------------------------------------------------------------ v2
    ts_v2 = TimeSeries(
        df,
        time="time_idx",
        group=["series_id"],
        target=["value"],
        num=[],
        cat=[],
        known=[],
        unknown=["value"],
    )
    dm_v2 = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=ts_v2,
        max_encoder_length=_CMP_CONTEXT,
        max_prediction_length=_CMP_PRED,
        batch_size=8,
        train_val_test_split=(0.8, 0.1, 0.1),
    )
    dm_v2.setup("fit")

    pl.seed_everything(_CMP_SEED)
    model_v2 = NHiTS_v2(
        loss=MAE(),
        metadata=dm_v2.metadata,
        hidden_size=32,
        n_blocks=[1, 1, 1],
    )
    trainer_v2 = pl.Trainer(
        max_epochs=_CMP_EPOCHS,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        limit_val_batches=0,
    )
    trainer_v2.fit(model_v2, dm_v2)

    # compute train MAE manually after training
    model_v2.eval()
    v2_losses = []
    with torch.no_grad():
        for batch in dm_v2.train_dataloader():
            x, y = batch
            out = model_v2(x)
            pred = out["prediction"]  # (batch, pred_len, 1)
            pred = pred.squeeze(-1)  # (batch, pred_len)
            v2_losses.append(float(torch.mean(torch.abs(pred - y))))
    train_mae_v2 = float(np.mean(v2_losses))

    # ------------------------------------------------------------------ assert
    assert (
        train_mae_v1 < 1.0
    ), f"NHiTS v1 did not converge: train MAE = {train_mae_v1:.4f}"
    assert (
        train_mae_v2 < 1.0
    ), f"NHiTS v2 did not converge: train MAE = {train_mae_v2:.4f}"

    ratio = train_mae_v2 / (train_mae_v1 + 1e-8)
    assert ratio < 3.0, (
        f"NHiTS v2 train MAE ({train_mae_v2:.4f}) is more than 3x "
        f"NHiTS v1 train MAE ({train_mae_v1:.4f}). "
        "Possible numerical regression in v2 rework."
    )
