"""NHiTS v2 tests that the generic v2 test framework does not cover.

The ``forward`` / ``fit`` / ``predict`` contract, checkpointing and the
architecture and loss variants listed in ``NHiTS_pkg_v2.get_test_train_params``
are all exercised by ``pytorch_forecasting.tests.test_all_v2``. This module only
covers behaviour that the framework cannot reach:

* the backcast loss weighting, which ``NHiTS_v2`` implements on top of the v2
  ``BaseModel`` loss, and which must agree numerically with the v1 ``NHiTS``
  weighting;
* the encoder mask semantics of the shared ``NHiTSModule``;
* numerical agreement between the v1 and v2 wrappers around that module, for
  both the point-forecast and the quantile output layouts.
"""

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_forecasting.models.nhits._nhits_v2 import NHiTS_v2

CONTEXT_LENGTH = 6
PREDICTION_LENGTH = 3
BATCH_SIZE = 4
N_SERIES = 3
N_SAMPLES = 80

HIDDEN_SIZE = 16
N_BLOCKS = [1, 1]


def _make_dataframe(n_series, n_samples, seed):
    """Build a synthetic multi-series sine DataFrame.

    Parameters
    ----------
    n_series : int
        Number of distinct series.
    n_samples : int
        Number of time steps per series.
    seed : int
        Seed for the additive noise.

    Returns
    -------
    df : pandas.DataFrame
        Columns ``time_idx``, ``series_id`` and ``value``.
    """
    rng = np.random.default_rng(seed)
    time_idx = np.arange(n_samples)
    frames = []
    for i in range(n_series):
        values = np.sin(2 * np.pi * time_idx / 20 + i) + rng.normal(0, 0.1, n_samples)
        frames.append(
            pd.DataFrame({"time_idx": time_idx, "series_id": i, "value": values})
        )
    return pd.concat(frames).reset_index(drop=True)


def _make_datamodule(df, context_length, prediction_length, batch_size):
    """Wrap a DataFrame into a set-up EncoderDecoderTimeSeriesDataModule.

    Parameters
    ----------
    df : pandas.DataFrame
        Data as returned by :func:`_make_dataframe`.
    context_length : int
        Encoder length.
    prediction_length : int
        Decoder length.
    batch_size : int
        Batch size of the returned dataloaders.

    Returns
    -------
    dm : EncoderDecoderTimeSeriesDataModule
        Data module after ``setup("fit")``.
    """
    ts = TimeSeries(
        df,
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
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
        batch_size=batch_size,
    )
    dm.setup("fit")
    return dm


@pytest.fixture
def sample_datamodule():
    """Small data module used by the loss and mask tests.

    Returns
    -------
    dm : EncoderDecoderTimeSeriesDataModule
        Data module over synthetic univariate series.
    """
    df = _make_dataframe(N_SERIES, N_SAMPLES, seed=0)
    return _make_datamodule(df, CONTEXT_LENGTH, PREDICTION_LENGTH, BATCH_SIZE)


@pytest.fixture
def sample_batch(sample_datamodule):
    """First training batch of :func:`sample_datamodule`.

    Returns
    -------
    batch : tuple of (dict[str, torch.Tensor], torch.Tensor)
        Input dictionary and future target.
    """
    return next(iter(sample_datamodule.train_dataloader()))


def _make_model(metadata, **kwargs):
    """Build a small NHiTS_v2 in eval mode.

    Parameters
    ----------
    metadata : dict
        Data module metadata.
    **kwargs
        Forwarded to :class:`NHiTS_v2`.

    Returns
    -------
    model : NHiTS_v2
        Model with a small architecture, in eval mode.
    """
    kwargs.setdefault("loss", MAE())
    kwargs.setdefault("hidden_size", HIDDEN_SIZE)
    kwargs.setdefault("n_blocks", N_BLOCKS)
    model = NHiTS_v2(metadata=metadata, **kwargs)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# backcast loss weighting
# ---------------------------------------------------------------------------


def test_backcast_weighting_matches_v1_formula(sample_datamodule, sample_batch):
    """The combined loss must use the same weights as the v1 NHiTS.

    v1 derives the backcast weight as ``r * prediction_length /
    context_length``, normalised by ``w / (w + 1)``, and gives the forecast the
    complementary weight. This recomputes those weights independently from the
    raw forecast and backcast losses and compares against ``_compute_loss``.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    sample_batch : tuple
        Fixture providing one training batch.
    """
    ratio = 0.5
    model = _make_model(sample_datamodule.metadata, backcast_loss_ratio=ratio)
    x, y = sample_batch

    with torch.no_grad():
        loss, out = model._compute_loss(x, y, "val")

        encoder_target = x["target_past"].squeeze(-1)
        forecast_loss = MAE()(out["prediction"], y)
        backcast_loss = MAE()(out["backcast"], encoder_target)

    backcast_weight = ratio * PREDICTION_LENGTH / CONTEXT_LENGTH
    backcast_weight = backcast_weight / (backcast_weight + 1)
    expected = (1 - backcast_weight) * forecast_loss + backcast_weight * backcast_loss

    assert torch.allclose(loss, expected), (
        f"combined loss {loss.item():.6f} does not match the v1 weighting "
        f"{expected.item():.6f}"
    )


def test_backcast_loss_ratio_zero_is_forecast_only(sample_datamodule, sample_batch):
    """With ``backcast_loss_ratio=0`` the loss must be the plain forecast loss.

    This pins the default path to the behaviour of the v2 ``BaseModel``, so the
    backcast override cannot silently change models that do not use it.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    sample_batch : tuple
        Fixture providing one training batch.
    """
    model = _make_model(sample_datamodule.metadata, backcast_loss_ratio=0.0)
    x, y = sample_batch

    with torch.no_grad():
        loss, out = model._compute_loss(x, y, "val")
        forecast_loss = MAE()(out["prediction"], y)

    assert torch.allclose(loss, forecast_loss)


def test_train_val_test_steps_share_weighted_loss(sample_datamodule, sample_batch):
    """All three Lightning steps must apply the same weighted loss.

    ``NHiTS_v2`` overrides ``training_step``, ``validation_step`` and
    ``test_step`` so that the backcast term is included everywhere. If one of
    them fell back to the ``BaseModel`` forecast-only loss, the reported values
    would differ.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    sample_batch : tuple
        Fixture providing one training batch.
    """
    model = _make_model(sample_datamodule.metadata, backcast_loss_ratio=0.5)
    x, y = sample_batch

    with torch.no_grad():
        train_loss = model.training_step((x, y), 0)["loss"]
        val_loss = model.validation_step((x, y), 0)["val_loss"]
        test_loss = model.test_step((x, y), 0)["test_loss"]

        forecast_loss = MAE()(model(x)["prediction"], y)

    assert torch.allclose(train_loss, val_loss)
    assert torch.allclose(train_loss, test_loss)
    assert not torch.allclose(train_loss, forecast_loss), (
        "the weighted loss equals the forecast-only loss, so the backcast term "
        "is not contributing"
    )


def test_backcast_loss_ratio_changes_gradients(sample_datamodule, sample_batch):
    """The backcast term must reach the parameters through the graph.

    The same model and batch are differentiated twice, only changing the
    backcast ratio. Identical gradients would mean the backcast loss is
    computed but never actually influences training.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    sample_batch : tuple
        Fixture providing one training batch.
    """
    model = _make_model(sample_datamodule.metadata, backcast_loss_ratio=0.0)
    x, y = sample_batch
    param = next(model.model.parameters())

    model.zero_grad()
    loss_without_backcast, _ = model._compute_loss(x, y, "train")
    loss_without_backcast.backward()
    grad_without_backcast = param.grad.detach().clone()

    model._backcast_loss_ratio = 0.8
    model.zero_grad()
    loss_with_backcast, _ = model._compute_loss(x, y, "train")
    loss_with_backcast.backward()
    grad_with_backcast = param.grad.detach().clone()

    assert not torch.allclose(grad_without_backcast, grad_with_backcast)


# ---------------------------------------------------------------------------
# encoder mask semantics
# ---------------------------------------------------------------------------


def test_encoder_mask_zeroes_backcast_at_masked_positions(
    sample_datamodule, sample_batch
):
    """Masked encoder positions must carry no residual into the backcast.

    ``NHiTSModule`` multiplies the residual by the mask after every block, so
    the returned backcast has to be exactly zero wherever the mask is zero.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    sample_batch : tuple
        Fixture providing one training batch.
    """
    model = _make_model(sample_datamodule.metadata)
    x, _ = sample_batch

    x_masked = dict(x)
    mask = torch.ones(x["target_past"].shape[0], CONTEXT_LENGTH)
    mask[:, :2] = 0.0
    x_masked["encoder_mask"] = mask

    with torch.no_grad():
        backcast = model(x_masked)["backcast"]

    masked_part = backcast[:, :2, :]
    assert torch.equal(masked_part, torch.zeros_like(masked_part))
    assert not torch.equal(
        backcast[:, 2:, :], torch.zeros_like(backcast[:, 2:, :])
    ), "unmasked positions should keep a non-zero residual"


def test_2d_and_3d_encoder_mask_are_equivalent(sample_datamodule, sample_batch):
    """A mask of shape ``(batch, time)`` and ``(batch, time, 1)`` must agree.

    ``forward`` squeezes a trailing singleton dimension before handing the mask
    to ``NHiTSModule``, which unsqueezes it again. Without the squeeze the mask
    would broadcast against the residual with the wrong rank, so this checks the
    two input layouts produce the same forecast rather than merely covering the
    branch.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    sample_batch : tuple
        Fixture providing one training batch.
    """
    model = _make_model(sample_datamodule.metadata)
    x, _ = sample_batch

    mask_2d = torch.ones(x["target_past"].shape[0], CONTEXT_LENGTH)
    mask_2d[:, 0] = 0.0

    x_2d = dict(x)
    x_2d["encoder_mask"] = mask_2d
    x_3d = dict(x)
    x_3d["encoder_mask"] = mask_2d.unsqueeze(-1)

    with torch.no_grad():
        out_2d = model(x_2d)
        out_3d = model(x_3d)

    assert torch.equal(out_2d["prediction"], out_3d["prediction"])
    assert torch.equal(out_2d["backcast"], out_3d["backcast"])


# ---------------------------------------------------------------------------
# NHiTS v1 vs v2 numerical validation
# ---------------------------------------------------------------------------

_CMP_N_SERIES = 4
_CMP_N_SAMPLES = 120
_CMP_CONTEXT = 12
_CMP_PRED = 4
_CMP_EPOCHS = 10
_CMP_SEED = 42


def _make_comparison_dataframe():
    """Shared synthetic DataFrame for the v1/v2 comparisons.

    Returns
    -------
    df : pandas.DataFrame
        Columns ``time_idx``, ``series_id`` and ``value``.
    """
    return _make_dataframe(_CMP_N_SERIES, _CMP_N_SAMPLES, seed=_CMP_SEED)


def _make_v1_dataset(df):
    """Build the v1 ``TimeSeriesDataSet`` used by the comparison tests.

    Parameters
    ----------
    df : pandas.DataFrame
        Data as returned by :func:`_make_comparison_dataframe`.

    Returns
    -------
    dataset : TimeSeriesDataSet
        Dataset with the same windowing as the v2 data module.
    """
    from pytorch_forecasting.data.timeseries import TimeSeriesDataSet

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="value",
        group_ids=["series_id"],
        time_varying_unknown_reals=["value"],
        max_encoder_length=_CMP_CONTEXT,
        max_prediction_length=_CMP_PRED,
        min_encoder_length=_CMP_CONTEXT,
    )


@pytest.mark.parametrize(
    "loss_factory",
    [
        pytest.param(lambda: MAE(), id="point"),
        pytest.param(lambda: QuantileLoss(), id="quantile"),
    ],
)
def test_nhits_v1_v2_identical_outputs_with_shared_weights(loss_factory):
    """v1 and v2 must produce identical module outputs given the same weights.

    Both wrappers delegate to the same ``NHiTSModule``, but each derives the
    module's ``output_size`` itself from the configured loss. Loading the v1
    state dict into the v2 module therefore also asserts that the two derive the
    same output layout: a mismatch raises before the numerical comparison.

    The point and quantile cases are both covered, since the quantile layout is
    where the v2 ``output_size`` derivation differs from the point case.

    Parameters
    ----------
    loss_factory : callable
        Returns a fresh loss instance for each wrapper.
    """
    from pytorch_forecasting.models import NHiTS as NHiTS_v1

    seed = 0
    df = _make_comparison_dataframe()
    dataset_v1 = _make_v1_dataset(df)

    pl.seed_everything(seed)
    model_v1 = NHiTS_v1.from_dataset(
        dataset_v1,
        hidden_size=HIDDEN_SIZE,
        n_blocks=N_BLOCKS,
        loss=loss_factory(),
    )

    dm_v2 = _make_datamodule(df, _CMP_CONTEXT, _CMP_PRED, BATCH_SIZE)

    pl.seed_everything(seed)
    model_v2 = NHiTS_v2(
        loss=loss_factory(),
        metadata=dm_v2.metadata,
        hidden_size=HIDDEN_SIZE,
        n_blocks=N_BLOCKS,
    )

    assert model_v2.model.output_size == model_v1.model.output_size

    model_v2.model.load_state_dict(model_v1.model.state_dict())
    model_v1.eval()
    model_v2.eval()

    torch.manual_seed(seed)
    encoder_y = torch.randn(BATCH_SIZE, _CMP_CONTEXT, 1)
    encoder_mask = torch.ones(BATCH_SIZE, _CMP_CONTEXT)

    with torch.no_grad():
        forecast_v1, backcast_v1, _, _ = model_v1.model(
            encoder_y, encoder_mask, None, None, None
        )
        forecast_v2, backcast_v2, _, _ = model_v2.model(
            encoder_y, encoder_mask, None, None, None
        )

    assert torch.allclose(forecast_v1, forecast_v2), (
        "forecast outputs differ between v1 and v2 despite identical weights, "
        f"max abs diff {(forecast_v1 - forecast_v2).abs().max().item():.2e}"
    )
    assert torch.allclose(backcast_v1, backcast_v2), (
        "backcast outputs differ between v1 and v2 despite identical weights, "
        f"max abs diff {(backcast_v1 - backcast_v2).abs().max().item():.2e}"
    )


def test_nhits_v1_v2_train_loss_comparable():
    """Smoke test against a systematic numerical regression in the v2 rework.

    Both wrappers are trained on the same data for the same number of epochs
    through their own data pipelines, so the losses are not expected to match
    exactly. The test only asserts that both converge and that v2 stays within
    3x of v1, which catches a broken pipeline or loss wiring without being
    sensitive to normalisation differences.
    """
    from pytorch_forecasting.metrics import MAE as MAE_v1
    from pytorch_forecasting.models import NHiTS as NHiTS_v1

    df = _make_comparison_dataframe()

    def _trainer():
        return pl.Trainer(
            max_epochs=_CMP_EPOCHS,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
            limit_val_batches=0,
        )

    dataset_v1 = _make_v1_dataset(df)
    dataloader_v1 = dataset_v1.to_dataloader(train=True, batch_size=8, num_workers=0)

    pl.seed_everything(_CMP_SEED)
    model_v1 = NHiTS_v1.from_dataset(
        dataset_v1,
        hidden_size=32,
        n_blocks=[1, 1, 1],
        learning_rate=1e-3,
        loss=MAE_v1(),
    )
    _trainer().fit(model_v1, dataloader_v1)

    model_v1.eval()
    v1_losses = []
    with torch.no_grad():
        for x, y in dataloader_v1:
            prediction = model_v1(x)["prediction"]
            if prediction.dim() == 3:
                prediction = prediction[..., 0]
            target = y if isinstance(y, torch.Tensor) else y[0]
            v1_losses.append(float(torch.mean(torch.abs(prediction - target))))
    train_mae_v1 = float(np.mean(v1_losses))

    dm_v2 = _make_datamodule(df, _CMP_CONTEXT, _CMP_PRED, 8)

    pl.seed_everything(_CMP_SEED)
    model_v2 = NHiTS_v2(
        loss=MAE(),
        metadata=dm_v2.metadata,
        hidden_size=32,
        n_blocks=[1, 1, 1],
    )
    _trainer().fit(model_v2, dm_v2)

    model_v2.eval()
    v2_losses = []
    with torch.no_grad():
        for x, y in dm_v2.train_dataloader():
            prediction = model_v2(x)["prediction"].squeeze(-1)
            v2_losses.append(float(torch.mean(torch.abs(prediction - y))))
    train_mae_v2 = float(np.mean(v2_losses))

    assert train_mae_v1 < 1.0, f"NHiTS v1 did not converge, MAE {train_mae_v1:.4f}"
    assert train_mae_v2 < 1.0, f"NHiTS v2 did not converge, MAE {train_mae_v2:.4f}"

    ratio = train_mae_v2 / (train_mae_v1 + 1e-8)
    assert ratio < 3.0, (
        f"NHiTS v2 train MAE {train_mae_v2:.4f} is more than 3x the v1 train MAE "
        f"{train_mae_v1:.4f}, possible numerical regression in the v2 rework"
    )
