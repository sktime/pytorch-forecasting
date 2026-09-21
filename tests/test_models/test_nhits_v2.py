"""NHiTS v2 tests that the generic v2 test framework does not cover.

The ``forward`` / ``fit`` / ``predict`` contract, checkpointing and the
architecture and loss variants listed in ``NHiTS_pkg_v2.get_test_train_params``
are all exercised by ``pytorch_forecasting.tests.test_all_v2``. This module only
covers behaviour that the framework cannot reach:

* the backcast semantics, since ``NHiTSModule`` returns the residual left after
  all blocks and the wrapper is responsible for turning it into a
  reconstruction of the encoder window;
* the backcast loss weighting, which ``NHiTS_v2`` implements on top of the v2
  ``BaseModel`` loss and which must agree numerically with the v1 ``NHiTS``;
* the encoder mask semantics of the shared module;
* numerical agreement between the v1 and v2 wrappers, at the wrapper level
  rather than only at the shared module they both delegate to.
"""

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_forecasting.models import NHiTS as NHiTS_v1
from pytorch_forecasting.models.nhits._nhits_v2 import NHiTS_v2
from pytorch_forecasting.utils import create_mask

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


def _module_residual(model, x):
    """Call the shared ``NHiTSModule`` directly and return its residual output.

    The module returns the residual left after every block has subtracted its
    backcast, which is what the wrapper has to convert into a reconstruction.

    Parameters
    ----------
    model : NHiTS_v2
        Model whose inner module is called.
    x : dict[str, torch.Tensor]
        Input batch.

    Returns
    -------
    residual : torch.Tensor
        Residual of shape ``(batch_size, context_length, 1)``.
    """
    encoder_mask = x["encoder_mask"]
    if encoder_mask.dim() == 3:
        encoder_mask = encoder_mask.squeeze(-1)
    with torch.no_grad():
        _, residual, _, _ = model.model(
            x["target_past"], encoder_mask, None, None, None
        )
    return residual


def _backcast_weights(ratio, prediction_length, context_length):
    """Return the ``(forecast_weight, backcast_weight)`` pair used by v1 NHiTS.

    Parameters
    ----------
    ratio : float
        Configured ``backcast_loss_ratio``.
    prediction_length : int
        Decoder length.
    context_length : int
        Encoder length.

    Returns
    -------
    forecast_weight : float
        Weight applied to the forecast loss.
    backcast_weight : float
        Weight applied to the backcast loss.
    """
    backcast_weight = ratio * prediction_length / context_length
    backcast_weight = backcast_weight / (backcast_weight + 1)
    return 1 - backcast_weight, backcast_weight


# ---------------------------------------------------------------------------
# backcast semantics
# ---------------------------------------------------------------------------


def test_backcast_is_reconstruction_not_residual(sample_datamodule, sample_batch):
    """``forward`` must return the reconstruction, not the raw module residual.

    ``NHiTSModule`` returns ``encoder_y`` minus the accumulated block backcasts.
    The v1 wrapper converts that into a reconstruction before it is scored
    against the history, and v2 has to do the same, otherwise the backcast loss
    compares a residual against the target and is meaningless.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    sample_batch : tuple
        Fixture providing one training batch.
    """
    model = _make_model(sample_datamodule.metadata)
    x, _ = sample_batch

    residual = _module_residual(model, x)
    with torch.no_grad():
        backcast = model(x)["backcast"]

    assert torch.allclose(backcast, x["target_past"] - residual)
    assert not torch.allclose(backcast, residual), (
        "forward returns the raw module residual, so the backcast loss scores a "
        "residual against the encoder history"
    )


# ---------------------------------------------------------------------------
# backcast loss weighting
# ---------------------------------------------------------------------------


def test_backcast_weighting_matches_v1_formula(sample_datamodule, sample_batch):
    """The combined loss must use the v1 weights over the v1 backcast.

    Both inputs to the weighting are rebuilt independently of ``forward``: the
    reconstruction is derived from the module residual, and the weights are
    recomputed from the configured ratio. A wrong backcast convention or a wrong
    weight therefore both fail here.

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

    residual = _module_residual(model, x)
    reconstruction = x["target_past"] - residual
    encoder_target = x["target_past"].squeeze(-1)

    with torch.no_grad():
        loss, out = model._compute_loss(x, y, "val")
        forecast_loss = MAE()(out["prediction"], y)
        backcast_loss = MAE()(reconstruction, encoder_target)

    forecast_weight, backcast_weight = _backcast_weights(
        ratio, PREDICTION_LENGTH, CONTEXT_LENGTH
    )
    expected = forecast_weight * forecast_loss + backcast_weight * backcast_loss

    assert torch.allclose(loss, expected), (
        f"combined loss {loss.item():.6f} does not match the v1 weighting over "
        f"the reconstruction {expected.item():.6f}"
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


def test_combined_loss_gradient_is_weighted_sum(sample_datamodule, sample_batch):
    """The gradient must be the weighted sum of both loss gradients.

    Comparing gradients at two different ratios would not be enough: changing
    the ratio also rescales the forecast weight, so a detached backcast term
    would still shift the gradient. This instead differentiates each loss term
    on its own and checks that the combined gradient is exactly the weighted
    sum, with a non-zero backcast contribution.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    sample_batch : tuple
        Fixture providing one training batch.
    """
    ratio = 0.8
    model = _make_model(sample_datamodule.metadata, backcast_loss_ratio=ratio)
    x, y = sample_batch
    param = next(model.model.parameters())
    encoder_target = x["target_past"].squeeze(-1)

    def _grad(loss_fn):
        model.zero_grad()
        loss_fn().backward()
        return param.grad.detach().clone()

    forecast_grad = _grad(lambda: MAE()(model(x)["prediction"], y))
    backcast_grad = _grad(lambda: MAE()(model(x)["backcast"], encoder_target))
    combined_grad = _grad(lambda: model._compute_loss(x, y, "train")[0])

    forecast_weight, backcast_weight = _backcast_weights(
        ratio, PREDICTION_LENGTH, CONTEXT_LENGTH
    )
    expected = forecast_weight * forecast_grad + backcast_weight * backcast_grad

    assert backcast_grad.abs().max() > 0, "backcast loss has no gradient at all"
    assert torch.allclose(combined_grad, expected, atol=1e-6), (
        "the combined gradient is not the weighted sum of the two loss "
        "gradients, so the backcast term does not reach the parameters"
    )


# ---------------------------------------------------------------------------
# encoder mask semantics
# ---------------------------------------------------------------------------


def test_masked_positions_reconstruct_the_input_exactly(
    sample_datamodule, sample_batch
):
    """Masked encoder positions must leave no residual in the reconstruction.

    ``NHiTSModule`` multiplies the residual by the mask after every block, so
    the residual is exactly zero where the mask is zero and the reconstruction
    returned by ``forward`` has to equal the input there.

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

    target = x["target_past"]
    assert torch.allclose(backcast[:, :2, :], target[:, :2, :])
    assert not torch.allclose(backcast[:, 2:, :], target[:, 2:, :]), (
        "unmasked positions reconstruct the input exactly, so the mask is not "
        "distinguishable from the unmasked case"
    )


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


def _make_v1_dataset(df, normalize=True):
    """Build the v1 ``TimeSeriesDataSet`` used by the comparison tests.

    Parameters
    ----------
    df : pandas.DataFrame
        Data as returned by :func:`_make_comparison_dataframe`.
    normalize : bool, default=True
        If False, disable the target normalizer so that the batch tensors are
        on the raw data scale and can be fed to the v2 wrapper unchanged.

    Returns
    -------
    dataset : TimeSeriesDataSet
        Dataset with the same windowing as the v2 data module.
    """
    kwargs = {} if normalize else {"target_normalizer": None}
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="value",
        group_ids=["series_id"],
        time_varying_unknown_reals=["value"],
        max_encoder_length=_CMP_CONTEXT,
        max_prediction_length=_CMP_PRED,
        min_encoder_length=_CMP_CONTEXT,
        **kwargs,
    )


def _make_aligned_v1_v2(loss_cls, **model_kwargs):
    """Build a v1 and a v2 wrapper that see exactly the same input.

    The v1 dataset is built without a target normalizer, so its batch is on the
    raw data scale, and the v2 input dictionary is assembled from that same
    batch. The v1 weights are copied into v2, which also asserts that both
    wrappers derive the same module ``output_size`` from the loss.

    Parameters
    ----------
    loss_cls : type
        Loss class, instantiated separately for each wrapper.
    **model_kwargs
        Extra keyword arguments passed to :class:`NHiTS_v2`.

    Returns
    -------
    model_v1 : NHiTS
        v1 wrapper in eval mode.
    model_v2 : NHiTS_v2
        v2 wrapper in eval mode, sharing the v1 weights.
    x_v1 : dict[str, torch.Tensor]
        Batch as produced by the v1 dataset.
    x_v2 : dict[str, torch.Tensor]
        The same batch in the v2 input layout.
    """
    df = _make_comparison_dataframe()
    dataset_v1 = _make_v1_dataset(df, normalize=False)
    dataloader_v1 = dataset_v1.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0
    )
    x_v1, _ = next(iter(dataloader_v1))

    dm_v2 = _make_datamodule(df, _CMP_CONTEXT, _CMP_PRED, BATCH_SIZE)

    pl.seed_everything(0)
    model_v1 = NHiTS_v1.from_dataset(
        dataset_v1,
        hidden_size=HIDDEN_SIZE,
        n_blocks=N_BLOCKS,
        loss=loss_cls(),
    )
    pl.seed_everything(0)
    model_v2 = NHiTS_v2(
        loss=loss_cls(),
        metadata=dm_v2.metadata,
        hidden_size=HIDDEN_SIZE,
        n_blocks=N_BLOCKS,
        **model_kwargs,
    )

    assert model_v2.model.output_size == model_v1.model.output_size

    model_v2.model.load_state_dict(model_v1.model.state_dict())
    model_v1.eval()
    model_v2.eval()

    encoder_mask = create_mask(
        x_v1["encoder_lengths"].max(), x_v1["encoder_lengths"], inverse=True
    )
    x_v2 = {
        "target_past": x_v1["encoder_cont"],
        "encoder_mask": encoder_mask.float(),
    }
    return model_v1, model_v2, x_v1, x_v2


@pytest.mark.parametrize("loss_cls", [MAE, QuantileLoss], ids=["point", "quantile"])
def test_nhits_v1_v2_wrapper_outputs_match(loss_cls):
    """The two wrappers must agree on forecast and backcast, not just the module.

    Both wrappers delegate to the same ``NHiTSModule``, so comparing the module
    directly cannot catch a difference in what the wrapper does around it, such
    as the residual to reconstruction conversion. This feeds identical,
    identically scaled tensors through both wrappers and compares their output.

    Parameters
    ----------
    loss_cls : type
        Loss class under test, covering the point and quantile output layouts.
    """
    model_v1, model_v2, x_v1, x_v2 = _make_aligned_v1_v2(loss_cls)

    with torch.no_grad():
        out_v1 = model_v1(x_v1)
        out_v2 = model_v2(x_v2)

    assert torch.allclose(out_v1["prediction"], out_v2["prediction"]), (
        "forecasts differ between the v1 and v2 wrappers, max abs diff "
        f"{(out_v1['prediction'] - out_v2['prediction']).abs().max().item():.2e}"
    )
    assert torch.allclose(out_v1["backcast"], out_v2["backcast"]), (
        "backcasts differ between the v1 and v2 wrappers, max abs diff "
        f"{(out_v1['backcast'] - out_v2['backcast']).abs().max().item():.2e}"
    )


def test_nhits_v1_v2_weighted_loss_matches():
    """The weighted loss must match the value v1 computes on the same batch.

    This reproduces the v1 combination, backcast loss against the encoder
    history plus the v1 weights, from the v1 wrapper output, and compares it
    against ``_compute_loss`` of v2 on the equivalent input. It covers the
    backcast convention and the weighting together, end to end through both
    wrappers.
    """
    ratio = 0.5
    model_v1, model_v2, x_v1, x_v2 = _make_aligned_v1_v2(MAE, backcast_loss_ratio=ratio)
    y = x_v1["decoder_target"]

    with torch.no_grad():
        out_v1 = model_v1(x_v1)
        forecast_loss_v1 = MAE()(out_v1["prediction"], y)
        backcast_loss_v1 = MAE()(out_v1["backcast"], x_v1["encoder_target"])

        loss_v2, _ = model_v2._compute_loss(x_v2, y, "val")

    forecast_weight, backcast_weight = _backcast_weights(ratio, _CMP_PRED, _CMP_CONTEXT)
    expected = forecast_weight * forecast_loss_v1 + backcast_weight * backcast_loss_v1

    assert torch.allclose(loss_v2, expected), (
        f"v2 weighted loss {loss_v2.item():.6f} does not match the v1 "
        f"combination {expected.item():.6f}"
    )


def test_nhits_v1_v2_train_loss_comparable():
    """Smoke test against a systematic numerical regression in the v2 rework.

    Both wrappers are trained on the same data for the same number of epochs
    through their own data pipelines, so the losses are not expected to match
    exactly. The test only asserts that both converge and that v2 stays within
    3x of v1, which catches a broken pipeline or loss wiring without being
    sensitive to normalisation differences.
    """
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
        loss=MAE(),
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
