"""Tests for BaseModel v2.

Covers:
- pretrain() sets is_pretrained_ = True
- _pretrain() is called by pretrain()
- load_pretrained_weights() loads checkpoint and sets is_pretrained_
- subclass can override _pretrain() for custom logic
- on_fit_start() freezes backbone when is_pretrained_ is True
- _freeze_backbone() and _unfreeze_backbone()
- _post_init_load_pretrained() loads weights when path is given
- _resolve_checkpoint_path() raises ValueError for invalid hf:// URI
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
from pytorch_forecasting.data.timeseries import TimeSeries
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.models.base._base_model_v2 import BaseModel

# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing
# ---------------------------------------------------------------------------


class _MinimalModel(BaseModel):
    """Minimal BaseModel subclass for testing pretrain hook.

    Uses encoder_cont from the DataModule batch format.
    Falls back to target (y) shape if no continuous features exist.
    """

    def __init__(self, input_size=1, prediction_length=3, hidden_size=8, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.prediction_length = prediction_length
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, prediction_length)

    def forward(self, x):
        # DataModule returns encoder_cont: (batch, seq_len, n_features)
        enc = x["encoder_cont"]  # (batch, seq_len, n_features)
        last = enc[:, -1, : self.input_size]  # (batch, input_size)
        pred = self.linear(last)  # (batch, prediction_length)
        pred = pred.unsqueeze(-1)  # (batch, prediction_length, 1)
        return {"prediction": pred}


class _CustomPretrain(BaseModel):
    """Subclass that overrides _pretrain() for custom logic testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linear = nn.Linear(1, 3)
        self._pretrain_called = False

    def forward(self, x):
        enc = x["encoder_cont"][:, -1, :1]  # (batch, 1)
        pred = self.linear(enc).unsqueeze(-1)  # (batch, 3, 1)
        return {"prediction": pred}

    def _pretrain(self, datamodule, trainer_kwargs=None):
        self._pretrain_called = True
        super()._pretrain(datamodule, trainer_kwargs=trainer_kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MAX_ENCODER_LENGTH = 10
MAX_PREDICTION_LENGTH = 3
BATCH_SIZE = 4


@pytest.fixture(scope="module")
def sample_datamodule():
    """Small synthetic DataModule for pretrain tests."""
    n_groups = 4
    series_len = MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH + 2
    rows = []
    for g in range(n_groups):
        for t in range(series_len):
            rows.append(
                {
                    "time_idx": t,
                    "group_id": f"g{g}",
                    "target": float(np.sin(t / 3.0) + g * 0.1),
                    "feature": float(np.cos(t / 3.0)),
                }
            )
    df = pd.DataFrame(rows)

    ts = TimeSeries(
        data=df,
        time="time_idx",
        target="target",
        group=["group_id"],
        num=["feature"],
    )
    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=ts,
        batch_size=BATCH_SIZE,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        train_val_test_split=(0.7, 0.15, 0.15),
    )
    return dm


@pytest.fixture
def minimal_model():
    return _MinimalModel(
        input_size=1,  # one continuous feature column
        prediction_length=MAX_PREDICTION_LENGTH,
        hidden_size=8,
        loss=MAE(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pretrain_sets_is_pretrained(minimal_model, sample_datamodule):
    """pretrain() must set is_pretrained_ = True."""
    assert not hasattr(minimal_model, "is_pretrained_")
    minimal_model.pretrain(
        sample_datamodule,
        trainer_kwargs={"max_epochs": 1, "enable_progress_bar": False},
    )
    assert hasattr(minimal_model, "is_pretrained_")
    assert minimal_model.is_pretrained_ is True


def test_pretrain_returns_self(minimal_model, sample_datamodule):
    """pretrain() must return self for method chaining."""
    result = minimal_model.pretrain(
        sample_datamodule,
        trainer_kwargs={"max_epochs": 1, "enable_progress_bar": False},
    )
    assert result is minimal_model


def test_custom_pretrain_hook_called(sample_datamodule):
    """Subclass _pretrain() override must be called by pretrain()."""
    model = _CustomPretrain(loss=MAE())
    assert not model._pretrain_called

    model.pretrain(
        sample_datamodule,
        trainer_kwargs={"max_epochs": 1, "enable_progress_bar": False},
    )

    assert model._pretrain_called
    assert model.is_pretrained_ is True


def test_weights_preserved_after_pretrain(minimal_model, sample_datamodule):
    """Weights after pretrain must differ from random init."""
    weights_before = {k: v.clone() for k, v in minimal_model.state_dict().items()}

    minimal_model.pretrain(
        sample_datamodule,
        trainer_kwargs={"max_epochs": 2, "enable_progress_bar": False},
    )

    weights_after = minimal_model.state_dict()
    # At least one parameter must have changed
    any_changed = any(
        not torch.allclose(weights_before[k], weights_after[k]) for k in weights_before
    )
    assert any_changed, "Weights unchanged after pretrain — training may not have run"


def test_load_pretrained_weights(minimal_model, sample_datamodule):
    """load_pretrained_weights() must restore weights and set is_pretrained_."""
    minimal_model.pretrain(
        sample_datamodule,
        trainer_kwargs={"max_epochs": 1, "enable_progress_bar": False},
    )
    pretrained_state = {k: v.clone() for k, v in minimal_model.state_dict().items()}

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    try:
        torch.save(minimal_model.state_dict(), ckpt_path)

        fresh_model = _MinimalModel(
            input_size=1,
            prediction_length=MAX_PREDICTION_LENGTH,
            hidden_size=8,
            loss=MAE(),
        )
        assert not hasattr(fresh_model, "is_pretrained_")

        fresh_model.load_pretrained_weights(ckpt_path)
    finally:
        os.unlink(ckpt_path)

    assert fresh_model.is_pretrained_ is True
    for k in pretrained_state:
        assert torch.allclose(
            pretrained_state[k], fresh_model.state_dict()[k]
        ), f"Weight mismatch for {k} after load_pretrained_weights"


def test_load_pretrained_weights_lightning_checkpoint(minimal_model, sample_datamodule):
    """load_pretrained_weights() must handle lightning-style checkpoints."""
    minimal_model.pretrain(
        sample_datamodule,
        trainer_kwargs={"max_epochs": 1, "enable_progress_bar": False},
    )

    lightning_ckpt = {"state_dict": minimal_model.state_dict(), "epoch": 1}

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        ckpt_path = f.name
    try:
        torch.save(lightning_ckpt, ckpt_path)

        fresh_model = _MinimalModel(
            input_size=1,
            prediction_length=MAX_PREDICTION_LENGTH,
            hidden_size=8,
            loss=MAE(),
        )
        fresh_model.load_pretrained_weights(ckpt_path)
    finally:
        os.unlink(ckpt_path)

    assert fresh_model.is_pretrained_ is True


def test_on_fit_start_freezes_backbone(minimal_model, sample_datamodule):
    """on_fit_start() must freeze backbone when is_pretrained_ is True."""
    minimal_model.pretrain(
        sample_datamodule,
        trainer_kwargs={"max_epochs": 1, "enable_progress_bar": False},
    )
    assert minimal_model.is_pretrained_ is True

    minimal_model.fine_tune_strategy = "freeze_backbone"
    minimal_model.on_fit_start()

    for param in minimal_model.parameters():
        assert not param.requires_grad, "Parameter should be frozen after on_fit_start"


def test_on_fit_start_no_freeze_when_not_pretrained(minimal_model):
    """on_fit_start() must not freeze params when model is not pretrained."""
    assert not getattr(minimal_model, "is_pretrained_", False)
    minimal_model.on_fit_start()

    for param in minimal_model.parameters():
        assert param.requires_grad, "Parameter should remain trainable"


def test_freeze_and_unfreeze_backbone(minimal_model):
    """_freeze_backbone() and _unfreeze_backbone() must toggle requires_grad."""
    minimal_model._freeze_backbone()
    for param in minimal_model.parameters():
        assert not param.requires_grad

    minimal_model._unfreeze_backbone()
    for param in minimal_model.parameters():
        assert param.requires_grad


def test_post_init_load_pretrained(minimal_model, sample_datamodule):
    """_post_init_load_pretrained() must load weights when path is set."""
    minimal_model.pretrain(
        sample_datamodule,
        trainer_kwargs={"max_epochs": 1, "enable_progress_bar": False},
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    try:
        torch.save(minimal_model.state_dict(), ckpt_path)

        fresh_model = _MinimalModel(
            input_size=1,
            prediction_length=MAX_PREDICTION_LENGTH,
            hidden_size=8,
            loss=MAE(),
        )
        fresh_model._pretrained_weights_path = ckpt_path
        fresh_model._post_init_load_pretrained()
    finally:
        os.unlink(ckpt_path)

    assert fresh_model.is_pretrained_ is True


def test_post_init_load_pretrained_no_path(minimal_model):
    """_post_init_load_pretrained() must be a no-op when path is None."""
    assert minimal_model._pretrained_weights_path is None
    minimal_model._post_init_load_pretrained()  # must not raise
    assert not hasattr(minimal_model, "is_pretrained_")


def test_resolve_checkpoint_path_local(tmp_path):
    """_resolve_checkpoint_path() must return local paths unchanged."""
    local_path = str(tmp_path / "model.pt")
    result = BaseModel._resolve_checkpoint_path(local_path)
    assert result == local_path


def test_resolve_checkpoint_path_hf_uri():
    """_resolve_checkpoint_path() raises for hf:// URI (bad format or missing dep)."""
    # Without huggingface_hub installed: ImportError
    # With huggingface_hub installed but malformed path: ValueError
    with pytest.raises((ImportError, ValueError)):
        BaseModel._resolve_checkpoint_path("hf://only-one-part")


def test_resolve_checkpoint_path_hf_invalid_format_mocked():
    """_resolve_checkpoint_path() raises ValueError for malformed hf:// URI."""
    from unittest.mock import MagicMock, patch

    mock_hub = MagicMock()
    with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
        with pytest.raises(ValueError, match="HuggingFace path must be"):
            BaseModel._resolve_checkpoint_path("hf://only-one-part")


def test_resolve_checkpoint_path_hf_valid_mocked():
    """_resolve_checkpoint_path() calls hf_hub_download for valid hf:// URI."""
    from unittest.mock import MagicMock, patch

    fake_local_path = "/fake/local/model.pt"
    mock_hub = MagicMock()
    mock_hub.hf_hub_download.return_value = fake_local_path
    with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
        result = BaseModel._resolve_checkpoint_path("hf://myorg/myrepo/model.pt")

    mock_hub.hf_hub_download.assert_called_once_with(
        repo_id="myorg/myrepo", filename="model.pt"
    )
    assert result == fake_local_path
