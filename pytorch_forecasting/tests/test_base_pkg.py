"""Tests for Base_pkg."""

import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest

from pytorch_forecasting.base._base_pkg import Base_pkg
from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.models.temporal_fusion_transformer._tft_pkg_v2 import (
    TFT_pkg_v2,
)


def test_load_config_pkl():
    """Test that _load_config correctly loads a .pkl file path."""
    cfg = {"moving_avg": 25}
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(cfg, f)
        pkl_path = f.name

    result = Base_pkg._load_config(pkl_path)
    assert result == {"moving_avg": 25}


def test_load_config_unsupported_format():
    """Test that _load_config raises ValueError for unsupported formats."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(b"{}")
        json_path = f.name

    with pytest.raises(ValueError, match="Unsupported config format"):
        Base_pkg._load_config(json_path)


def _make_ts(n_cont: int) -> TimeSeries:
    rng = np.random.default_rng(0)
    rows = []
    for g in range(2):
        for t in range(20):
            row = {
                "time_idx": t,
                "group": g,
                "target": float(rng.normal()),
                "weight": 1.0,
            }
            for i in range(n_cont):
                row[f"x{i}"] = float(rng.normal())
            rows.append(row)

    return TimeSeries(
        data=pd.DataFrame(rows),
        time="time_idx",
        target=["target"],
        group=["group"],
        weight="weight",
        num=[f"x{i}" for i in range(n_cont)],
        known=[f"x{i}" for i in range(n_cont)],
        unknown=[],
        static=["group"],
    )


def _make_tft_pkg(**trainer_overrides):
    trainer_cfg = {
        "max_epochs": 1,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "accelerator": "cpu",
    }
    trainer_cfg.update(trainer_overrides)
    return TFT_pkg_v2(
        model_cfg={
            "loss": SMAPE(),
            "hidden_size": 16,
            "attention_head_size": 2,
        },
        datamodule_cfg={
            "max_encoder_length": 8,
            "max_prediction_length": 3,
            "train_val_test_split": (0.8, 0.2),
            "batch_size": 2,
        },
        trainer_cfg=trainer_cfg,
    )


def test_fit_rebuilds_model_for_new_data():
    """Refit with different data must rebuild the model from new metadata."""
    pkg = _make_tft_pkg()

    pkg.fit(_make_ts(2), save_ckpt=False)
    model_after_first = pkg.model

    pkg.fit(_make_ts(5), save_ckpt=False)

    assert pkg.model is not model_after_first
    assert pkg.model is not None


def test_ckpt_metadata_mismatch_raises(tmp_path):
    """Loading a checkpoint with different metadata must fail clearly."""
    pkg = _make_tft_pkg(enable_checkpointing=True)
    ckpt_path = pkg.fit(
        _make_ts(2),
        save_ckpt=True,
        ckpt_dir=tmp_path / "checkpoints",
        ckpt_kwargs={"monitor": "train_loss_epoch"},
    )
    assert ckpt_path is not None

    loaded = TFT_pkg_v2(ckpt_path=ckpt_path)
    bad_metadata = dict(loaded.metadata)
    bad_metadata["encoder_cont"] = bad_metadata["encoder_cont"] + 10

    with pytest.raises(ValueError, match="does not match"):
        loaded._build_model(bad_metadata)


def test_fit_from_ckpt_with_different_data_raises(tmp_path):
    """Refitting a checkpoint-loaded pkg on incompatible data must fail."""
    pkg = _make_tft_pkg(enable_checkpointing=True)
    ckpt_path = pkg.fit(
        _make_ts(2),
        save_ckpt=True,
        ckpt_dir=tmp_path / "checkpoints",
        ckpt_kwargs={"monitor": "train_loss_epoch"},
    )

    loaded = TFT_pkg_v2(
        ckpt_path=ckpt_path,
        trainer_cfg={
            "max_epochs": 1,
            "limit_train_batches": 1,
            "limit_val_batches": 1,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "accelerator": "cpu",
        },
    )

    with pytest.raises(ValueError, match="does not match"):
        loaded.fit(_make_ts(5), save_ckpt=False)
