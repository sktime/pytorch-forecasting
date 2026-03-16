"""Tests for Base_pkg._load_config."""

import pickle
import tempfile

import pytest

from pytorch_forecasting.base._base_pkg import Base_pkg


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
