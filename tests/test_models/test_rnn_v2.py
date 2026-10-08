"""Unit tests for RecurrentNetwork_v2 (v2 interface)."""

import warnings

import pytest
import torch

from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_forecasting.models.rnn._rnn_v2 import RecurrentNetwork_v2


def _make_model(**kwargs):
    """Create a minimal RecurrentNetwork_v2 with default metadata."""
    metadata = {
        "max_encoder_length": 6,
        "max_prediction_length": 2,
        "encoder_cont": 3,
    }
    defaults = {
        "loss": MAE(),
        "cell_type": "LSTM",
        "hidden_size": 8,
        "rnn_layers": 1,
        "dropout": 0.0,
        "metadata": metadata,
    }
    defaults.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return RecurrentNetwork_v2(**defaults)


def test_invalid_cell_type_raises():
    """Invalid cell_type must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid cell_type"):
        _make_model(cell_type="RNN")


def test_gru_cell_type():
    """GRU cell type should construct successfully."""
    model = _make_model(cell_type="GRU")
    assert model.cell_type == "GRU"


def test_quantile_output_size():
    """With QuantileLoss, n_quantiles and output_size should be set correctly."""
    model = _make_model(loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]))
    assert model.n_quantiles == 3
    assert model.output_size == model.max_prediction_length * 3


def _make_batch(batch_size, seq_len, encoder_cont_dim, has_target_past):
    x = {}
    if encoder_cont_dim > 0:
        x["encoder_cont"] = torch.randn(batch_size, seq_len, encoder_cont_dim)
    if has_target_past:
        x["target_past"] = torch.randn(batch_size, seq_len)
    return x


def test_build_input_tensor_both_sources():
    """encoder_cont + target_past -> concatenated tensor (batch, seq, cont_dim+1)."""
    model = _make_model()
    x = _make_batch(4, 6, encoder_cont_dim=3, has_target_past=True)
    inp = model._build_input_tensor(x)
    assert inp.shape == (4, 6, 4)


def test_build_input_tensor_only_target_past():
    """Only target_past -> (batch, seq, 1)."""
    model = _make_model()
    x = {"target_past": torch.randn(4, 6)}
    inp = model._build_input_tensor(x)
    assert inp.shape == (4, 6, 1)


def test_build_input_tensor_only_encoder_cont():
    """Only encoder_cont -> encoder_cont tensor unchanged."""
    model = _make_model()
    x = {"encoder_cont": torch.randn(4, 6, 3)}
    inp = model._build_input_tensor(x)
    assert inp.shape == (4, 6, 3)


def test_build_input_tensor_neither_raises():
    """No target_past and no encoder_cont -> KeyError."""
    model = _make_model()
    with pytest.raises(KeyError, match="Neither"):
        model._build_input_tensor({})


def test_forward_prediction_shape_mae():
    """MAE loss: prediction shape (batch, pred_len, 1)."""
    model = _make_model()
    x = _make_batch(4, 6, encoder_cont_dim=3, has_target_past=True)
    out = model(x)
    assert "prediction" in out
    assert out["prediction"].shape == (4, 2, 1)


def test_forward_prediction_shape_quantile():
    """QuantileLoss: prediction shape (batch, pred_len, n_quantiles)."""
    model = _make_model(loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]))
    x = _make_batch(4, 6, encoder_cont_dim=3, has_target_past=True)
    out = model(x)
    assert out["prediction"].shape == (4, 2, 3)
