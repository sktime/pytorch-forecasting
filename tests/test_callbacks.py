from unittest.mock import MagicMock

import torch

from pytorch_forecasting.callbacks.predict import PredictCallback


def _make_tensor(*shape):
    # Non-leaf tensor so grad_fn is not None before detach, None after
    return torch.zeros(*shape, requires_grad=True) + 0


def _make_batch(batch_size=4, enc_len=10, dec_len=5):
    x = {
        "encoder_cont": _make_tensor(batch_size, enc_len, 2),
        "decoder_cont": _make_tensor(batch_size, dec_len, 1),
        "decoder_lengths": _make_tensor(batch_size).long(),
    }
    y = (_make_tensor(batch_size, dec_len), _make_tensor(batch_size, dec_len))
    return x, y


def _make_trainer():
    return MagicMock()


def _make_pl_module(return_value=None):
    pl_module = MagicMock()
    if return_value is not None:
        pl_module.to_prediction.return_value = return_value
        pl_module.to_quantiles.return_value = return_value
    return pl_module


def test_predictions_moved_to_cpu_prediction_mode():
    """Predictions collected in prediction mode are detached and on CPU."""
    output = _make_tensor(4, 5)
    cb = PredictCallback(mode="prediction")
    batch = _make_batch()
    pl_module = _make_pl_module(return_value=output)

    cb.on_predict_batch_end(_make_trainer(), pl_module, output, batch, batch_idx=0)

    assert len(cb.predictions) == 1
    assert cb.predictions[0].device == torch.device("cpu")
    assert cb.predictions[0].grad_fn is None


def test_raw_mode_dict_moved_to_cpu():
    """Raw mode dict outputs are detached and moved to CPU before collection."""
    outputs = {
        "prediction": _make_tensor(4, 5),
        "output": _make_tensor(4, 5, 2),
    }
    cb = PredictCallback(mode="raw")
    batch = _make_batch()

    cb.on_predict_batch_end(_make_trainer(), MagicMock(), outputs, batch, batch_idx=0)

    assert isinstance(cb.predictions[0], dict)
    for v in cb.predictions[0].values():
        assert v.device == torch.device("cpu")
        assert v.grad_fn is None


def test_return_info_x_moved_to_cpu():
    """When return_info includes 'x', the x dict is detached and on CPU."""
    output = _make_tensor(4, 5)
    cb = PredictCallback(mode="prediction", return_info=["x"])
    batch = _make_batch()
    pl_module = _make_pl_module(return_value=output)

    cb.on_predict_batch_end(_make_trainer(), pl_module, output, batch, batch_idx=0)

    x_stored = cb.info["x"][0]
    assert isinstance(x_stored, dict)
    for v in x_stored.values():
        if isinstance(v, torch.Tensor):
            assert v.device == torch.device("cpu")
            assert v.grad_fn is None


def test_return_info_y_and_decoder_lengths_moved_to_cpu():
    """y[0] and decoder_lengths are detached and on CPU when requested."""
    output = _make_tensor(4, 5)
    cb = PredictCallback(mode="prediction", return_info=["y", "decoder_lengths"])
    batch = _make_batch()
    pl_module = _make_pl_module(return_value=output)

    cb.on_predict_batch_end(_make_trainer(), pl_module, output, batch, batch_idx=0)

    assert cb.info["y"][0].device == torch.device("cpu")
    assert cb.info["y"][0].grad_fn is None
    assert cb.info["decoder_lengths"][0].device == torch.device("cpu")
