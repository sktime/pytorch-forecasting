from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from pytorch_forecasting.callbacks.predict import PredictCallback


def _make_tensor(*shape):
    # Non-leaf tensor so grad_fn is not None before detach, None after
    return torch.zeros(*shape, requires_grad=True) + 0


def _make_batch(batch_size=4, enc_len=10, dec_len=5):
    x = {
        "encoder_cont": _make_tensor(batch_size, enc_len, 2),
        "decoder_cont": _make_tensor(batch_size, dec_len, 1),
        "decoder_lengths": torch.full((batch_size,), dec_len, dtype=torch.long),
    }
    y = _make_tensor(batch_size, dec_len)
    return x, y


def _make_trainer(dataset=None):
    return SimpleNamespace(predict_dataloaders=SimpleNamespace(dataset=dataset))


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
    batch[0]["target_scale"] = [_make_tensor(4), _make_tensor(4)]
    batch[0]["__window_idx"] = torch.arange(4)
    pl_module = _make_pl_module(return_value=output)

    cb.on_predict_batch_end(_make_trainer(), pl_module, output, batch, batch_idx=0)

    x_stored = cb.info["x"][0]
    assert isinstance(x_stored, dict)
    assert "__window_idx" not in x_stored
    for v in x_stored.values():
        values = v if isinstance(v, list) else [v]
        for value in values:
            if isinstance(value, torch.Tensor):
                assert value.device == torch.device("cpu")
                assert value.grad_fn is None


def test_return_info_y_keeps_full_batch_and_collates_batches():
    """The full batched target is returned and concatenated across batches."""
    cb = PredictCallback(mode="prediction", return_info=["y", "decoder_lengths"])
    pl_module = _make_pl_module()
    expected_y = []

    for batch_idx, batch_size in enumerate((2, 3)):
        batch = _make_batch(batch_size=batch_size)
        output = _make_tensor(batch_size, 5)
        pl_module.to_prediction.return_value = output
        expected_y.append(batch[1].detach())
        cb.on_predict_batch_end(
            _make_trainer(), pl_module, output, batch, batch_idx=batch_idx
        )

    cb.on_predict_epoch_end(_make_trainer(), pl_module)

    assert torch.equal(cb.result["y"], torch.cat(expected_y))
    assert cb.result["y"].shape == (5, 5)
    assert cb.result["y"].grad_fn is None
    assert cb.result["decoder_lengths"].tolist() == [5] * 5


def test_return_info_y_keeps_multi_target_structure():
    """Multi-target batches stay as a target-wise list after epoch collation."""
    cb = PredictCallback(mode="prediction", return_info=["y"])
    pl_module = _make_pl_module()

    for batch_idx, batch_size in enumerate((2, 1)):
        x, _ = _make_batch(batch_size=batch_size)
        y = [_make_tensor(batch_size, 5), _make_tensor(batch_size, 5)]
        output = _make_tensor(batch_size, 5)
        pl_module.to_prediction.return_value = output
        cb.on_predict_batch_end(
            _make_trainer(), pl_module, output, (x, y), batch_idx=batch_idx
        )

    cb.on_predict_epoch_end(_make_trainer(), pl_module)

    assert isinstance(cb.result["y"], list)
    assert len(cb.result["y"]) == 2
    assert all(target.shape == (3, 5) for target in cb.result["y"])


def test_decoder_lengths_uses_future_length_for_tslib_batch():
    """TSLib future_length is exposed through the decoder_lengths public key."""
    x, y = _make_batch(batch_size=2)
    del x["decoder_lengths"]
    x["future_length"] = torch.tensor([4, 4])
    output = _make_tensor(2, 5)
    cb = PredictCallback(mode="prediction", return_info=["decoder_lengths"])
    pl_module = _make_pl_module(return_value=output)

    cb.on_predict_batch_end(_make_trainer(), pl_module, output, (x, y), batch_idx=0)

    assert cb.info["decoder_lengths"][0].tolist() == [4, 4]


def test_return_info_index_uses_dataset_protocol_and_collates_dataframes():
    """Prediction indices come from the dataset rather than target positions."""

    class DatasetWithIndex:
        def x_to_index(self, x):
            return pd.DataFrame(
                {
                    "time": x["__window_idx"].numpy(),
                    "group": ["a"] * len(x["__window_idx"]),
                }
            )

    dataset = DatasetWithIndex()
    cb = PredictCallback(mode="prediction", return_info=["index", "x"])
    pl_module = _make_pl_module()

    for batch_idx, window_indices in enumerate(([4, 5], [8])):
        x, y = _make_batch(batch_size=len(window_indices))
        x["__window_idx"] = torch.tensor(window_indices)
        output = _make_tensor(len(window_indices), 5)
        pl_module.to_prediction.return_value = output
        cb.on_predict_batch_end(
            _make_trainer(dataset), pl_module, output, (x, y), batch_idx=batch_idx
        )

    cb.on_predict_epoch_end(_make_trainer(dataset), pl_module)

    assert cb.result["index"].to_dict("list") == {
        "time": [4, 5, 8],
        "group": ["a", "a", "a"],
    }
    assert "__window_idx" not in cb.result["x"]


def test_return_info_index_requires_dataset_protocol():
    """Unsupported datasets fail instead of interpreting a target as an index."""
    output = _make_tensor(2, 5)
    cb = PredictCallback(mode="prediction", return_info=["index"])
    batch = _make_batch(batch_size=2)
    pl_module = _make_pl_module(return_value=output)

    with pytest.raises(TypeError, match="implement x_to_index"):
        cb.on_predict_batch_end(_make_trainer(object()), pl_module, output, batch, 0)
