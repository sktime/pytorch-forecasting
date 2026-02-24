"""Unit tests for ``pytorch_forecasting.callbacks.predict.PredictCallback``.

Two branches in ``on_predict_batch_end`` were introduced by the GPU memory-leak
fix (PR #2086) but were not exercised by the existing test suite:

* Line 77 – the ``"y"`` branch:
      ``self.info[key].append(move_to_device(detach(y[0]), "cpu"))``
* Line 81 – the ``"decoder_lengths"`` branch:
      ``self.info[key].append(x["decoder_lengths"].detach().cpu())``

These tests directly invoke ``on_predict_batch_end`` to trigger each branch and
assert that the stored tensors are detached and reside on CPU.
"""

import pytest
import torch

from pytorch_forecasting.callbacks.predict import PredictCallback


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_batch(batch_size: int = 2, pred_len: int = 4):
    """Return a minimal (x, y) batch pair suitable for raw-mode tests."""
    x = {
        "decoder_lengths": torch.full((batch_size,), pred_len, dtype=torch.long),
    }
    target = torch.randn(batch_size, pred_len)
    index = torch.arange(batch_size)
    return x, (target, index)


def _batch_end(callback: PredictCallback, batch, outputs):
    """Call ``on_predict_batch_end`` with null trainer/pl_module (raw mode)."""
    callback.on_predict_batch_end(
        trainer=None,
        pl_module=None,
        outputs=outputs,
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_return_y_stores_detached_cpu_tensor():
    """``return_info=["y"]`` must append a detached CPU copy of ``y[0]`` (line 77).

    Uses a target tensor with ``requires_grad=True`` so that the detach step
    is observable: the stored tensor must not carry a gradient.
    """
    callback = PredictCallback(mode="raw", return_info=["y"])

    x, (target, index) = _make_batch(batch_size=2, pred_len=4)
    # Attach a gradient so we can verify that detach() was called.
    target_with_grad = target.requires_grad_(True)
    outputs = torch.randn(2, 4)

    _batch_end(callback, (x, (target_with_grad, index)), outputs)

    assert len(callback.info["y"]) == 1, "expected exactly one entry in info['y']"
    stored = callback.info["y"][0]

    assert stored.device.type == "cpu", "stored y[0] must be on CPU"
    assert not stored.requires_grad, "stored y[0] must be detached from the graph"
    assert stored.shape == target_with_grad.shape
    assert torch.allclose(stored, target_with_grad.detach())


def test_return_decoder_lengths_stores_detached_cpu_tensor():
    """``return_info=["decoder_lengths"]`` must append a detached CPU tensor (line 81).

    Verifies that the stored tensor is on CPU, has no gradient, and matches the
    original ``x["decoder_lengths"]`` values.
    """
    callback = PredictCallback(mode="raw", return_info=["decoder_lengths"])

    x, y = _make_batch(batch_size=3, pred_len=5)
    outputs = torch.randn(3, 5)

    _batch_end(callback, (x, y), outputs)

    assert len(callback.info["decoder_lengths"]) == 1, (
        "expected exactly one entry in info['decoder_lengths']"
    )
    stored = callback.info["decoder_lengths"][0]

    assert stored.device.type == "cpu", "stored decoder_lengths must be on CPU"
    assert not stored.requires_grad, "stored decoder_lengths must be detached"
    assert torch.equal(stored, x["decoder_lengths"]), (
        "stored decoder_lengths values must match the original tensor"
    )


def test_return_y_and_decoder_lengths_together():
    """Both ``"y"`` and ``"decoder_lengths"`` branches execute in a single pass.

    Exercises lines 77 and 81 simultaneously to ensure both return_info keys
    are collected correctly when specified together.
    """
    callback = PredictCallback(mode="raw", return_info=["y", "decoder_lengths"])

    x, (target, index) = _make_batch(batch_size=2, pred_len=3)
    target_with_grad = target.requires_grad_(True)
    outputs = torch.randn(2, 3)

    _batch_end(callback, (x, (target_with_grad, index)), outputs)

    # --- "y" branch ---
    assert len(callback.info["y"]) == 1
    stored_y = callback.info["y"][0]
    assert stored_y.device.type == "cpu"
    assert not stored_y.requires_grad

    # --- "decoder_lengths" branch ---
    assert len(callback.info["decoder_lengths"]) == 1
    stored_dl = callback.info["decoder_lengths"][0]
    assert stored_dl.device.type == "cpu"
    assert not stored_dl.requires_grad
    assert torch.equal(stored_dl, x["decoder_lengths"])
