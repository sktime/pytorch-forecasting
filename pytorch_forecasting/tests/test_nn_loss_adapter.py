"""Tests for :class:`~pytorch_forecasting.metrics.NNLossAdapter`.

The adapter supports three families of ``torch.nn`` losses commonly used with
forecasting models:

* **point** — same-shape ``[B, T]`` after squeeze (``MSELoss``, ``L1Loss``, …)
* **class** — logits ``[B, T, C]`` vs labels ``[B, T]`` (``CrossEntropyLoss``,
  ``NLLLoss``)
* **gaussian_nll** — mean/var head ``[B, T, 2]`` (``GaussianNLLLoss``)
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_forecasting.metrics import MAE, MultiLoss, NNLossAdapter
from pytorch_forecasting.models.base._base_model_v2 import BaseModel

POINT_NN_LOSSES = [
    nn.MSELoss(),
    nn.L1Loss(),
    nn.SmoothL1Loss(),
    nn.HuberLoss(),
    nn.BCELoss(),
    nn.BCEWithLogitsLoss(),
    nn.PoissonNLLLoss(log_input=True),
]

CLASS_NN_LOSSES = [
    nn.CrossEntropyLoss(),
    nn.NLLLoss(),
]


def _point_tensors(loss_fn: nn.Module, batch=4, time=5):
    """Build pred/target tensors suitable for a point nn loss."""
    if isinstance(loss_fn, nn.BCELoss):
        y_pred = torch.rand(batch, time, 1).clamp(1e-4, 1 - 1e-4)
        target = torch.randint(0, 2, (batch, time), dtype=torch.float32)
    elif isinstance(loss_fn, nn.BCEWithLogitsLoss):
        y_pred = torch.randn(batch, time, 1)
        target = torch.randint(0, 2, (batch, time), dtype=torch.float32)
    elif isinstance(loss_fn, nn.PoissonNLLLoss):
        y_pred = torch.randn(batch, time, 1)
        target = torch.randint(0, 5, (batch, time), dtype=torch.float32)
    else:
        y_pred = torch.randn(batch, time, 1)
        target = torch.randn(batch, time)
    return y_pred, target


@pytest.mark.parametrize("loss_fn", POINT_NN_LOSSES, ids=lambda x: type(x).__name__)
def test_nn_loss_adapter_all_point_losses(loss_fn):
    """Adapter works for common same-shape nn losses (with and without weights)."""
    adapter = NNLossAdapter(loss_fn)
    y_pred, target = _point_tensors(loss_fn)

    loss = adapter(y_pred, target)
    expected = loss_fn(y_pred.squeeze(-1), target)
    assert loss.ndim == 0
    assert torch.allclose(loss, expected)

    weight = torch.rand_like(target)
    loss_weighted = adapter(y_pred, (target, weight))
    assert loss_weighted.ndim == 0
    assert torch.isfinite(loss_weighted)


@pytest.mark.parametrize("loss_fn", CLASS_NN_LOSSES, ids=lambda x: type(x).__name__)
def test_nn_loss_adapter_all_class_losses(loss_fn):
    """CrossEntropy / NLL accept logits [B, T, C] and integer labels [B, T]."""
    adapter = NNLossAdapter(loss_fn)
    n_classes = 4
    y_pred = torch.randn(3, 5, n_classes, requires_grad=True)
    target = torch.randint(0, n_classes, (3, 5))

    loss = adapter(y_pred, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    flat_pred = y_pred.reshape(-1, n_classes)
    flat_target = target.reshape(-1)
    if isinstance(loss_fn, nn.CrossEntropyLoss):
        expected = loss_fn(flat_pred, flat_target)
    else:
        expected = loss_fn(F.log_softmax(flat_pred, dim=-1), flat_target)
    assert torch.allclose(loss, expected)

    loss.backward()
    assert y_pred.grad is not None

    weight = torch.rand(3, 5)
    loss_w = adapter(y_pred.detach(), (target, weight))
    assert loss_w.ndim == 0
    assert torch.isfinite(loss_w)


def test_nn_loss_adapter_gaussian_nll():
    """GaussianNLLLoss uses last dim as (mean, raw_variance)."""
    adapter = NNLossAdapter(nn.GaussianNLLLoss())
    y_pred = torch.randn(4, 5, 2, requires_grad=True)
    target = torch.randn(4, 5)

    loss = adapter(y_pred, target)
    mean = y_pred[..., 0]
    var = F.softplus(y_pred[..., 1]) + 1e-6
    expected = nn.GaussianNLLLoss()(mean, target, var)
    assert torch.allclose(loss, expected)

    loss.backward()
    assert y_pred.grad is not None
    assert adapter.to_prediction(y_pred.detach()).shape == (4, 5)


def test_nn_loss_adapter_point_rejects_multi_output_head():
    adapter = NNLossAdapter(nn.MSELoss())
    y_pred = torch.randn(4, 5, 2)
    target = torch.randn(4, 5)

    with pytest.raises(ValueError, match="H=1"):
        adapter(y_pred, target)


def test_nn_loss_adapter_single_target():
    loss_fn = nn.MSELoss()
    adapter = NNLossAdapter(loss_fn)

    y_pred = torch.randn(4, 5, 1)  # [B, T, H]
    target = torch.randn(4, 5)  # [B, T]

    # Test without weights
    loss = adapter(y_pred, target)
    expected_loss = loss_fn(y_pred.squeeze(-1), target)
    assert torch.allclose(loss, expected_loss)

    # Test with weights
    weight = torch.rand(4, 5)
    loss_weighted = adapter(y_pred, (target, weight))

    # Manual weighted mean
    raw_loss = nn.MSELoss(reduction="none")(y_pred.squeeze(-1), target)
    expected_weighted_loss = (raw_loss * weight).sum() / weight.sum()
    assert torch.allclose(loss_weighted, expected_weighted_loss)


def test_nn_loss_adapter_multi_target():
    loss_fn = nn.MSELoss()
    adapter = NNLossAdapter(loss_fn)

    y_pred = torch.randn(4, 5, 2)  # [B, T, N]
    targets = [torch.randn(4, 5), torch.randn(4, 5)]  # List of [B, T]

    # Test without weights
    loss = adapter(y_pred, (targets, None))
    expected_loss = loss_fn(y_pred[..., 0], targets[0]) + loss_fn(
        y_pred[..., 1], targets[1]
    )
    assert torch.allclose(loss, expected_loss)

    # Test with weights
    weight = torch.rand(4, 5)
    loss_weighted = adapter(y_pred, (targets, weight))

    # Manual weighted mean for each target then sum
    raw_loss0 = nn.MSELoss(reduction="none")(y_pred[..., 0], targets[0])
    raw_loss1 = nn.MSELoss(reduction="none")(y_pred[..., 1], targets[1])
    expected_weighted_loss = (raw_loss0 * weight).sum() / weight.sum() + (
        raw_loss1 * weight
    ).sum() / weight.sum()
    assert torch.allclose(loss_weighted, expected_weighted_loss)


def test_nn_loss_adapter_mismatch_error():
    adapter = NNLossAdapter(nn.MSELoss())
    y_pred = torch.randn(4, 5, 2)  # N=2
    targets = [torch.randn(4, 5)]  # N=1

    with pytest.raises(ValueError, match="does not match number of targets"):
        adapter(y_pred, (targets, None))


def test_base_model_auto_wrap():
    class SimpleModel(BaseModel):
        def forward(self, x):
            return {"prediction": torch.randn(4, 5, 1)}

    # Should wrap
    model = SimpleModel(loss=nn.MSELoss())
    assert isinstance(model.loss, NNLossAdapter)

    # Should NOT wrap
    model_ptf = SimpleModel(loss=MAE())
    assert isinstance(model_ptf.loss, MAE)

    # Should NOT wrap MultiLoss
    model_multi = SimpleModel(loss=MultiLoss([MAE()]))
    assert isinstance(model_multi.loss, MultiLoss)


def test_nn_loss_adapter_to_prediction():
    adapter = NNLossAdapter(nn.MSELoss())
    y_pred = torch.randn(4, 5, 1)

    out = adapter.to_prediction(y_pred)
    assert out.shape == (4, 5)
    assert torch.allclose(out, y_pred.squeeze(-1))

    y_pred_2d = torch.randn(4, 5)
    out_2d = adapter.to_prediction(y_pred_2d)
    assert out_2d.shape == (4, 5)
    assert torch.allclose(out_2d, y_pred_2d)

    ce_adapter = NNLossAdapter(nn.CrossEntropyLoss())
    logits = torch.tensor([[[0.1, 2.0, 0.0], [3.0, 0.0, 0.0]]])
    assert torch.equal(ce_adapter.to_prediction(logits), torch.tensor([[1, 0]]))


def test_nn_loss_adapter_reduction_sum():
    loss_fn = nn.MSELoss(reduction="sum")
    adapter = NNLossAdapter(loss_fn)

    y_pred = torch.randn(4, 5, 1)
    target = torch.randn(4, 5)
    weight = torch.rand(4, 5)

    loss = adapter(y_pred, (target, weight))

    raw_loss = nn.MSELoss(reduction="none")(y_pred.squeeze(-1), target)
    expected_loss = (raw_loss * weight).sum()
    assert torch.allclose(loss, expected_loss)
    assert loss_fn.reduction == "sum"  # Check it was restored


def test_nn_loss_adapter_list_pred_error():
    adapter = NNLossAdapter(nn.MSELoss())
    y_pred = [torch.randn(4, 5), torch.randn(4, 5)]
    target = torch.randn(4, 5)

    with pytest.raises(
        ValueError,
        match="does not support list of predictions with single target tensor",
    ):
        adapter(y_pred, target)


def test_tft_cross_entropy_on_discrete_target(tmp_path):
    """TFT + CrossEntropyLoss works when the target is discrete class labels."""
    from lightning.pytorch.loggers import TensorBoardLogger

    from pytorch_forecasting.models.temporal_fusion_transformer._tft_pkg_v2 import (
        TFT_pkg_v2,
    )
    from pytorch_forecasting.tests._data_scenarios import (
        data_with_covariates_v2,
        make_datasets_v2,
    )

    raw = data_with_covariates_v2()
    # special_event_1 is binary col already present
    raw["class_label"] = raw["special_event_1"].astype("int64")
    n_classes = int(raw["class_label"].nunique())

    datasets = make_datasets_v2(raw, target="class_label")
    dm_cfg = {
        "max_encoder_length": 4,
        "max_prediction_length": 3,
        "batch_size": 2,
        "train_val_test_split": (0.8, 0.2),
        "add_relative_time_idx": True,
    }
    pkg = TFT_pkg_v2(
        model_cfg=dict(
            loss=nn.CrossEntropyLoss(),
            output_size=n_classes,
            hidden_size=16,
            attention_head_size=2,
        ),
        trainer_cfg={
            "max_epochs": 1,
            "limit_train_batches": 2,
            "limit_val_batches": 1,
            "accelerator": "cpu",
            "enable_checkpointing": False,
            "logger": TensorBoardLogger(str(tmp_path)),
            "default_root_dir": str(tmp_path),
        },
        datamodule_cfg=dm_cfg,
    )
    pkg.fit(datasets["training_dataset"], save_ckpt=False)

    pred = pkg.predict(datasets["validation_dataset"], mode="raw")
    assert pred["prediction"].ndim == 3
    assert pred["prediction"].shape[-1] == n_classes
