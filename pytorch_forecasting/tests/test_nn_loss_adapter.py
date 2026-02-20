import pytest
import torch
import torch.nn as nn

from pytorch_forecasting.metrics import MAE, MultiLoss, NNLossAdapter
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


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


def test_nn_loss_adapter_h_error():
    adapter = NNLossAdapter(nn.MSELoss())
    y_pred = torch.randn(4, 5, 2)  # H=2, but single target expected
    target = torch.randn(4, 5)

    with pytest.raises(ValueError, match="only supports point predictions"):
        adapter(y_pred, target)


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
