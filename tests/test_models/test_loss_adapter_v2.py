import pytest
import torch
import torch.nn as nn

from pytorch_forecasting.models.base._base_model_v2 import BaseModel
from pytorch_forecasting.models.base._loss_adapter_v2 import NNLossAdapter


class DummyV2Model(BaseModel):
    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"prediction": x["prediction"]}


def test_nn_loss_adapter_same_shape_squeezes_last_dim():
    loss = NNLossAdapter(nn.MSELoss())

    y_pred = torch.tensor([[[1.0], [3.0]]])
    target = torch.tensor([[2.0, 1.0]])

    actual = loss(y_pred, target)
    expected = nn.MSELoss()(y_pred.squeeze(-1), target)

    assert torch.isclose(actual, expected)


def test_nn_loss_adapter_class_index_flattens_for_cross_entropy():
    loss = NNLossAdapter(nn.CrossEntropyLoss())

    y_pred = torch.tensor(
        [
            [[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]],
            [[-0.2, 1.2, 0.0], [1.0, 0.0, 0.5]],
        ]
    )
    target = torch.tensor([[0, 1], [1, 2]])

    actual = loss(y_pred, target)
    expected = nn.CrossEntropyLoss()(y_pred.reshape(-1, 3), target.reshape(-1).long())

    assert torch.isclose(actual, expected)


def test_base_model_multi_target_single_nn_loss_is_applied_per_target():
    model = DummyV2Model(loss=nn.MSELoss())

    y_hat = torch.tensor(
        [
            [[1.0, 5.0], [3.0, 7.0]],
            [[2.0, 6.0], [4.0, 8.0]],
        ]
    )
    target_0 = torch.tensor([[0.0, 2.0], [1.0, 3.0]])
    target_1 = torch.tensor([[4.0, 6.0], [5.0, 7.0]])

    actual = model._compute_loss(y_hat, ([target_0, target_1], None))

    expected_0 = nn.MSELoss()(y_hat[..., 0], target_0)
    expected_1 = nn.MSELoss()(y_hat[..., 1], target_1)
    expected = (expected_0 + expected_1) / 2

    assert torch.isclose(actual, expected)


def test_nn_loss_adapter_same_shape_raises_on_mismatch():
    loss = NNLossAdapter(nn.MSELoss())
    y_pred = torch.randn(2, 3, 2)
    target = torch.randn(2, 3)

    with pytest.raises(ValueError, match="Shape mismatch"):
        loss(y_pred, target)
