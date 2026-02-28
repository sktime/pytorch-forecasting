import torch
from pytorch_forecasting.metrics.quantile import QuantileLoss


def test_quantile_loss_handles_2d_input():
    loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    y_pred = torch.randn(8, 3)
    target = torch.randn(8)
    loss = loss_fn.loss(y_pred, target)
    assert loss.shape == (8, 3)


def test_quantile_loss_handles_3d_input():
    loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    y_pred = torch.randn(8, 5, 3)
    target = torch.randn(8, 5)
    loss = loss_fn.loss(y_pred, target)
    assert loss.shape == (8, 5, 3)