"""Numerical regression tests for the vectorized quantile loss."""

import pytest
import torch

from pytorch_forecasting.metrics import QuantileLoss


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("shape", [(1, 2), (2, 1)])
def test_quantile_loss_finite_residuals(
    dtype: torch.dtype, large: bool, sign: int, shape: tuple[int, int]
) -> None:
    amplitude = (
        {torch.float16: 40000.0, torch.float32: 2e38, torch.float64: 1e308}[dtype]
        if large
        else 2.0
    )
    predictions = torch.full(
        (*shape, 3), -sign * amplitude, dtype=dtype, requires_grad=True
    )
    target = torch.zeros(shape, dtype=dtype)
    result = QuantileLoss(quantiles=[0.25, 0.5, 0.75]).loss(predictions, target)
    # Pinball slopes are q below the target, and 1-q above it.
    weights = [0.25, 0.5, 0.75] if sign > 0 else [0.75, 0.5, 0.25]
    expected = predictions.new_tensor([amplitude * (2 * w) for w in weights]).expand_as(
        result
    )
    torch.testing.assert_close(result, expected)
    gradients = torch.autograd.grad(
        result, predictions, grad_outputs=torch.ones_like(result)
    )[0]
    expected_grad = predictions.new_tensor([-sign * 2 * w for w in weights]).expand_as(
        predictions
    )
    torch.testing.assert_close(gradients, expected_grad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_quantile_loss_zero_residual_subgradient(dtype: torch.dtype) -> None:
    predictions = torch.zeros(1, 1, 3, dtype=dtype, requires_grad=True)
    result = QuantileLoss(quantiles=[0.25, 0.5, 0.75]).loss(
        predictions, torch.zeros(1, 1, dtype=dtype)
    )
    torch.testing.assert_close(result, torch.zeros_like(result))
    gradient = torch.autograd.grad(result.sum(), predictions)[0]
    torch.testing.assert_close(gradient, predictions.new_tensor([[[0.5, 0.0, -0.5]]]))


def test_quantile_loss_metric_keeps_finite_half_loss() -> None:
    prediction = torch.tensor([[[40000.0]]], dtype=torch.float16, requires_grad=True)
    target = torch.zeros(1, 1, dtype=torch.float16)
    result = QuantileLoss(quantiles=[0.75])(prediction, target)
    torch.testing.assert_close(result, torch.tensor(20000.0, dtype=result.dtype))
