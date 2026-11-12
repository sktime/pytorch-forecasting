"""Tests for FlattenHead covering the newly merged ``individual`` option."""

import pytest
import torch

from pytorch_forecasting.layers import FlattenHead

B, N_VARS, D1, D2, TARGET_WINDOW = 2, 4, 8, 16, 12


@pytest.mark.parametrize("n_quantiles", [None, 3])
@pytest.mark.parametrize("individual", [False, True])
def test_output_shape(individual, n_quantiles):
    """Output shape depends only on n_quantiles, not on individual."""
    head = FlattenHead(
        n_vars=N_VARS,
        nf=D1 * D2,
        target_window=TARGET_WINDOW,
        n_quantiles=n_quantiles,
        individual=individual,
    )
    out = head(torch.randn(B, N_VARS, D1, D2))

    if n_quantiles is None:
        assert out.shape == (B, TARGET_WINDOW, N_VARS)
    else:
        assert out.shape == (B, TARGET_WINDOW * N_VARS, n_quantiles)


@pytest.mark.parametrize("n_quantiles", [None, 3])
def test_individual_matches_shared_with_same_weights(n_quantiles):
    """With identical per-variable weights, individual=True equals individual=False.

    Ensures both paths lay out the output identically.
    """
    kwargs = dict(
        n_vars=N_VARS, nf=D1 * D2, target_window=TARGET_WINDOW, n_quantiles=n_quantiles
    )
    shared = FlattenHead(**kwargs, individual=False).eval()
    individual = FlattenHead(**kwargs, individual=True).eval()
    for linear in individual.linears:
        linear.load_state_dict(shared.linear.state_dict())

    x = torch.randn(B, N_VARS, D1, D2)
    assert torch.allclose(shared(x), individual(x), atol=1e-6)
