"""
Unit tests for the unified FlattenHead layer.

Covers all four combinations of individual={True,False} × n_quantiles={None,k}
and verifies output shapes match the documented contract.
"""

import pytest
import torch

from pytorch_forecasting.layers._output._flatten_head import FlattenHead


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

BATCH = 2
N_VARS = 4
NF1 = 8   # first "patch" dim
NF2 = 16  # second "feature" dim; NF = NF1 * NF2
NF = NF1 * NF2
TARGET_WINDOW = 12
N_QUANTILES = 3


def _make_input():
    """Return a 4-D tensor of shape (B, n_vars, NF1, NF2)."""
    return torch.randn(BATCH, N_VARS, NF1, NF2)


# ---------------------------------------------------------------------------
# output shape tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("individual", [False, True])
def test_output_shape_no_quantiles(individual):
    """n_quantiles=None → (B, target_window, n_vars)."""
    head = FlattenHead(
        n_vars=N_VARS,
        nf=NF,
        target_window=TARGET_WINDOW,
        individual=individual,
    )
    out = head(_make_input())
    assert out.shape == (BATCH, TARGET_WINDOW, N_VARS), (
        f"individual={individual}: expected {(BATCH, TARGET_WINDOW, N_VARS)}, got {out.shape}"
    )


@pytest.mark.parametrize("individual", [False, True])
def test_output_shape_with_quantiles(individual):
    """n_quantiles=k → (B, target_window * n_vars, k)."""
    head = FlattenHead(
        n_vars=N_VARS,
        nf=NF,
        target_window=TARGET_WINDOW,
        n_quantiles=N_QUANTILES,
        individual=individual,
    )
    out = head(_make_input())
    expected = (BATCH, TARGET_WINDOW * N_VARS, N_QUANTILES)
    assert out.shape == expected, (
        f"individual={individual}: expected {expected}, got {out.shape}"
    )


# ---------------------------------------------------------------------------
# backward-compat: individual=False must match old FlattenHead
# ---------------------------------------------------------------------------


def test_individual_false_no_quantiles_matches_legacy():
    """individual=False, n_quantiles=None — same module structure as the
    original FlattenHead (single flatten/linear/dropout + permute)."""
    head = FlattenHead(n_vars=N_VARS, nf=NF, target_window=TARGET_WINDOW)
    assert hasattr(head, "flatten")
    assert hasattr(head, "linear")
    assert hasattr(head, "dropout")
    assert not hasattr(head, "linears")  # no per-var list


def test_individual_true_uses_per_var_modules():
    """individual=True — ModuleList of flatten/linear/dropout per variable."""
    head = FlattenHead(
        n_vars=N_VARS, nf=NF, target_window=TARGET_WINDOW, individual=True
    )
    assert hasattr(head, "linears")
    assert len(head.linears) == N_VARS
    assert len(head.flattens) == N_VARS
    assert len(head.dropouts) == N_VARS
    assert not hasattr(head, "linear")  # no shared linear


# ---------------------------------------------------------------------------
# deprecated alias
# ---------------------------------------------------------------------------


def test_flatten_head_deprecated_alias():
    """Flatten_Head emits DeprecationWarning and produces the same output."""
    import warnings

    from pytorch_forecasting.layers._head._flatten_head import Flatten_Head

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old_head = Flatten_Head(
            individual=False,
            n_vars=N_VARS,
            nf=NF,
            target_window=TARGET_WINDOW,
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    new_head = FlattenHead(
        n_vars=N_VARS, nf=NF, target_window=TARGET_WINDOW, individual=False
    )
    # copy weights so both produce identical output
    new_head.load_state_dict(old_head.state_dict())

    x = _make_input()
    assert torch.allclose(old_head(x), new_head(x))
