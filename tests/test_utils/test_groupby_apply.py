import pytest
import torch

from pytorch_forecasting.utils import groupby_apply


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_groupby_apply_unsorted_keys(reduction):
    """Regression test for #1788: values must be grouped by key, not by position."""
    keys = torch.tensor([2, 0, 1, 0, 2, 1])
    values = torch.tensor([20.0, 0.0, 10.0, 2.0, 22.0, 12.0])

    reduced, hist = groupby_apply(
        keys, values, bins=4, reduction=reduction, return_histogram=True
    )

    if reduction == "mean":
        expected = torch.tensor([1.0, 11.0, 21.0, 0.0])
    else:
        expected = torch.tensor([2.0, 22.0, 42.0, 0.0])
    torch.testing.assert_close(reduced, expected)
    torch.testing.assert_close(hist, torch.tensor([2, 2, 2, 0]))


def test_groupby_apply_mismatched_shapes():
    """Keys and values of different shapes raise instead of being truncated."""
    with pytest.raises(ValueError, match="same shape"):
        groupby_apply(torch.tensor([1, 0]), torch.tensor([1.0, 2.0, 3.0]), bins=2)
