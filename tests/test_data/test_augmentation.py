"""Tests for time series augmentation transforms."""

__author__ = ["echo-xiao"]

import torch

from pytorch_forecasting.data.augmentation import BaseAugmentation


def _window(ctx=6, pred=3, n_cont=2, n_cat=1, n_targets=1):
    """A window whose values identify their own position, for alignment checks.

    ``cont[t, c] == t * 10 + c``, ``cat[t, 0] == t`` and ``target[t, 0] == t``,
    so any reordering or resampling along time is visible in the values.
    """
    total = ctx + pred
    steps = torch.arange(total, dtype=torch.float32).unsqueeze(-1)
    cont = steps * 10 + torch.arange(n_cont, dtype=torch.float32)
    cat = torch.arange(total).unsqueeze(-1).repeat(1, n_cat)
    target = steps.repeat(1, n_targets)
    return {
        "cont": cont,
        "cat": cat,
        "target": target,
        "ctx_len": ctx,
        "pred_len": pred,
    }


def test_base_augmentation_is_registered_as_an_object():
    """The base class declares the tags the data module dispatches on."""
    tags = BaseAugmentation.get_class_tags()
    assert tags["object_type"] == "augmentation"
    assert tags["requires_neighbours"] is False
    assert tags["touches_time_axis"] is False


def test_resample_positions_interpolates_cont_and_target():
    """Half-integer positions give the midpoint of the two neighbours."""
    window = _window(ctx=4, pred=2, n_cont=1)
    src_pos = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    out = BaseAugmentation.resample_positions(window, src_pos)

    assert torch.allclose(
        out["cont"][:, 0], torch.tensor([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    )
    assert torch.allclose(
        out["target"][:, 0], torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    )


def test_resample_positions_takes_nearest_index_for_cat():
    """Categorical codes are not interpolable, so they snap to the nearest step."""
    window = _window(ctx=4, pred=2, n_cat=1)
    src_pos = torch.tensor([0.0, 0.4, 0.6, 1.0, 2.2, 2.8])

    out = BaseAugmentation.resample_positions(window, src_pos)

    assert torch.equal(out["cat"][:, 0], torch.tensor([0, 0, 1, 1, 2, 3]))


def test_resample_positions_clamps_out_of_range():
    """Positions outside the window clamp instead of wrapping or erroring."""
    window = _window(ctx=4, pred=2, n_cont=1)
    src_pos = torch.tensor([-2.0, 0.0, 1.0, 2.0, 3.0, 99.0])

    out = BaseAugmentation.resample_positions(window, src_pos)

    assert out["cont"][0, 0] == 0.0
    assert out["cont"][-1, 0] == 50.0


def test_resample_positions_does_not_mutate_the_input():
    """The transform contract forbids in-place writes."""
    window = _window()
    before = {k: window[k].clone() for k in ("cont", "cat", "target")}

    BaseAugmentation.resample_positions(window, torch.zeros(9))

    for key, value in before.items():
        assert torch.equal(window[key], value), key


def test_resample_positions_keeps_the_split_lengths():
    """``ctx_len`` and ``pred_len`` are carried through untouched."""
    window = _window(ctx=6, pred=3)

    out = BaseAugmentation.resample_positions(
        window, torch.arange(9, dtype=torch.float32)
    )

    assert out["ctx_len"] == 6
    assert out["pred_len"] == 3
