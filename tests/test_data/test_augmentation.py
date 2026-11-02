"""Tests for time series augmentation transforms."""

__author__ = ["echo-xiao"]

import pandas as pd
import pytest
import torch

from pytorch_forecasting.data.augmentation import BaseAugmentation
from pytorch_forecasting.data.data_module._tslib_data_module import TslibDataModule
from pytorch_forecasting.data.timeseries._timeseries_v2 import TimeSeries


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


class _AddOne(BaseAugmentation):
    """Adds 1.0 to every continuous value. Deterministic, for wiring tests."""

    def transform(self, window, rng, neighbours=None):
        out = dict(window)
        out["cont"] = window["cont"] + 1.0
        return out


class _TimesTwo(BaseAugmentation):
    """Doubles every continuous value. Deterministic, for ordering tests."""

    def transform(self, window, rng, neighbours=None):
        out = dict(window)
        out["cont"] = window["cont"] * 2.0
        return out


def _make_dm(n_series=6, n_steps=40, **kwargs):
    """A minimal TslibDataModule over synthetic data."""
    frames = []
    for series in range(n_series):
        frames.append(
            pd.DataFrame(
                {
                    "series": series,
                    "time_idx": range(n_steps),
                    "x": range(n_steps),
                    "y": [float(t) for t in range(n_steps)],
                }
            )
        )
    data = pd.concat(frames, ignore_index=True)

    ts = TimeSeries(
        data=data,
        time="time_idx",
        target="y",
        group=["series"],
        num=["x"],
    )
    defaults = dict(
        time_series_dataset=ts,
        context_length=8,
        prediction_length=4,
        batch_size=4,
    )
    defaults.update(kwargs)
    return TslibDataModule(**defaults)


def test_augmentation_defaults_to_off():
    """Not passing the parameter leaves every split unaugmented."""
    dm = _make_dm()
    dm.setup(stage="fit")
    dm.setup(stage="test")

    assert dm.train_dataset.augmentation is None
    assert dm.val_dataset.augmentation is None
    assert dm.test_dataset.augmentation is None


def test_augmentation_reaches_the_train_split_only():
    """The transform is handed to the train dataset and nothing else."""
    dm = _make_dm(augmentation=_AddOne())
    dm.setup(stage="fit")
    dm.setup(stage="test")

    assert dm.train_dataset.augmentation is not None
    assert dm.val_dataset.augmentation is None
    assert dm.test_dataset.augmentation is None


def test_augmentation_changes_the_training_sample():
    """``_AddOne`` shifts history_cont by exactly 1.0 on the train split."""
    plain = _make_dm()
    plain.setup(stage="fit")
    augmented = _make_dm(augmentation=_AddOne())
    augmented.setup(stage="fit")

    x_plain, _ = plain.train_dataset[0]
    x_aug, _ = augmented.train_dataset[0]

    assert torch.allclose(x_aug["history_cont"], x_plain["history_cont"] + 1.0)


def test_a_single_transform_is_normalised_to_a_list():
    """Passing one transform and passing a one-element list are the same."""
    single = _make_dm(augmentation=_AddOne())
    listed = _make_dm(augmentation=[_AddOne()])

    assert len(single._augmentation) == 1
    assert len(listed._augmentation) == 1


def test_a_list_of_transforms_is_applied_in_order():
    """``[_AddOne, _TimesTwo]`` gives ``(x + 1) * 2``, not ``x * 2 + 1``."""
    plain = _make_dm()
    plain.setup(stage="fit")
    dm = _make_dm(augmentation=[_AddOne(), _TimesTwo()])
    dm.setup(stage="fit")

    x_plain, _ = plain.train_dataset[0]
    x_aug, _ = dm.train_dataset[0]

    assert torch.allclose(x_aug["history_cont"], (x_plain["history_cont"] + 1.0) * 2.0)


def test_the_same_index_is_resampled_on_every_access():
    """Augmentation redraws per access, so two reads of index 0 differ."""

    class _Noise(BaseAugmentation):
        def transform(self, window, rng, neighbours=None):
            out = dict(window)
            out["cont"] = window["cont"] + torch.randn(
                window["cont"].shape, generator=rng
            )
            return out

    dm = _make_dm(augmentation=_Noise())
    dm.setup(stage="fit")

    first, _ = dm.train_dataset[0]
    second, _ = dm.train_dataset[0]

    assert not torch.equal(first["history_cont"], second["history_cont"])
