"""Tests for time series augmentation transforms."""

__author__ = ["echo-xiao"]

import pandas as pd
import pytest
import torch

from pytorch_forecasting.data.augmentation import (
    BaseAugmentation,
    Jitter,
    MagnitudeWarp,
    NeighbourIndex,
    Permutation,
    Scaling,
    SignFlip,
    TimeWarp,
    WindowSlice,
    WindowWarp,
)
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


AMPLITUDE = [Jitter(), Scaling(), MagnitudeWarp(), SignFlip()]


@pytest.mark.parametrize("transform", AMPLITUDE, ids=lambda t: type(t).__name__)
def test_amplitude_transform_changes_the_continuous_block(transform):
    window = _window()
    rng = torch.Generator().manual_seed(0)

    out = transform.transform(window, rng)

    assert not torch.equal(out["cont"], window["cont"])


@pytest.mark.parametrize("transform", AMPLITUDE, ids=lambda t: type(t).__name__)
def test_amplitude_transform_keeps_every_shape(transform):
    window = _window()
    rng = torch.Generator().manual_seed(0)

    out = transform.transform(window, rng)

    for key in ("cont", "cat", "target"):
        assert out[key].shape == window[key].shape, key


@pytest.mark.parametrize("transform", AMPLITUDE, ids=lambda t: type(t).__name__)
def test_amplitude_transform_leaves_the_categorical_block_alone(transform):
    """Perturbing an integer code would produce a code that means nothing."""
    window = _window()
    rng = torch.Generator().manual_seed(0)

    out = transform.transform(window, rng)

    assert torch.equal(out["cat"], window["cat"])


@pytest.mark.parametrize("transform", AMPLITUDE, ids=lambda t: type(t).__name__)
def test_amplitude_transform_does_not_mutate_the_input(transform):
    window = _window()
    before = {k: window[k].clone() for k in ("cont", "cat", "target")}
    rng = torch.Generator().manual_seed(0)

    transform.transform(window, rng)

    for key, value in before.items():
        assert torch.equal(window[key], value), key


@pytest.mark.parametrize("transform", AMPLITUDE, ids=lambda t: type(t).__name__)
def test_amplitude_transform_is_reproducible_for_one_seed(transform):
    window = _window()

    first = transform.transform(window, torch.Generator().manual_seed(7))
    second = transform.transform(window, torch.Generator().manual_seed(7))

    assert torch.equal(first["cont"], second["cont"])


@pytest.mark.parametrize("transform", AMPLITUDE, ids=lambda t: type(t).__name__)
def test_amplitude_transform_differs_across_seeds(transform):
    window = _window()

    first = transform.transform(window, torch.Generator().manual_seed(1))
    second = transform.transform(window, torch.Generator().manual_seed(2))

    assert not torch.equal(first["cont"], second["cont"])


def test_scaling_multiplies_each_channel_by_one_factor():
    """A per-channel factor means the ratio is constant down each column."""
    window = _window(ctx=6, pred=2, n_cont=2)
    rng = torch.Generator().manual_seed(0)

    out = Scaling().transform(window, rng)

    ratio = out["cont"] / window["cont"].clamp(min=1e-6)
    for channel in range(window["cont"].shape[1]):
        column = ratio[1:, channel]
        assert torch.allclose(column, column[0].expand_as(column), atol=1e-4)


def test_sign_flip_does_not_permute_feature_columns():
    """Unlike thuml's rotation, columns keep their identity.

    ``cont_known_mask`` in the data module selects known features by column
    position, so permuting columns would feed "known future" values from the
    wrong feature, silently.
    """
    window = _window(ctx=6, pred=2, n_cont=2)
    rng = torch.Generator().manual_seed(0)

    out = SignFlip().transform(window, rng)

    assert torch.equal(out["cont"].abs(), window["cont"].abs())


TIME_AXIS = [Permutation(), TimeWarp(), WindowSlice(), WindowWarp()]


@pytest.mark.parametrize("transform", TIME_AXIS, ids=lambda t: type(t).__name__)
def test_time_axis_transform_declares_its_tag(transform):
    assert transform.get_class_tags()["touches_time_axis"] is True


@pytest.mark.parametrize("transform", TIME_AXIS, ids=lambda t: type(t).__name__)
def test_time_axis_transform_changes_the_continuous_block(transform):
    """A transform that silently no-ops would pass every other test here.

    ``WindowSlice`` and ``WindowWarp`` return the window untouched when it is
    too short for their span, so this uses a window long enough for all four.
    """
    window = _window(ctx=16, pred=8)
    rng = torch.Generator().manual_seed(0)

    out = transform.transform(window, rng)

    assert not torch.equal(out["cont"], window["cont"])


@pytest.mark.parametrize("transform", TIME_AXIS, ids=lambda t: type(t).__name__)
def test_time_axis_transform_keeps_every_shape(transform):
    """Even WindowSlice, which samples a shorter span, interpolates back."""
    window = _window(ctx=16, pred=8)
    rng = torch.Generator().manual_seed(0)

    out = transform.transform(window, rng)

    for key in ("cont", "cat", "target"):
        assert out[key].shape == window[key].shape, key


@pytest.mark.parametrize("transform", TIME_AXIS, ids=lambda t: type(t).__name__)
def test_time_axis_transform_keeps_cat_aligned_with_cont(transform):
    """The whole point of resample_positions.

    In the fixture ``cont[t, 0] == t * 10`` and ``cat[t, 0] == t``, so after any
    resampling the two must still agree about which timestep each row is. A
    transform that warped only the continuous block would pass the shape test
    and fail this one.
    """
    window = _window(ctx=16, pred=8, n_cont=1, n_cat=1)
    rng = torch.Generator().manual_seed(0)

    out = transform.transform(window, rng)

    implied_step = out["cont"][:, 0] / 10.0
    assert torch.allclose(implied_step, out["cat"][:, 0].to(torch.float32), atol=0.5)


@pytest.mark.parametrize("transform", TIME_AXIS, ids=lambda t: type(t).__name__)
def test_time_axis_transform_does_not_mutate_the_input(transform):
    window = _window(ctx=16, pred=8)
    before = {k: window[k].clone() for k in ("cont", "cat", "target")}
    rng = torch.Generator().manual_seed(0)

    transform.transform(window, rng)

    for key, value in before.items():
        assert torch.equal(window[key], value), key


@pytest.mark.parametrize("transform", TIME_AXIS, ids=lambda t: type(t).__name__)
def test_time_axis_transform_is_reproducible_for_one_seed(transform):
    window = _window(ctx=16, pred=8)

    first = transform.transform(window, torch.Generator().manual_seed(7))
    second = transform.transform(window, torch.Generator().manual_seed(7))

    assert torch.equal(first["cont"], second["cont"])


def test_permutation_keeps_the_multiset_of_steps():
    """Segments are reordered, so every original step appears exactly once."""
    window = _window(ctx=16, pred=8, n_cat=1)
    rng = torch.Generator().manual_seed(3)

    out = Permutation().transform(window, rng)

    assert sorted(out["cat"][:, 0].tolist()) == sorted(window["cat"][:, 0].tolist())


def test_permutation_actually_reorders():
    window = _window(ctx=16, pred=8, n_cat=1)
    rng = torch.Generator().manual_seed(3)

    out = Permutation().transform(window, rng)

    assert not torch.equal(out["cat"], window["cat"])


def test_time_warp_shares_one_warp_across_channels():
    """All channels must map to the same timestep, or cross-feature alignment
    at a given time would be destroyed.
    """
    window = _window(ctx=16, pred=8, n_cont=3)
    rng = torch.Generator().manual_seed(5)

    out = TimeWarp().transform(window, rng)

    # cont[t, c] == t * 10 + c, so subtracting the channel offset must leave
    # the same implied timestep in every channel.
    offsets = torch.arange(3, dtype=torch.float32)
    implied = (out["cont"] - offsets) / 10.0
    for channel in range(1, 3):
        assert torch.allclose(implied[:, 0], implied[:, channel], atol=1e-3)


def test_window_slice_samples_a_sub_span():
    """The output spans less than the full window, stretched back to length."""
    window = _window(ctx=16, pred=8, n_cat=1)
    rng = torch.Generator().manual_seed(0)

    out = WindowSlice(reduce_ratio=0.5).transform(window, rng)

    covered = out["cat"][:, 0].max() - out["cat"][:, 0].min()
    assert covered < 23


def test_the_transforms_are_importable_from_the_data_package():
    """Users write TslibDataModule(augmentation=Jitter()), so the short path
    has to work."""
    from pytorch_forecasting.data import (
        BaseAugmentation as Base,
        Jitter as J,
        MagnitudeWarp as M,
        Permutation as P,
        Scaling as S,
        SignFlip as F,
        TimeWarp as T,
        WindowSlice as WS,
        WindowWarp as WW,
    )

    for cls in (J, S, M, F, P, T, WS, WW):
        assert issubclass(cls, Base)


class _NeedsNeighbours(BaseAugmentation):
    """Copies the neighbour's continuous block in. For plumbing tests."""

    _tags = {"requires_neighbours": True}

    def transform(self, window, rng, neighbours=None):
        if neighbours is None:
            raise AssertionError("the data module must pass a NeighbourIndex")
        out = dict(window)
        out["_saw_neighbours"] = True
        return out


def _make_labelled_dm(n_per_label=3, n_labels=2, **kwargs):
    """A data module whose static categorical ``label`` is the class label."""
    frames = []
    series = 0
    for label in range(n_labels):
        for _ in range(n_per_label):
            frames.append(
                pd.DataFrame(
                    {
                        "series": series,
                        "time_idx": range(40),
                        "x": range(40),
                        "y": [float(t) for t in range(40)],
                        "label": label,
                    }
                )
            )
            series += 1
    data = pd.concat(frames, ignore_index=True)

    ts = TimeSeries(
        data=data,
        time="time_idx",
        target="y",
        group=["series"],
        num=["x"],
        cat=["label"],
        static=["label"],
    )
    defaults = dict(
        time_series_dataset=ts,
        context_length=8,
        prediction_length=4,
        batch_size=4,
    )
    defaults.update(kwargs)
    return TslibDataModule(**defaults)


def test_missing_augmentation_label_is_an_error():
    """A guided transform without a label column cannot work; say so early."""
    dm = _make_labelled_dm(augmentation=_NeedsNeighbours())

    with pytest.raises(ValueError, match="augmentation_label"):
        dm.setup(stage="fit")


def test_unknown_augmentation_label_names_the_column():
    dm = _make_labelled_dm(
        augmentation=_NeedsNeighbours(), augmentation_label="not_a_column"
    )

    with pytest.raises(ValueError, match="not_a_column"):
        dm.setup(stage="fit")


def test_neighbour_index_is_built_when_a_transform_needs_it():
    # n_per_label=5: with the default 0.7/0.15/0.15 split and 2 labels, the
    # worst-case random draw still leaves >= 2 series of each label in
    # training (train_size=7, so at most 5 of one label plus 2 of the
    # other), so this holds for every permutation instead of by luck.
    dm = _make_labelled_dm(
        n_per_label=5, augmentation=_NeedsNeighbours(), augmentation_label="label"
    )
    dm.setup(stage="fit")

    assert dm.train_dataset.neighbours is not None


def test_no_neighbour_index_when_nothing_needs_it():
    """val/test and the per-sample-only case must pay nothing."""
    dm = _make_labelled_dm(augmentation=Jitter(), augmentation_label="label")
    dm.setup(stage="fit")

    assert dm.train_dataset.neighbours is None
    assert dm.val_dataset.neighbours is None


def test_neighbour_comes_from_a_different_series():
    """thuml's "sample" maps to our series, not our window, so other windows
    of the same series are not neighbours."""
    # n_per_label=5: see test_neighbour_index_is_built_when_a_transform_needs_it
    # for why this makes the "at least 2 series per label in training"
    # precondition hold for every permutation of the random split.
    dm = _make_labelled_dm(
        n_per_label=5, augmentation=_NeedsNeighbours(), augmentation_label="label"
    )
    dm.setup(stage="fit")
    index = dm.train_dataset.neighbours
    rng = torch.Generator().manual_seed(0)

    for window_idx in range(min(20, len(dm.train_dataset))):
        own_series = index.series_of_window[window_idx]
        neighbour_idx = index.sample_index(window_idx, rng)
        assert index.series_of_window[neighbour_idx] != own_series
        assert index.labels[neighbour_idx] == index.labels[window_idx]


def test_other_label_neighbour_has_a_different_label():
    """DiscriminativeGuidedWarp needs negative examples, not just positive."""
    # n_per_label=5: see test_neighbour_index_is_built_when_a_transform_needs_it
    # for why this makes the "at least 2 series per label in training"
    # precondition hold for every permutation of the random split.
    dm = _make_labelled_dm(
        n_per_label=5, augmentation=_NeedsNeighbours(), augmentation_label="label"
    )
    dm.setup(stage="fit")
    index = dm.train_dataset.neighbours
    rng = torch.Generator().manual_seed(0)

    for window_idx in range(min(20, len(dm.train_dataset))):
        other = index.sample_other_label_index(window_idx, rng)
        assert index.labels[other] != index.labels[window_idx]


def test_a_label_covering_one_series_raises_at_build_time():
    """thuml degrades silently here; we refuse to start instead."""
    dm = _make_labelled_dm(
        n_per_label=1,
        n_labels=2,
        augmentation=_NeedsNeighbours(),
        augmentation_label="label",
        train_val_test_split=(1.0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="only one series"):
        dm.setup(stage="fit")


def test_as_array_concatenates_cont_and_target_without_cat():
    """Categorical codes carry no distance, so they stay out of the DTW input."""
    window = _window(ctx=6, pred=2, n_cont=2, n_cat=1, n_targets=1)

    array = NeighbourIndex.as_array(window)

    assert array.shape == (8, 3)
