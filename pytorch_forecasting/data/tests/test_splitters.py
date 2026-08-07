"""Tests for train_test_split strategies."""

import numpy as np
import pytest
import torch

from pytorch_forecasting.data.splitters import (
    random_series_split,
    temporal_window_split,
    group_time_split
)


class TestRandomSeriesSplit:
    """Tests for random_series_split."""

    def test_basic_split_proportions(self):
        """All indices should be assigned, no overlap."""
        train, val, test = random_series_split(100, (0.7, 0.15, 0.15))
        all_indices = torch.cat([train, val, test])
        assert len(all_indices.unique()) == 100

    def test_single_series(self):
        """Edge case: only 1 series."""
        train, val, test = random_series_split(1, (0.7, 0.15, 0.15))
        total = len(train) + len(val) + len(test)
        assert total == 1

    def test_zero_test_split(self):
        """Test with no test set."""
        train, val, test = random_series_split(10, (0.8, 0.2, 0.0))
        assert len(test) == 0


class TestTemporalWindowSplit:
    """Tests for timestamp-based temporal_window_split."""

    @pytest.fixture
    def overlapping_series_setup(self):
        enc_len, pred_len = 4, 2
        windows = []

        for s in range(0, 24 - enc_len - pred_len + 1):
            windows.append((0, s, enc_len, pred_len))

        for s in range(0, 24 - enc_len - pred_len + 1):
            windows.append((1, s, enc_len, pred_len))

        series_timestamps = {
            0: np.arange(0, 24),
            1: np.arange(6, 30),
        }

        return windows, series_timestamps

    def test_no_leakage(self, overlapping_series_setup):
        """No leakage within each individual series."""
        windows, series_timestamps = overlapping_series_setup

        train_w, val_w, test_w = temporal_window_split(
            windows, (0.7, 0.15, 0.15), series_timestamps
        )

        def get_end_time(w):
            s_idx, start, enc, pred = w
            end = min(start + enc + pred - 1, len(series_timestamps[s_idx]) - 1)
            return float(series_timestamps[s_idx][end])

        all_series = {w[0] for w in train_w + val_w + test_w}
        for s_idx in all_series:
            s_train = [w for w in train_w if w[0] == s_idx]
            s_val = [w for w in val_w if w[0] == s_idx]
            s_test = [w for w in test_w if w[0] == s_idx]

            if s_train and s_val:
                max_train = max(get_end_time(w) for w in s_train)
                min_val = min(get_end_time(w) for w in s_val)
                assert max_train <= min_val, (
                    f"Series {s_idx}: LEAKAGE train->val: "
                    f"max_train={max_train} > min_val={min_val}"
                )

            if s_val and s_test:
                max_val = max(get_end_time(w) for w in s_val)
                min_test = min(get_end_time(w) for w in s_test)
                assert max_val <= min_test, (
                    f"Series {s_idx}: LEAKAGE val->test: "
                    f"max_val={max_val} > min_test={min_test}"
                )

    def test_single_series(self):
        """A single series should split correctly by time."""
        enc_len, pred_len = 3, 2
        timestamps = np.arange(0, 20)
        windows = [
            (0, s, enc_len, pred_len) for s in range(20 - enc_len - pred_len + 1)
        ]
        series_timestamps = {0: timestamps}

        train_w, val_w, test_w = temporal_window_split(
            windows, (0.6, 0.2, 0.2), series_timestamps
        )

        total = len(train_w) + len(val_w) + len(test_w)
        assert total == len(windows)
        assert len(train_w) > 0
        assert len(val_w) > 0
        assert len(test_w) > 0

    def test_identical_timestamps_fallback(self):
        """When all timestamps are identical, should fall back gracefully."""
        timestamps = np.array([5, 5, 5, 5, 5])
        windows = [(0, 0, 2, 1), (0, 1, 2, 1), (0, 2, 2, 1)]
        series_timestamps = {0: timestamps}

        train_w, val_w, test_w = temporal_window_split(
            windows, (0.7, 0.15, 0.15), series_timestamps
        )

        total = len(train_w) + len(val_w) + len(test_w)
        assert total == len(windows), "All windows must be assigned"

    def test_empty_windows(self):
        """Empty window list should return three empty lists."""
        train_w, val_w, test_w = temporal_window_split([], (0.7, 0.15, 0.15), {})
        assert train_w == []
        assert val_w == []
        assert test_w == []


class TestTemporalWindowSplitAbsoluteMode:
    """Tests for absolute cutoff mode."""

    def test_absolute_cutoffs_basic(self):
        """Windows should be split at exact timestamp boundaries."""
        enc_len, pred_len = 3, 2
        timestamps = np.arange(0, 20)
        windows = [
            (0, s, enc_len, pred_len) for s in range(20 - enc_len - pred_len + 1)
        ]
        series_timestamps = {0: timestamps}

        cutoffs = {"end_train": 10.0, "start_test": 15.0}
        train_w, val_w, test_w = temporal_window_split(
            windows,
            (0.7, 0.15, 0.15),
            series_timestamps,
            temporal_cutoffs=cutoffs,
        )
        for w in train_w:
            end_idx = min(w[1] + w[2] + w[3] - 1, len(timestamps) - 1)
            assert timestamps[end_idx] <= 10.0

        for w in test_w:
            end_idx = min(w[1] + w[2] + w[3] - 1, len(timestamps) - 1)
            assert timestamps[end_idx] >= 15.0

        total = len(train_w) + len(val_w) + len(test_w)
        assert total == len(windows)

    def test_absolute_no_gap(self):
        """When start_test == end_train, val should be empty."""
        timestamps = np.arange(0, 10)
        windows = [(0, s, 2, 1) for s in range(8)]
        series_timestamps = {0: timestamps}

        cutoffs = {"end_train": 5.0, "start_test": 5.0}
        train_w, val_w, test_w = temporal_window_split(
            windows,
            (0.7, 0.15, 0.15),
            series_timestamps,
            temporal_cutoffs=cutoffs,
        )
        assert len(val_w) == 0

    def test_absolute_invalid_cutoffs_raises(self):
        """start_test < end_train should raise ValueError."""
        windows = [(0, 0, 2, 1)]
        series_timestamps = {0: np.arange(5)}

        with pytest.raises(ValueError, match="start_test"):
            temporal_window_split(
                windows,
                (0.7, 0.15, 0.15),
                series_timestamps,
                temporal_cutoffs={"end_train": 10.0, "start_test": 5.0},
            )


class TestTemporalWindowSplitPercentageMode:
    """Tests for per-series percentage mode."""

    def test_per_series_cutoff_ordering(self):
        """Within each series, train windows must end before val, val before test.

        Per-series percentage split no longer guarantees a global cross-series
        ordering (a train window in Series B may end later than a val window in
        Series A). The invariant is per-series monotonicity only.
        """
        enc_len, pred_len = 3, 2
        ts_a = np.arange(0, 20)
        ts_b = np.arange(10, 30)

        windows_a = [
            (0, s, enc_len, pred_len) for s in range(20 - enc_len - pred_len + 1)
        ]
        windows_b = [
            (1, s, enc_len, pred_len) for s in range(20 - enc_len - pred_len + 1)
        ]
        windows = windows_a + windows_b
        series_timestamps = {0: ts_a, 1: ts_b}

        train_w, val_w, test_w = temporal_window_split(
            windows, (0.7, 0.15, 0.15), series_timestamps
        )

        def get_end(w):
            s_idx, start, enc, pred = w
            end = min(start + enc + pred - 1, len(series_timestamps[s_idx]) - 1)
            return series_timestamps[s_idx][end]

        for s_idx in (0, 1):
            s_train = [w for w in train_w if w[0] == s_idx]
            s_val = [w for w in val_w if w[0] == s_idx]
            s_test = [w for w in test_w if w[0] == s_idx]

            if s_train and s_val:
                assert max(get_end(w) for w in s_train) <= min(
                    get_end(w) for w in s_val
                ), f"Series {s_idx}: train leaks into val"

            if s_val and s_test:
                assert max(get_end(w) for w in s_val) <= min(
                    get_end(w) for w in s_test
                ), f"Series {s_idx}: val leaks into test"

    def test_per_series_proportional_split(self):
        """Each series must contribute windows to all three folds.

        This is the maintainer's core requirement: a series with timestamps
        [1..10] and another with [11..20] should both be split 80/10/10
        independently, not skewed by a global cutoff.
        """
        enc_len, pred_len = 1, 1
        ts_g1 = np.arange(1, 11)  # [1..10]
        ts_g2 = np.arange(11, 21)  # [11..20]

        windows_g1 = [(0, s, enc_len, pred_len) for s in range(len(ts_g1) - 1)]
        windows_g2 = [(1, s, enc_len, pred_len) for s in range(len(ts_g2) - 1)]
        windows = windows_g1 + windows_g2
        series_timestamps = {0: ts_g1, 1: ts_g2}

        train_w, val_w, test_w = temporal_window_split(
            windows, (0.8, 0.1, 0.1), series_timestamps
        )

        # Both series must appear in train
        train_series = {w[0] for w in train_w}
        assert 0 in train_series, "G1 must have training windows"
        assert 1 in train_series, "G2 must have training windows"

        # Both series must have some windows outside train (val or test)
        g1_non_train = [w for w in val_w + test_w if w[0] == 0]
        g2_non_train = [w for w in val_w + test_w if w[0] == 1]
        assert len(g1_non_train) > 0, "G1 must have val/test windows"
        assert len(g2_non_train) > 0, "G2 must have val/test windows"

        # All windows must be assigned
        total = len(train_w) + len(val_w) + len(test_w)
        assert total == len(windows)

    def test_datetime_timestamps(self):
        """Percentage split should work with datetime64 timestamps."""
        timestamps = np.arange("2023-01", "2023-07", dtype="datetime64[M]")
        enc_len, pred_len = 2, 1
        windows = [
            (0, s, enc_len, pred_len)
            for s in range(len(timestamps) - enc_len - pred_len + 1)
        ]
        series_timestamps = {0: timestamps}

        train_w, val_w, test_w = temporal_window_split(
            windows, (0.6, 0.2, 0.2), series_timestamps
        )

        total = len(train_w) + len(val_w) + len(test_w)
        assert total == len(windows)
        assert len(train_w) > 0

    def test_warning_on_conflicting_params(self):
        """A warning should fire when both cutoffs and custom split are given."""
        timestamps = np.arange(0, 10)
        windows = [(0, 0, 2, 1)]
        series_timestamps = {0: timestamps}

        with pytest.warns(UserWarning, match="temporal_cutoffs"):
            temporal_window_split(
                windows,
                (0.5, 0.25, 0.25),
                series_timestamps,
                temporal_cutoffs={"end_train": 5.0},
            )

class TestGroupTimeSplit:
    """Tests for the two-phase group-time split."""

    def test_train_and_test_groups_are_disjoint(self):
        """Groups in train must never appear in test (the core invariant)."""
        ts = {i: np.arange(0, 20) for i in range(10)}
        enc, pred = 2, 1
        windows = [
            (s_idx, s, enc, pred)
            for s_idx in range(10)
            for s in range(20 - enc - pred + 1)
        ]

        train_w, val_w, test_w = group_time_split(
            windows, ts, (0.7, 0.15, 0.15), group_split=(0.6, 0.2, 0.2)
        )

        train_groups = {w[0] for w in train_w}
        test_only_groups = {w[0] for w in test_w} - train_groups

        assert len(test_only_groups) > 0, "There must be groups entirely held out for test"
        assert train_groups.isdisjoint(test_only_groups), (
            "Train and test groups from phase 1 must have no shared groups"
        )

    def test_all_windows_assigned(self):
        """Every window must end up in exactly one fold."""
        ts = {i: np.arange(0, 20) for i in range(5)}
        enc, pred = 2, 1
        windows = [
            (s_idx, s, enc, pred)
            for s_idx in range(5)
            for s in range(20 - enc - pred + 1)
        ]

        train_w, val_w, test_w = group_time_split(
            windows, ts, (0.7, 0.15, 0.15)
        )
        assert len(train_w) + len(val_w) + len(test_w) == len(windows)

    def test_empty_windows(self):
        """Empty input returns three empty lists."""
        train_w, val_w, test_w = group_time_split([], {}, (0.7, 0.15, 0.15))
        assert train_w == val_w == test_w == []

    def test_train_group_windows_are_temporally_ordered(self):
        """Within train groups, the temporal split must not leak."""
        ts = {i: np.arange(0, 30) for i in range(6)}
        enc, pred = 3, 2
        windows = [
            (s_idx, s, enc, pred)
            for s_idx in range(6)
            for s in range(30 - enc - pred + 1)
        ]

        train_w, val_w, test_w = group_time_split(
            windows, ts, (0.7, 0.15, 0.15), group_split=(0.6, 0.2, 0.2)
        )

        def get_end(w):
            s_idx, start, enc_, pred_ = w
            end = min(start + enc_ + pred_ - 1, len(ts[s_idx]) - 1)
            return ts[s_idx][end]

        train_groups = {w[0] for w in train_w}
        for s_idx in train_groups:
            s_train = [w for w in train_w if w[0] == s_idx]
            s_val = [w for w in val_w if w[0] == s_idx]
            s_test = [w for w in test_w if w[0] == s_idx]

            if s_train and s_val:
                assert max(get_end(w) for w in s_train) <= min(
                    get_end(w) for w in s_val
                ), f"Series {s_idx}: temporal leakage train→val"

            if s_val and s_test:
                assert max(get_end(w) for w in s_val) <= min(
                    get_end(w) for w in s_test
                ), f"Series {s_idx}: temporal leakage val→test"
