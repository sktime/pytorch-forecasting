"""Tests for train_test_split strategies."""

import numpy as np
import pytest
import torch

from pytorch_forecasting.data.split.splitters import (
    GroupTimeSplitter,
    RandomSplitter,
    TemporalSplitter,
)


class TestRandomSeriesSplit:
    """Tests for RandomSplitter."""

    def test_basic_split_proportions(self):
        """All indices should be assigned, no overlap."""
        splitter = RandomSplitter((0.7, 0.15, 0.15))
        train, val, test = splitter.split_series(100, None)
        all_indices = torch.cat([train, val, test])
        assert len(all_indices.unique()) == 100

    def test_single_series(self):
        """Edge case: only 1 series."""
        splitter = RandomSplitter((0.7, 0.15, 0.15))
        train, val, test = splitter.split_series(1, None)
        total = len(train) + len(val) + len(test)
        assert total == 1

    def test_zero_test_split(self):
        """Test with no test set."""
        splitter = RandomSplitter((0.8, 0.2, 0.0))
        train, val, test = splitter.split_series(10, None)
        assert len(test) == 0


class TestTemporalWindowSplit:
    """Tests for TemporalSplitter."""

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
        dataset = {k: {"t": v} for k, v in series_timestamps.items()}

        splitter = TemporalSplitter((0.7, 0.15, 0.15))
        train_w, val_w, test_w = splitter.split_windows(windows, dataset)

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
                assert max_train <= min_val

            if s_val and s_test:
                max_val = max(get_end_time(w) for w in s_val)
                min_test = min(get_end_time(w) for w in s_test)
                assert max_val <= min_test

    def test_single_series(self):
        """A single series should split correctly by time."""
        enc_len, pred_len = 3, 2
        timestamps = np.arange(0, 20)
        windows = [
            (0, s, enc_len, pred_len) for s in range(20 - enc_len - pred_len + 1)
        ]
        dataset = {0: {"t": timestamps}}

        splitter = TemporalSplitter((0.6, 0.2, 0.2))
        train_w, val_w, test_w = splitter.split_windows(windows, dataset)

        total = len(train_w) + len(val_w) + len(test_w)
        assert total == len(windows)
        assert len(train_w) > 0

    def test_identical_timestamps_fallback(self):
        """When all timestamps are identical, should fall back gracefully."""
        timestamps = np.array([5, 5, 5, 5, 5])
        windows = [(0, 0, 2, 1), (0, 1, 2, 1), (0, 2, 2, 1)]
        dataset = {0: {"t": timestamps}}

        splitter = TemporalSplitter((0.7, 0.15, 0.15))
        train_w, val_w, test_w = splitter.split_windows(windows, dataset)

        total = len(train_w) + len(val_w) + len(test_w)
        assert total == len(windows)

    def test_empty_windows(self):
        """Empty window list should return three empty lists."""
        splitter = TemporalSplitter((0.7, 0.15, 0.15))
        train_w, val_w, test_w = splitter.split_windows([], {})
        assert train_w == []


class TestTemporalWindowSplitAbsoluteMode:
    """Tests for absolute cutoff mode."""

    def test_absolute_cutoffs_basic(self):
        """Windows should be split at exact timestamp boundaries."""
        enc_len, pred_len = 3, 2
        timestamps = np.arange(0, 20)
        windows = [
            (0, s, enc_len, pred_len) for s in range(20 - enc_len - pred_len + 1)
        ]
        dataset = {0: {"t": timestamps}}

        cutoffs = {"end_train": 10.0, "start_test": 15.0}
        splitter = TemporalSplitter((0.7, 0.15, 0.15), temporal_cutoffs=cutoffs)
        train_w, val_w, test_w = splitter.split_windows(windows, dataset)

        for w in train_w:
            end_idx = min(w[1] + w[2] + w[3] - 1, len(timestamps) - 1)
            assert timestamps[end_idx] <= 10.0

        for w in test_w:
            end_idx = min(w[1] + w[2] + w[3] - 1, len(timestamps) - 1)
            assert timestamps[end_idx] >= 15.0

    def test_absolute_no_gap(self):
        """When start_test == end_train, val should be empty."""
        timestamps = np.arange(0, 10)
        windows = [(0, s, 2, 1) for s in range(8)]
        dataset = {0: {"t": timestamps}}

        cutoffs = {"end_train": 5.0, "start_test": 5.0}
        splitter = TemporalSplitter((0.7, 0.15, 0.15), temporal_cutoffs=cutoffs)
        train_w, val_w, test_w = splitter.split_windows(windows, dataset)

        assert len(val_w) == 0


class TestTemporalWindowSplitPercentageMode:
    """Tests for per-series percentage mode."""

    def test_per_series_cutoff_ordering(self):
        """Within each series, train windows must end before val, val before test."""
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
        dataset = {0: {"t": ts_a}, 1: {"t": ts_b}}

        splitter = TemporalSplitter((0.7, 0.15, 0.15))
        train_w, val_w, test_w = splitter.split_windows(windows, dataset)

        def get_end(w):
            s_idx, start, enc, pred = w
            end = min(start + enc + pred - 1, len(dataset[s_idx]["t"]) - 1)
            return dataset[s_idx]["t"][end]

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
        """Each series must contribute windows to all three folds."""
        enc_len, pred_len = 1, 1
        ts_g1 = np.arange(1, 11)
        ts_g2 = np.arange(11, 21)

        windows_g1 = [(0, s, enc_len, pred_len) for s in range(len(ts_g1) - 1)]
        windows_g2 = [(1, s, enc_len, pred_len) for s in range(len(ts_g2) - 1)]
        windows = windows_g1 + windows_g2
        dataset = {0: {"t": ts_g1}, 1: {"t": ts_g2}}

        splitter = TemporalSplitter((0.8, 0.1, 0.1))
        train_w, val_w, test_w = splitter.split_windows(windows, dataset)

        train_series = {w[0] for w in train_w}
        assert 0 in train_series
        assert 1 in train_series

    def test_datetime_timestamps(self):
        """Percentage split should work with datetime64 timestamps."""
        timestamps = np.arange("2023-01", "2023-07", dtype="datetime64[M]")
        enc_len, pred_len = 2, 1
        windows = [
            (0, s, enc_len, pred_len)
            for s in range(len(timestamps) - enc_len - pred_len + 1)
        ]
        dataset = {0: {"t": timestamps}}

        splitter = TemporalSplitter((0.6, 0.2, 0.2))
        train_w, val_w, test_w = splitter.split_windows(windows, dataset)

        total = len(train_w) + len(val_w) + len(test_w)
        assert total == len(windows)
        assert len(train_w) > 0


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
        dataset = {k: {"t": v} for k, v in ts.items()}

        splitter = GroupTimeSplitter((0.7, 0.15, 0.15), group_split=(0.6, 0.2, 0.2))
        train_w, val_w, test_w = splitter.split_windows(windows, dataset)

        train_groups = {w[0] for w in train_w}
        test_only_groups = {w[0] for w in test_w} - train_groups

        assert len(test_only_groups) > 0
        assert train_groups.isdisjoint(test_only_groups)

    def test_all_windows_assigned(self):
        """Every window must end up in exactly one fold."""
        ts = {i: np.arange(0, 20) for i in range(5)}
        enc, pred = 2, 1
        windows = [
            (s_idx, s, enc, pred)
            for s_idx in range(5)
            for s in range(20 - enc - pred + 1)
        ]
        dataset = {k: {"t": v} for k, v in ts.items()}

        splitter = GroupTimeSplitter((0.7, 0.15, 0.15))
        train_w, val_w, test_w = splitter.split_windows(windows, dataset)
        assert len(train_w) + len(val_w) + len(test_w) == len(windows)

    def test_empty_windows(self):
        """Empty input returns three empty lists."""
        splitter = GroupTimeSplitter((0.7, 0.15, 0.15))
        train_w, val_w, test_w = splitter.split_windows([], {})
        assert train_w == []
        assert val_w == []
        assert test_w == []

    def test_train_group_windows_are_temporally_ordered(self):
        """Within train groups, the temporal split must not leak."""
        ts = {i: np.arange(0, 30) for i in range(6)}
        enc, pred = 3, 2
        windows = [
            (s_idx, s, enc, pred)
            for s_idx in range(6)
            for s in range(30 - enc - pred + 1)
        ]
        dataset = {k: {"t": v} for k, v in ts.items()}

        splitter = GroupTimeSplitter((0.7, 0.15, 0.15), group_split=(0.6, 0.2, 0.2))
        train_w, val_w, test_w = splitter.split_windows(windows, dataset)

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
