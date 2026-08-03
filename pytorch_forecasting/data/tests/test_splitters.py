"""Tests for train_test_split strategies."""

import numpy as np
import pytest
import torch

from pytorch_forecasting.data.splitters import (
    random_series_split,
    temporal_window_split,
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

    def test_all_windows_assigned_overlapping_series(self, overlapping_series_setup):
        """Verify all windows are assigned and both series appear in train."""
        windows, series_timestamps = overlapping_series_setup

        train_w, val_w, test_w = temporal_window_split(
            windows, (0.7, 0.15, 0.15), series_timestamps
        )

        total = len(train_w) + len(val_w) + len(test_w)
        assert total == len(windows)

        train_series = {w[0] for w in train_w}
        assert 0 in train_series
        assert 1 in train_series

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

    def test_non_overlapping_series(self):
        """When 2 series aren't overlapping, per-series split applies."""
        enc_len, pred_len = 2, 1
        ts_a = np.arange(0, 24)
        ts_b = np.arange(24, 48)

        windows_a = [
            (0, s, enc_len, pred_len) for s in range(24 - enc_len - pred_len + 1)
        ]
        windows_b = [
            (1, s, enc_len, pred_len) for s in range(24 - enc_len - pred_len + 1)
        ]
        windows = windows_a + windows_b
        series_timestamps = {0: ts_a, 1: ts_b}

        train_w, val_w, test_w = temporal_window_split(
            windows, (0.7, 0.15, 0.15), series_timestamps
        )

        total = len(train_w) + len(val_w) + len(test_w)
        assert total == len(windows)

        for s_idx in [0, 1]:
            s_train = [w for w in train_w if w[0] == s_idx]
            s_test = [w for w in test_w if w[0] == s_idx]
            assert len(s_train) > 0, f"Series {s_idx} should have train windows"
            assert len(s_test) > 0, f"Series {s_idx} should have test windows"


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

    def test_different_lifespans_proportional(self):
        """Each series should get ~70% train regardless of when it starts."""
        enc_len, pred_len = 2, 1

        # Series A: timestamps 0-23,  Series B: timestamps 100-123
        ts_a = np.arange(0, 24)
        ts_b = np.arange(100, 124)

        windows_a = [
            (0, s, enc_len, pred_len) for s in range(24 - enc_len - pred_len + 1)
        ]
        windows_b = [
            (1, s, enc_len, pred_len) for s in range(24 - enc_len - pred_len + 1)
        ]
        windows = windows_a + windows_b
        series_timestamps = {0: ts_a, 1: ts_b}

        train_w, val_w, test_w = temporal_window_split(
            windows,
            (0.7, 0.15, 0.15),
            series_timestamps,
        )

        train_series = {w[0] for w in train_w}
        assert 0 in train_series
        assert 1 in train_series
        test_series = {w[0] for w in test_w}
        assert 0 in test_series, "Series A should have test windows in per-series mode"
        assert 1 in test_series
