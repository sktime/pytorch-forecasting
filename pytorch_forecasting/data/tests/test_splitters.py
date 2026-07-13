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
        """Critical test: no train window's end timestamp should exceed
        any val window's end timestamp."""
        windows, series_timestamps = overlapping_series_setup

        train_w, val_w, test_w = temporal_window_split(
            windows, (0.7, 0.15, 0.15), series_timestamps
        )

        def get_end_time(w):
            s_idx, start, enc, pred = w
            end = min(start + enc + pred - 1, len(series_timestamps[s_idx]) - 1)
            return float(series_timestamps[s_idx][end])

        if train_w and val_w:
            max_train_time = max(get_end_time(w) for w in train_w)
            min_val_time = min(get_end_time(w) for w in val_w)
            assert max_train_time <= min_val_time, (
                f"LEAKAGE: max train time {max_train_time} > "
                f"min val time {min_val_time}"
            )

        if val_w and test_w:
            max_val_time = max(get_end_time(w) for w in val_w)
            min_test_time = min(get_end_time(w) for w in test_w)
            assert max_val_time <= min_test_time, (
                f"LEAKAGE: max val time {max_val_time} > "
                f"min test time {min_test_time}"
            )

    def test_global_cutoff_with_overlapping_series(self, overlapping_series_setup):
        """Verify both series are split by the SAME global cutoff."""
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
        """When 2 series aren't overlapping"""
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

        # With global range 0..47, train_cutoff ≈ 32.9
        # Series A (ends at 23) → all in train
        series_a_in_train = [w for w in train_w if w[0] == 0]
        assert len(series_a_in_train) == len(
            windows_a
        ), "All of Series A should be in train since it ends before cutoff"
