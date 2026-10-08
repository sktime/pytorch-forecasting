"""
Splitting strategies for train/val/test partitioning of time series data.

Provides class-based splitters for random, stratified, temporal,
and group-time splitting consumed by the data modules.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from ._primitives import (
    classify_windows_by_cutoffs,
    compute_split_boundaries,
)


class BaseSplitter(ABC):
    """Base class for all time-series splitting strategies."""

    @property
    def has_window_split(self) -> bool:
        """Returns True if the splitter performs temporal/window-level splitting."""
        return False

    @abstractmethod
    def split_series(
        self, total_series: int, dataset: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Splits the dataset at the series level.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Train, validation,
                and test indices.
        """
        pass

    def split_windows(
        self, windows: list[tuple[int, int, int, int]], dataset: Any
    ) -> tuple[list, list, list]:
        """Splits the dataset at the sliding window level.

        Only called if `has_window_split` is True.
        """
        raise NotImplementedError(
            "This splitter does not support window-level splitting."
        )


class RandomSplitter(BaseSplitter):
    """Randomly splits the dataset at the series (group) level.

    This ensures all data points from a specific group stay within the same fold.

    Parameters:
        train_val_test_split (tuple[float, float, float]): Proportions for
            train, val, and test folds.
    """

    def __init__(
        self, train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15)
    ):
        self.train_val_test_split = train_val_test_split

    def split_series(
        self, total_series: int, dataset: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shuffle series indices and slice them into train/val/test."""
        split_indices = torch.randperm(total_series)
        train_end, val_end = compute_split_boundaries(
            total_series, self.train_val_test_split
        )

        train_indices = split_indices[:train_end]
        val_indices = split_indices[train_end:val_end]
        test_indices = split_indices[val_end:]

        return train_indices, val_indices, test_indices


class StratifiedSplitter(BaseSplitter):
    """Stratified split to ensure class distributions are preserved.

    It extracts a class label for each series (e.g., majority target or a
    static categorical feature).

    Parameters:
        target_idx (int): The index of the target variable to stratify on.
        train_val_test_split (tuple[float, float, float]): Proportions for
            train, val, and test folds.
    """

    def __init__(
        self,
        target_idx: int = 0,
        train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        self.target_idx = target_idx
        self.train_val_test_split = train_val_test_split

    def split_series(
        self, total_series: int, dataset: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split series while preserving class balance."""
        from sklearn.model_selection import StratifiedShuffleSplit

        labels = []
        for i in range(total_series):
            sample = dataset[i]
            st = sample.get("st")
            label = (
                st[self.target_idx].item()
                if st is not None and len(st) > self.target_idx
                else 0
            )
            labels.append(label)

        labels = np.array(labels)
        indices = np.arange(total_series)

        test_val_size = self.train_val_test_split[1] + self.train_val_test_split[2]
        val_prop = (
            self.train_val_test_split[1] / test_val_size if test_val_size > 0 else 0
        )

        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_val_size)
        try:
            train_idx, val_test_idx = next(sss1.split(indices, labels))
        except ValueError:
            random_splitter = RandomSplitter(self.train_val_test_split)
            return random_splitter.split_series(total_series, dataset)

        if self.train_val_test_split[2] == 0:
            val_idx = val_test_idx
            test_idx = np.array([])
        else:
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_prop)
            try:
                val_idx_rel, test_idx_rel = next(
                    sss2.split(val_test_idx, labels[val_test_idx])
                )
                val_idx = val_test_idx[val_idx_rel]
                test_idx = val_test_idx[test_idx_rel]
            except ValueError:
                split_pt = int(len(val_test_idx) * val_prop)
                val_idx = val_test_idx[:split_pt]
                test_idx = val_test_idx[split_pt:]

        return torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)


class TemporalSplitter(BaseSplitter):
    """Split windows temporally to prevent data leakage.

    Supports two modes:
    - **Absolute mode** (when `temporal_cutoffs` is provided): Uses global
      timestamp boundaries.
    - **Percentage mode** (default, when `temporal_cutoffs` is None): Computes
      cutoffs per series from each series' own timeline.

    Parameters:
        train_val_test_split (tuple[float, float, float]): Proportions for the
            three folds. Used only in percentage mode.
        temporal_cutoffs (dict[str, float] | None): Absolute temporal cutoffs.
            Expected keys:
            - `"end_train"` : float  last timestamp included in training
            - `"start_test"` : float  first timestamp included in testing
    """

    def __init__(
        self,
        train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
        temporal_cutoffs: dict[str, float] | None = None,
    ):
        self.train_val_test_split = train_val_test_split
        self.temporal_cutoffs = temporal_cutoffs

    @property
    def has_window_split(self) -> bool:
        """Always True  temporal splitting operates on windows, not series."""
        return True

    def split_series(
        self, total_series: int, dataset: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all series for every fold. Actual splitting happens at window
        level."""
        import warnings

        warnings.warn(
            "Using TemporalSplitter: all groups appear in every fold. Consider "
            "adding the group column as a known categorical feature so the model "
            "can leverage group identity during training.",
            UserWarning,
            stacklevel=2,
        )
        indices = torch.arange(total_series)
        return indices, indices, indices

    def split_windows(
        self, windows: list[tuple[int, int, int, int]], dataset: Any
    ) -> tuple[list, list, list]:
        """Route to absolute or percentage mode based on whether cutoffs were
        provided."""
        if not windows:
            return [], [], []

        series_timestamps = {}
        for w in windows:
            s_idx = w[0]
            if s_idx not in series_timestamps:
                idx = s_idx.item() if isinstance(s_idx, torch.Tensor) else s_idx
                series_timestamps[s_idx] = dataset[idx]["t"]

        if self.temporal_cutoffs is not None:
            return self._split_absolute(
                windows, series_timestamps, self.temporal_cutoffs
            )
        return self._split_percentage(
            windows, series_timestamps, self.train_val_test_split
        )

    def _split_absolute(self, windows, series_timestamps, temporal_cutoffs):
        """Classify windows using explicit timestamp boundaries shared across
        all series."""
        end_train = temporal_cutoffs["end_train"]
        start_test = temporal_cutoffs.get("start_test", end_train)
        if start_test < end_train:
            raise ValueError(
                f"start_test ({start_test}) must be >= end_train ({end_train})"
            )

        all_series = {w[0] for w in windows}
        cutoffs_map = {s_idx: (end_train, start_test) for s_idx in all_series}
        return classify_windows_by_cutoffs(windows, series_timestamps, cutoffs_map)

    def _split_percentage(self, windows, series_timestamps, train_val_test_split):
        """Compute per-series cutoffs from each series' own timeline, then
        classify."""
        series_cutoffs: dict[int, tuple | None] = {}
        for s_idx, timestamps in series_timestamps.items():
            unique_ts = np.unique(timestamps)
            n = len(unique_ts)
            if n <= 1:
                series_cutoffs[s_idx] = None
                continue
            train_pos = min(int(np.round(train_val_test_split[0] * n)), n - 1)
            val_pos = min(
                int(np.round((train_val_test_split[0] + train_val_test_split[1]) * n)),
                n - 1,
            )
            series_cutoffs[s_idx] = (unique_ts[train_pos], unique_ts[val_pos])

        if all(v is None for v in series_cutoffs.values()):
            train_end, val_end = compute_split_boundaries(
                len(windows), train_val_test_split
            )
            return windows[:train_end], windows[train_end:val_end], windows[val_end:]

        return classify_windows_by_cutoffs(windows, series_timestamps, series_cutoffs)


class GroupTimeSplitter(BaseSplitter):
    """Group-Time-Split: First split by group, then temporally within train groups.

    Phase 1 randomly assigns groups to train, val, and test. Phase 2 applies a
    temporal percentage split on windows that belong to train groups only.

    Parameters:
        train_val_test_split (tuple[float, float, float]): Temporal split ratios
            for train group only.
        group_split (tuple[float, float, float]): Group assignment ratios for
            Phase 1.
    """

    def __init__(
        self,
        train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
        group_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        self.train_val_test_split = train_val_test_split
        self.group_split = group_split

    @property
    def has_window_split(self) -> bool:
        """Always True  group-time splitting operates on windows, not
        series."""
        return True

    def split_series(
        self, total_series: int, dataset: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all series for every fold. Actual splitting happens at window
        level."""
        import warnings

        warnings.warn(
            "Using GroupTimeSplitter: validation and test sets contain unseen groups."
            "Do NOT use the group column as a categorical feature"
            "the model cannot generalize embeddings to groups it has never seen.",
            UserWarning,
            stacklevel=2,
        )
        indices = torch.arange(total_series)
        return indices, indices, indices

    def split_windows(
        self, windows: list[tuple[int, int, int, int]], dataset: Any
    ) -> tuple[list, list, list]:
        """
        Phase 1: assign groups randomly.
        Phase 2: split train-group windows temporally.

        Parameters:
            windows (list[tuple[int, int, int, int]]): List of windows to split.
            dataset (Any): Dataset containing the time series.

        Returns:
            tuple[list, list, list]: Tuple of lists containing train,
                validation, and test windows.
        """
        if not windows:
            return [], [], []

        series_timestamps = {}
        for w in windows:
            s_idx = w[0]
            if s_idx not in series_timestamps:
                idx = s_idx.item() if isinstance(s_idx, torch.Tensor) else s_idx
                series_timestamps[s_idx] = dataset[idx]["t"]

        all_series = sorted(series_timestamps.keys())
        total_groups = len(all_series)

        # phase 1: random split at the series level
        random_splitter = RandomSplitter(self.group_split)
        train_group_ids, val_group_ids, test_group_ids = random_splitter.split_series(
            total_groups, None
        )

        train_groups = {all_series[i] for i in train_group_ids.tolist()}
        val_groups = {all_series[i] for i in val_group_ids.tolist()}
        test_groups = {all_series[i] for i in test_group_ids.tolist()}

        train_group_windows = [w for w in windows if w[0] in train_groups]
        val_group_windows = [w for w in windows if w[0] in val_groups]
        test_group_windows = [w for w in windows if w[0] in test_groups]

        train_group_ts = {
            s_idx: ts
            for s_idx, ts in series_timestamps.items()
            if s_idx in train_groups
        }

        # phase 2: temporal percentage split within the train groups
        if train_group_windows and train_group_ts:
            temporal_splitter = TemporalSplitter(self.train_val_test_split)
            t_win, v_win, te_win = temporal_splitter._split_percentage(
                train_group_windows, train_group_ts, self.train_val_test_split
            )
        else:
            t_win, v_win, te_win = [], [], []

        return t_win, v_win + val_group_windows, te_win + test_group_windows
