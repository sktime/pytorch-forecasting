from typing import Any

import numpy as np
import torch


def random_series_split(
    total_series: int, train_val_test_split: tuple[float, float, float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly splits the dataset at the series (group) level.
    This ensures all data points from a specific group stay within the
    same fold.
    """
    split_indices = torch.randperm(total_series)

    train_size = int(np.round(train_val_test_split[0] * total_series))
    if train_size == 0 and train_val_test_split[0] > 0 and total_series > 0:
        train_size = 1

    val_size = int(np.round(train_val_test_split[1] * total_series))
    # ensure we don't exceed total_series
    if train_size + val_size > total_series:
        val_size = total_series - train_size

    train_indices = split_indices[:train_size]
    val_indices = split_indices[train_size : train_size + val_size]
    test_indices = split_indices[train_size + val_size :]

    return train_indices, val_indices, test_indices


def stratified_series_split(
    time_series_dataset: Any,
    target_idx: int,
    train_val_test_split: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stratified split to ensure class distributions are preserved.
    It extracts a class label for each series (e.g., majority target or
    a static categorical feature).
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    total_series = len(time_series_dataset)
    labels = []

    # Extract the stratify label for each series.
    # For time series, this is usually a static feature or majority target class.
    # In this basic implementation, we assume we stratify on the first static
    # categorical feature.
    for i in range(total_series):
        sample = time_series_dataset[i]
        st = sample.get("st")
        # Ensure we have a label, if none, we default to 0
        label = st[target_idx].item() if st is not None and len(st) > target_idx else 0
        labels.append(label)

    labels = np.array(labels)
    indices = np.arange(total_series)

    # Compute test + val proportion
    test_val_size = train_val_test_split[1] + train_val_test_split[2]
    val_prop = train_val_test_split[1] / test_val_size if test_val_size > 0 else 0

    # First split: Train vs (Val + Test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_val_size)
    try:
        train_idx, val_test_idx = next(sss1.split(indices, labels))
    except ValueError:
        # Fallback if classes are too few
        return random_series_split(total_series, train_val_test_split)

    # Second split: Val vs Test
    if train_val_test_split[2] == 0:
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
            # Fallback
            split_pt = int(len(val_test_idx) * val_prop)
            val_idx = val_test_idx[:split_pt]
            test_idx = val_test_idx[split_pt:]

    return torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)


def temporal_window_split(
    windows: list[tuple[int, int, int, int]],
    train_val_test_split: tuple[float, float, float],
    series_timestamps: dict[int, np.ndarray],
    temporal_cutoffs: dict[str, float] | None = None,
) -> tuple[
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
]:
    """Split windows temporally to prevent data leakage.
    Supports two modes:
    - **Absolute mode** (when ``temporal_cutoffs`` is provided):
      Uses global timestamp boundaries. All series share the same cutoffs.
    - **Percentage mode** (default, when ``temporal_cutoffs`` is None):
      Computes cutoffs per-series based on each series' own time range,
      so each series contributes proportionally to train/val/test.
    Parameters
    ----------
    windows : list of (series_idx, start_idx, enc_len, pred_len)
        All sliding windows across all series.
    train_val_test_split : (train_ratio, val_ratio, test_ratio)
        Proportions for the three folds. Used only in percentage mode.
    series_timestamps : dict mapping series_idx to np.ndarray of timestamps
        The actual time values for each series.
    temporal_cutoffs : dict or None, default=None
        If provided, activates absolute mode. Expected keys:
        - ``"end_train"`` : float — last timestamp included in training
        - ``"start_test"`` : float — first timestamp included in testing
        Windows with end_time <= end_train go to train.
        Windows with end_time >= start_test go to test.
        Windows in between go to validation.
        If ``start_test`` is not provided, it defaults to ``end_train``
        (no gap between val and test).
    Returns
    -------
    train_windows, val_windows, test_windows
        Three lists of window tuples.
    """
    if not windows:
        return [], [], []
    if temporal_cutoffs is not None:
        return _split_absolute(windows, series_timestamps, temporal_cutoffs)
    return _split_percentage(windows, series_timestamps, train_val_test_split)


def _get_window_end_time(
    w: tuple[int, int, int, int],
    series_timestamps: dict[int, np.ndarray],
) -> float:
    """Compute the real-world end timestamp of a window.
    Why end_time? Because data leakage is about whether the MODEL
    has seen future information. The window's last timestep is the
    latest point it 'sees', so that's what we compare against cutoffs.
    """
    series_idx, start_idx, enc_len, pred_len = w
    timestamps = series_timestamps[series_idx]
    end_idx = min(start_idx + enc_len + pred_len - 1, len(timestamps) - 1)
    return float(timestamps[end_idx])


def _split_absolute(
    windows: list[tuple[int, int, int, int]],
    series_timestamps: dict[int, np.ndarray],
    temporal_cutoffs: dict[str, float],
) -> tuple[
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
]:
    """Absolute mode: split using user-specified timestamp boundaries.
    The user says: "train ends at time X, test starts at time Y."
    Everything between X and Y is validation.
    """
    end_train = temporal_cutoffs["end_train"]
    start_test = temporal_cutoffs.get("start_test", end_train)
    if start_test < end_train:
        raise ValueError(
            f"start_test ({start_test}) must be >= end_train ({end_train})"
        )
    train_windows, val_windows, test_windows = [], [], []
    for w in windows:
        end_time = _get_window_end_time(w, series_timestamps)
        if end_time <= end_train:
            train_windows.append(w)
        elif end_time >= start_test:
            test_windows.append(w)
        else:
            val_windows.append(w)
    return train_windows, val_windows, test_windows


def _split_percentage(
    windows: list[tuple[int, int, int, int]],
    series_timestamps: dict[int, np.ndarray],
    train_val_test_split: tuple[float, float, float],
) -> tuple[
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
]:
    """Percentage mode: compute cutoffs per-series from its own time range."""
    series_cutoffs = {}
    for s_idx, timestamps in series_timestamps.items():
        t_min = float(np.min(timestamps))
        t_max = float(np.max(timestamps))
        t_range = t_max - t_min
        if t_range == 0:
            series_cutoffs[s_idx] = None
        else:
            train_cutoff = t_min + train_val_test_split[0] * t_range
            val_cutoff = train_cutoff + train_val_test_split[1] * t_range
            series_cutoffs[s_idx] = (train_cutoff, val_cutoff)
    all_zero = all(v is None for v in series_cutoffs.values())

    if all_zero:
        total_w = len(windows)
        train_end = int(np.round(train_val_test_split[0] * total_w))

        if train_end == 0 and train_val_test_split[0] > 0 and total_w > 0:
            train_end = 1
        val_end = train_end + int(np.round(train_val_test_split[1] * total_w))
        val_end = min(val_end, total_w)
        return windows[:train_end], windows[train_end:val_end], windows[val_end:]

    train_windows, val_windows, test_windows = [], [], []
    for w in windows:
        s_idx = w[0]
        cutoffs = series_cutoffs.get(s_idx)
        end_time = _get_window_end_time(w, series_timestamps)
        if cutoffs is None:
            train_windows.append(w)
        else:
            train_cutoff, val_cutoff = cutoffs
            if end_time <= train_cutoff:
                train_windows.append(w)
            elif end_time <= val_cutoff:
                val_windows.append(w)
            else:
                test_windows.append(w)
    return train_windows, val_windows, test_windows
