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
    target_idx: int,  # Or a categorical static feature to stratify on
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
        label = st[0].item() if st is not None and len(st) > 0 else 0
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
) -> tuple[
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
]:
    """Split windows by global timestamp cutoffs to prevent data leakage.

    Computes global time cutoffs across all series and assigns each window
    to train/val/test based on the window's end timestamp (the last prediction
    time step). This prevents future information from leaking into training.

    Parameters
    ----------
    windows : list of (series_idx, start_idx, enc_len, pred_len)
        All sliding windows across all series.
    train_val_test_split : (train_ratio, val_ratio, test_ratio)
        Proportions for the three folds. Must sum to 1.0.
    series_timestamps : dict mapping series_idx to np.ndarray of timestamps
        The actual time index values for each series (from TimeSeries["t"]).

    Returns
    -------
    train_windows, val_windows, test_windows
        Three lists of window tuples, split by global time boundaries.
    """
    if not windows:
        return [], [], []

    # Find global time range across ALL series
    all_timestamps = np.concatenate(list(series_timestamps.values()))
    global_min = float(np.min(all_timestamps))
    global_max = float(np.max(all_timestamps))
    time_range = global_max - global_min

    # Edge case: all timestamps identical → fall back to positional split
    if time_range == 0:
        total_w = len(windows)
        train_end = int(np.round(train_val_test_split[0] * total_w))
        if train_end == 0 and train_val_test_split[0] > 0 and total_w > 0:
            train_end = 1
        val_end = train_end + int(np.round(train_val_test_split[1] * total_w))
        if val_end > total_w:
            val_end = total_w
        return windows[:train_end], windows[train_end:val_end], windows[val_end:]

    # Compute cutoff points on the actual timeline
    train_cutoff = global_min + train_val_test_split[0] * time_range
    val_cutoff = train_cutoff + train_val_test_split[1] * time_range

    # Assign each window based on its END timestamp
    train_windows, val_windows, test_windows = [], [], []

    for w in windows:
        series_idx, start_idx, enc_len, pred_len = w
        timestamps = series_timestamps[series_idx]

        # The window covers [start_idx, start_idx + enc_len + pred_len - 1]
        end_idx = start_idx + enc_len + pred_len - 1
        # Clamp to array bounds
        end_idx = min(end_idx, len(timestamps) - 1)

        end_time = float(timestamps[end_idx])

        if end_time <= train_cutoff:
            train_windows.append(w)
        elif end_time <= val_cutoff:
            val_windows.append(w)
        else:
            test_windows.append(w)

    return train_windows, val_windows, test_windows
