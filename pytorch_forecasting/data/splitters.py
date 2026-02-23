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
    train_size = int(train_val_test_split[0] * total_series)
    val_size = int(train_val_test_split[1] * total_series)

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
) -> tuple[
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
]:
    """
    Time-Series Splitting: Implementation of a sliding/expanding window split.
    Instead of splitting series, we take all windows for a series and
    split them temporally. The first X% of windows from series A go to Train,
    next Y% to Val, last Z% to Test.
    """
    # Group windows by series_idx
    series_windows = {}
    for w in windows:
        s_idx = w[0]
        if s_idx not in series_windows:
            series_windows[s_idx] = []
        series_windows[s_idx].append(w)

    train_windows, val_windows, test_windows = [], [], []

    for s_idx, sw in series_windows.items():
        # Ensure windows are sorted by time (start_idx: w[1])
        sw.sort(key=lambda x: x[1])
        total_w = len(sw)

        train_end = int(train_val_test_split[0] * total_w)
        val_end = train_end + int(train_val_test_split[1] * total_w)

        train_windows.extend(sw[:train_end])
        val_windows.extend(sw[train_end:val_end])
        test_windows.extend(sw[val_end:])

    return train_windows, val_windows, test_windows
