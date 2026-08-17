from typing import Any
from warnings import warn

import numpy as np
import torch


def random_series_split(
    total_series: int,
    train_val_test_split: tuple[float, float, float],
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
      Computes cutoffs from a global timeline across all series,
      ensuring a single boundary that prevents cross-series leakage.

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
    if temporal_cutoffs is not None and train_val_test_split != (0.7, 0.15, 0.15):
        warn(
            "Both 'temporal_cutoffs' and a custom 'train_val_test_split' were "
            "provided. 'temporal_cutoffs' takes precedence and the percentage "
            "split will be ignored.",
            UserWarning,
            stacklevel=2,
        )
    if temporal_cutoffs is not None:
        return _split_absolute(windows, series_timestamps, temporal_cutoffs)
    return _split_percentage(windows, series_timestamps, train_val_test_split)


def group_time_split(
    windows: list[tuple[int, int, int, int]],
    series_timestamps: dict[int, np.ndarray],
    train_val_test_split: tuple[float, float, float],
    group_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> tuple[
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
]:
    """Group-Time-Split: First split by group, then temporally within train groups.

    Phase 1 randomly assigns groups to train,val and test. Phase 2 applies a
    temporal percentage split on windows that belong to train groups only.
    Windows from val and test groups are held out entirely, so the model never
    sees those groups during training,testing generalization to new groups.

    Parameters
    ----------
    windows : list of (series_idx, start_idx, enc_len, pred_len)
        All sliding windows across all series.
    series_timestamps : dict mapping series_idx to np.ndarray of timestamps
    train_val_test_split : temporal split ratios for train group only
    group_split : group assignment ratios for Phase 1

    Returns
    -------
    train_windows, val_windows, test_windows
    """
    if not windows:
        return [], [], []

    # sort for deterministic group indexing
    all_series = sorted(series_timestamps.keys())
    total_groups = len(all_series)
    
    # phase 1: random split at the series level
    train_group_ids, val_group_ids, test_group_ids = random_series_split(
        total_groups, group_split
    )

    # map split indices back to actual series IDs
    train_groups = {all_series[i] for i in train_group_ids.tolist()}
    val_groups = {all_series[i] for i in val_group_ids.tolist()}
    test_groups = {all_series[i] for i in test_group_ids.tolist()}

    # partition windows by their group assignment
    train_group_windows = [w for w in windows if w[0] in train_groups]
    val_group_windows = [w for w in windows if w[0] in val_groups]
    test_group_windows = [w for w in windows if w[0] in test_groups]

    # isolate timestamps exclusively for the train groups
    train_group_ts = {
        s_idx: ts for s_idx, ts in series_timestamps.items() if s_idx in train_groups
    }

    # phase 2: temporal percentage split within the train groups
    if train_group_windows and train_group_ts:
        t_win, v_win, te_win = _split_percentage(
            train_group_windows, train_group_ts, train_val_test_split
        )
    else:
        t_win, v_win, te_win = [], [], []

    # combine temporally split val/test windows with fully held-out val/test groups
    return t_win, v_win + val_group_windows, te_win + test_group_windows


def _get_window_end_time(
    w: tuple[int, int, int, int],
    series_timestamps: dict[int, np.ndarray],
):
    """Compute the real-world end timestamp of a window.

    Returns the raw timestamp value (int, float, or datetime64) so that
    comparisons work regardless of the timestamp dtype.
    """
    series_idx, start_idx, enc_len, pred_len = w
    timestamps = series_timestamps[series_idx]
    end_idx = min(start_idx + enc_len + pred_len - 1, len(timestamps) - 1)
    return timestamps[end_idx]


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
    """Percentage mode: compute cutoffs per series from each series' own timeline.

    For each series, picks cutoff values at the requested percentile
    positions within that series' unique timestamps. This ensures each
    series contributes proportionally to train/val/test regardless of
    its absolute timestamp range.

    Parameters
    ----------
        windows (list[tuple[int, int, int, int]]): A list containing data windows,
            where each window is represented by a tuple of 4 integers.
        series_timestamps (dict[int, np.ndarray]): A dictionary mapping a time
            series index (integer) to a NumPy array of all timestamps for that series.
        train_val_test_split (tuple[float, float, float]): A tuple containing three
            floats representing the requested percentage proportions for the training,
            validation, and test datasets, respectively (e.g., (0.7, 0.2, 0.1)).

    Returns
    -------
        tuple[list[tuple[int, int, int, int]],
              list[tuple[int, int, int, int]],
              list[tuple[int, int, int, int]]]:
            A tuple containing three separate lists of windows representing the finalized
            training windows, validation windows, and test windows, respectively.
    """
    series_cutoffs: dict[int, tuple | None] = {}
    for s_idx, timestamps in series_timestamps.items():
        # deduplicate so percentile positions reflect distinct time steps
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

    # fallback: no series had enough timestamps for proper cutoffs, split by count
    if all(v is None for v in series_cutoffs.values()):
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

        # unsplittable series default to train
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
