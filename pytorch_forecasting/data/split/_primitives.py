"""
Low-level building blocks shared by splitting strategies.

These are internal helpers consumed by ``splitters.py``.
"""

from __future__ import annotations

import numpy as np


def get_window_end_time(
    w: tuple[int, int, int, int],
    series_timestamps: dict[int, np.ndarray],
):
    """Compute the real-world end timestamp of a window."""
    series_idx, start_idx, enc_len, pred_len = w
    timestamps = series_timestamps[series_idx]
    end_idx = min(start_idx + enc_len + pred_len - 1, len(timestamps) - 1)
    return timestamps[end_idx]


def classify_windows_by_cutoffs(
    windows: list[tuple[int, int, int, int]],
    series_timestamps: dict[int, np.ndarray],
    cutoffs_map: dict[int, tuple | None],
) -> tuple[
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
]:
    """Classify each window into train/val/test using pre-computed cutoffs.

    Parameters
    ----------
    cutoffs_map : dict
        ``{series_idx: (train_cutoff, val_cutoff)}`` or ``None``
        for unsplittable series (those default to train).
    """
    train_w, val_w, test_w = [], [], []

    for w in windows:
        s_idx = w[0]
        cutoffs = cutoffs_map.get(s_idx)
        end_time = get_window_end_time(w, series_timestamps)

        if cutoffs is None:
            train_w.append(w)
        else:
            train_cutoff, val_cutoff = cutoffs
            if end_time <= train_cutoff:
                train_w.append(w)
            elif end_time <= val_cutoff:
                val_w.append(w)
            else:
                test_w.append(w)

    return train_w, val_w, test_w


def compute_split_boundaries(
    total: int,
    ratios: tuple[float, float, float],
) -> tuple[int, int]:
    """Compute integer split positions from proportional ratios.

    Returns ``(train_end, val_end)`` such that:
    - train = ``[:train_end]``
    - val   = ``[train_end:val_end]``
    - test  = ``[val_end:]``
    """
    train_end = int(np.round(ratios[0] * total))
    if train_end == 0 and ratios[0] > 0 and total > 0:
        train_end = 1

    val_end = train_end + int(np.round(ratios[1] * total))
    if val_end > total:
        val_end = total

    return train_end, val_end
