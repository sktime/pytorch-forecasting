"""
Timeseries dataset - v2 prototype.

Beta version, experimental - use for testing but not in production.
"""

from typing import Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pytorch_forecasting.utils._coerce import _coerce_to_list

#######################################################################################
# Disclaimer: This dataset class is still work in progress and experimental, please
# use with care. This class is a basic skeleton of how the data-handling pipeline may
# look like in the future.
# This is the D1 layer that is a "Raw Dataset Layer" mainly for raw data ingestion
# and turning the data to tensors.
# For now, this pipeline handles the simplest situation: The whole data can be loaded
# into the memory.
#######################################################################################


class TimeSeries(Dataset):
    """PyTorch Dataset for time series data stored in pandas DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        data frame with sequence data.
        Column names must all be str, and contain str as referred to below.
    data_future : pd.DataFrame, optional, default=None
        data frame with future data.
        Column names must all be str, and contain str as referred to below.
        May contain only columns that are in time, group, weight, known, or static.
    time : str, optional, default = first col not in group_ids, weight, target, static.
        integer typed column denoting the time index within ``data``.
        This column is used to determine the sequence of samples.
        If there are no missing observations,
        the time index should increase by ``+1`` for each subsequent sample.
        The first time_idx for each series does not necessarily
        have to be ``0`` but any value is allowed.
    target : str or List[str], optional, default = last column (at iloc -1)
        column(s) in ``data`` denoting the forecasting target.
        Can be categorical or numerical dtype.
    group : List[str], optional, default = None
        list of column names identifying a time series instance within ``data``.
        This means that the ``group`` together uniquely identify an instance,
        and ``group`` together with ``time`` uniquely identify a single observation
        within a time series instance.
        If ``None``, the dataset is assumed to be a single time series.
    weight : str, optional, default=None
        column name for weights.
        If ``None``, it is assumed that there is no weight column.
    num : list of str, optional, default = all columns with dtype in "fi"
        list of numerical variables in ``data``,
        list may also contain list of str, which are then grouped together.
    cat : list of str, optional, default = all columns with dtype in "Obc"
        list of categorical variables in ``data``,
        list may also contain list of str, which are then grouped together
        (e.g. useful for product categories).
    known : list of str, optional, default = all variables
        list of variables that change over time and are known in the future,
        list may also contain list of str, which are then grouped together
        (e.g. useful for special days or promotion categories).
    unknown : list of str, optional, default = no variables
        list of variables that are not known in the future,
        list may also contain list of str, which are then grouped together
        (e.g. useful for weather categories).
    static : list of str, optional, default = all variables not in known, unknown
        list of variables that do not change over time,
        list may also contain list of str, which are then grouped together.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        data_future: Optional[pd.DataFrame] = None,
        time: Optional[str] = None,
        target: Optional[Union[str, list[str]]] = None,
        group: Optional[list[str]] = None,
        weight: Optional[str] = None,
        num: Optional[list[Union[str, list[str]]]] = None,
        cat: Optional[list[Union[str, list[str]]]] = None,
        known: Optional[list[Union[str, list[str]]]] = None,
        unknown: Optional[list[Union[str, list[str]]]] = None,
        static: Optional[list[Union[str, list[str]]]] = None,
    ):

        self.data = data
        self.data_future = data_future
        self.time = time
        self.target = _coerce_to_list(target)
        self.group = _coerce_to_list(group)
        self.weight = weight
        self.num = _coerce_to_list(num)
        self.cat = _coerce_to_list(cat)
        self.known = _coerce_to_list(known)
        self.unknown = _coerce_to_list(unknown)
        self.static = _coerce_to_list(static)

        warn(
            "TimeSeries is part of an experimental rework of the "
            "pytorch-forecasting data layer, "
            "scheduled for release with v2.0.0. "
            "The API is not stable and may change without prior warning. "
            "For beta testing, but not for stable production use. "
            "Feedback and suggestions are very welcome in "
            "pytorch-forecasting issue 1736, "
            "https://github.com/sktime/pytorch-forecasting/issues/1736",
            UserWarning,
        )

        self.feature_cols = [
            col
            for col in data.columns
            if col not in [self.time] + self.group + [self.weight] + self.target
        ]
        if self.group:
            self._groups = self.data.groupby(self.group).groups
            self._group_ids = list(self._groups.keys())
        else:
            self._groups = {"_single_group": self.data.index}
            self._group_ids = ["_single_group"]

        self._prepare_metadata()

    def _prepare_metadata(self):
        """Prepare metadata for the dataset.

        The funcion returns metadata that contains:

        * ``cols``: dict { 'y': list[str], 'x': list[str], 'st': list[str] }
          Names of columns for y, x, and static features.
          List elements are in same order as column dimensions.
          Columns not appearing are assumed to be named (x0, x1, etc.),
          (y0, y1, etc.), (st0, st1, etc.).
        * ``col_type``: dict[str, str]
          maps column names to data types "F" (numerical) and "C" (categorical).
          Column names not occurring are assumed "F".
        * ``col_known``: dict[str, str]
          maps column names to "K" (future known) or "U" (future unknown).
          Column names not occurring are assumed "K".
        """
        self.metadata = {
            "cols": {
                "y": self.target,
                "x": self.feature_cols,
                "st": self.static,
            },
            "col_type": {},
            "col_known": {},
        }

        all_cols = self.target + self.feature_cols + self.static
        for col in all_cols:
            self.metadata["col_type"][col] = "C" if col in self.cat else "F"

            self.metadata["col_known"][col] = "K" if col in self.known else "U"

    def __len__(self) -> int:
        """Return number of time series in the dataset."""
        return len(self._group_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get time series data for given index.

        Returns
        -------
        t : numpy.ndarray of shape (n_timepoints,)
            Time index for each time point in the past or present. Aligned with `y`,
            and `x` not ending in `f`.

        y : torch.Tensor of shape (n_timepoints, n_targets)
            Target values for each time point. Rows are time points, aligned with `t`.

        x : torch.Tensor of shape (n_timepoints, n_features)
            Features for each time point. Rows are time points, aligned with `t`.

        group : torch.Tensor of shape (n_groups,)
            Group identifiers for time series instances.

        st : torch.Tensor of shape (n_static_features,)
            Static features.

        cutoff_time : float or numpy.float64
            Cutoff time for the time series instance.

        Other Returns
        -------------
        weights : torch.Tensor of shape (n_timepoints,), optional
            Only included if weights are not `None`.
        """
        group_id = self._group_ids[index]

        if self.group:
            mask = self._groups[group_id]
            data = self.data.loc[mask]
        else:
            data = self.data

        cutoff_time = data[self.time].max()

        result = {
            "t": data[self.time].values,
            "y": torch.tensor(data[self.target].values),
            "x": torch.tensor(data[self.feature_cols].values),
            "group": torch.tensor([hash(str(group_id))]),
            "st": torch.tensor(data[self.static].iloc[0].values if self.static else []),
            "cutoff_time": cutoff_time,
        }

        if self.data_future is not None:
            if self.group:
                future_mask = self.data_future.groupby(self.group).groups[group_id]
                future_data = self.data_future.loc[future_mask]
            else:
                future_data = self.data_future

            combined_times = np.concatenate(
                [data[self.time].values, future_data[self.time].values]
            )
            combined_times = np.unique(combined_times)
            combined_times.sort()

            num_timepoints = len(combined_times)
            x_merged = np.full((num_timepoints, len(self.feature_cols)), np.nan)
            y_merged = np.full((num_timepoints, len(self.target)), np.nan)

            current_time_indices = {t: i for i, t in enumerate(combined_times)}
            for i, t in enumerate(data[self.time].values):
                idx = current_time_indices[t]
                x_merged[idx] = data[self.feature_cols].values[i]
                y_merged[idx] = data[self.target].values[i]

            for i, t in enumerate(future_data[self.time].values):
                if t in current_time_indices:
                    idx = current_time_indices[t]
                    for j, col in enumerate(self.known):
                        if col in self.feature_cols:
                            feature_idx = self.feature_cols.index(col)
                            x_merged[idx, feature_idx] = future_data[col].values[i]

            result.update(
                {
                    "t": combined_times,
                    "x": torch.tensor(x_merged, dtype=torch.float32),
                    "y": torch.tensor(y_merged, dtype=torch.float32),
                }
            )

        if self.weight:
            if self.data_future is not None and self.weight in self.data_future.columns:
                weights_merged = np.full(num_timepoints, np.nan)
                for i, t in enumerate(data[self.time].values):
                    idx = current_time_indices[t]
                    weights_merged[idx] = data[self.weight].values[i]

                for i, t in enumerate(future_data[self.time].values):
                    if t in current_time_indices and self.weight in future_data.columns:
                        idx = current_time_indices[t]
                        weights_merged[idx] = future_data[self.weight].values[i]

                result["weights"] = torch.tensor(weights_merged, dtype=torch.float32)
            else:
                result["weights"] = torch.tensor(
                    data[self.weight].values, dtype=torch.float32
                )

        return result

    def get_metadata(self) -> dict:
        """Return metadata about the dataset.

        Returns
        -------
        Dict
            Dictionary containing:
            - cols: column names for y, x, and static features
            - col_type: mapping of columns to their types (F/C)
            - col_known: mapping of columns to their future known status (K/U)
        """
        return self.metadata
