"""
Timeseries datatype.

Beta version, experimental - use for testing but not in production.
"""

from dataclasses import replace
from warnings import warn

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pytorch_forecasting.data._metadata import TimeSeriesMetadata
from pytorch_forecasting.utils._coerce import _coerce_to_list
from pytorch_forecasting.utils._validation import _check_column_names, _check_type

#######################################################################################
# Disclaimer: This datatype is still work in progress and experimental, please
# use with care. This class is a basic skeleton of how the data-handling pipeline may
# look like in the future.
# This class is the standard input and output type of the v2 API - a data
# frame plus its schema.
# For now, this pipeline handles the simplest situation: The whole data can be loaded
# into the memory.
#######################################################################################


class TimeSeries(Dataset):
    """Time series data stored in a pandas DataFrame, plus its schema.

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
    known : list of str, optional, default = no variables
        list of variables that change over time and are known in the future,
        list may also contain list of str, which are then grouped together
        (e.g. useful for special days or promotion categories).
    unknown : list of str, optional, default = no variables
        list of variables that are not known in the future,
        list may also contain list of str, which are then grouped together
        (e.g. useful for weather categories).
    static : list of str, optional, default = no variables
        list of variables that do not change over time,
        list may also contain list of str, which are then grouped together.

    Notes
    -----
    Columns that are not passed are inferred:

    * ``target`` is the last column of ``data``.
    * ``time`` is the first column not already used as ``group``, ``target``,
      ``static`` or ``weight``. If no such column exists, ``ValueError``
      is raised - pass ``time`` explicitly.
    * ``num`` and ``cat`` are the dtype split over all columns that are not ``group``,
      ``time`` or ``weight``.

    Parameters are stored on same-named attributes exactly as passed.

    Examples
    --------
    >>> import pandas as pd
    >>> from pytorch_forecasting.data.timeseries import TimeSeries
    >>> df = pd.DataFrame({"time_idx": [0, 1, 2], "target": [1.0, 2.0, 3.0]})
    >>> ts = TimeSeries(df)
    >>> len(ts)
    1
    """

    def __init__(
        self,
        data: pd.DataFrame,
        data_future: pd.DataFrame | None = None,
        time: str | None = None,
        target: str | list[str] | None = None,
        group: list[str] | None = None,
        weight: str | None = None,
        num: list[str | list[str]] | None = None,
        cat: list[str | list[str]] | None = None,
        known: list[str | list[str]] | None = None,
        unknown: list[str | list[str]] | None = None,
        static: list[str | list[str]] | None = None,
    ):
        self.data = data
        self.data_future = data_future
        self.time = time
        self.target = target
        self.group = group
        self.weight = weight
        self.num = num
        self.cat = cat
        self.known = known
        self.unknown = unknown
        self.static = static

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

        self._time = time
        self._target = _coerce_to_list(target)
        self._group = _coerce_to_list(group)
        self._num = _coerce_to_list(num)
        self._cat = _coerce_to_list(cat)
        self._known = _coerce_to_list(known)
        self._unknown = _coerce_to_list(unknown)
        self._static = _coerce_to_list(static)

        self._validate_data()
        self._validate_columns()
        self._infer_columns()

        self.feature_cols = [
            col
            for col in data.columns
            if col not in [self._time] + self._group + [self.weight] + self._target
        ]
        if self._group:
            group_arg = self._group[0] if len(self._group) == 1 else self._group
            self._groups = self.data.groupby(group_arg).groups
            self._group_ids = list(self._groups.keys())
        else:
            self._groups = {"_single_group": self.data.index}
            self._group_ids = ["_single_group"]
        # create mapping from group id to index for efficient lookup
        self._group_to_idx = {gid: i for i, gid in enumerate(self._group_ids)}

        self._prepare_metadata()

    def _validate_data(self):
        """Check the data frames, before anything indexes into them.

        Raises
        ------
        TypeError
            If ``data`` or ``data_future`` is not a ``pandas.DataFrame``.
        ValueError
            If ``data`` has no columns.
        """
        _check_type(self.data, pd.DataFrame, "data")
        _check_type(self.data_future, pd.DataFrame, "data_future", allow_none=True)

        if len(self.data.columns) == 0:
            raise ValueError("`data` has no columns.")

    def _validate_columns(self):
        """Check that every column named by the user exists in ``data``.

        Raises
        ------
        ValueError
            If a named column is not in ``data``, or if the same column is
            given as both known and unknown.
        """
        _check_column_names(
            {
                "time": self._time,
                "target": self._target,
                "group": self._group,
                "weight": self.weight,
                "num": self._num,
                "cat": self._cat,
                "known": self._known,
                "unknown": self._unknown,
                "static": self._static,
            },
            self.data.columns,
        )

        both = set(self._known) & set(self._unknown)
        if both:
            raise ValueError(
                f"columns given as both `known` and `unknown`: {sorted(both)}."
            )

    def _infer_columns(self):
        """Fill in the column roles that were not passed by the user.

        Writes to the underscored attributes only. A role counts as "not passed"
        when its public attribute, which holds the unmodified argument, is None.

        Raises
        ------
        ValueError
            If no time column was passed and none can be inferred.
        """
        cols = list(self.data.columns)

        if self.target is None:
            self._target = [cols[-1]]

        if self._time is None:
            used = set(self._group + self._target + self._static + [self.weight])
            self._time = next((col for col in cols if col not in used), None)
            if self._time is None:
                raise ValueError(
                    f"Could not infer a time column: all columns of `data` "
                    f"({cols}) are already used as group, target, static or "
                    "weight. Pass `time` explicitly."
                )

        if self.num is None or self.cat is None:
            index_like = set(self._group + [self._time, self.weight])
            for col in cols:
                if col in index_like:
                    continue
                kind = self.data[col].dtype.kind
                if self.num is None and kind in "fi":
                    self._num.append(col)
                elif self.cat is None and kind in "Obc":
                    self._cat.append(col)

    def _prepare_metadata(self):
        """Prepare metadata for the dataset.

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
        all_cols = self._target + self.feature_cols + self._static
        self.metadata = TimeSeriesMetadata(
            cols={
                "y": self._target,
                "x": self.feature_cols,
                "st": self._static,
            },
            col_type={col: "C" if col in self._cat else "F" for col in all_cols},
            col_known={col: "K" if col in self._known else "U" for col in all_cols},
        )

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
        time = self._time
        feature_cols = self.feature_cols
        target = self._target
        known = self._known
        static = self._static
        group = self._group
        weight = self.weight
        data_future = self.data_future

        group_id = self._group_ids[index]

        if group:
            mask = self._groups[group_id]
            data = self.data.loc[mask]
        else:
            data = self.data

        cutoff_time = data[time].max()

        # PyTorch wants writeable arrays
        data_vals = data[time].to_numpy(copy=True)
        data_tgt_vals = data[target].to_numpy(copy=True)
        data_feat_vals = data[feature_cols].to_numpy(copy=True)

        result = {
            "t": data_vals,
            "y": torch.tensor(data_tgt_vals),
            "x": torch.tensor(data_feat_vals),
            "group": torch.tensor([self._group_to_idx[group_id]], dtype=torch.long),
            # PyTorch wants writeable arrays
            "st": torch.tensor(
                data[static].iloc[0].to_numpy(copy=True) if static else []
            ),
            "cutoff_time": cutoff_time,
        }

        if data_future is not None:
            if group:
                group_arg = group[0] if len(group) == 1 else group
                future_mask = self.data_future.groupby(group_arg).groups[group_id]
                future_data = self.data_future.loc[future_mask]
            else:
                future_data = self.data_future

            data_fut_vals = future_data[time].values

            combined_times = np.concatenate([data_vals, data_fut_vals])
            combined_times = np.unique(combined_times)
            combined_times.sort()

            num_timepoints = len(combined_times)
            x_merged = np.full((num_timepoints, len(feature_cols)), np.nan)
            y_merged = np.full((num_timepoints, len(target)), np.nan)

            current_time_indices = {t: i for i, t in enumerate(combined_times)}
            for i, t in enumerate(data_vals):
                idx = current_time_indices[t]
                x_merged[idx] = data_feat_vals[i]
                y_merged[idx] = data_tgt_vals[i]

            for i, t in enumerate(data_fut_vals):
                if t in current_time_indices:
                    idx = current_time_indices[t]
                    for j, col in enumerate(known):
                        if col in feature_cols:
                            feature_idx = feature_cols.index(col)
                            # PyTorch wants writeable arrays
                            x_merged[idx, feature_idx] = future_data[col].to_numpy(
                                copy=True
                            )[i]

            result.update(
                {
                    "t": combined_times,
                    "x": torch.tensor(x_merged, dtype=torch.float32),
                    "y": torch.tensor(y_merged, dtype=torch.float32),
                }
            )

        if weight:
            if self.data_future is not None and self.weight in self.data_future.columns:
                weights_merged = np.full(num_timepoints, np.nan)
                for i, t in enumerate(data_vals):
                    idx = current_time_indices[t]
                    # PyTorch wants writeable arrays
                    weights_merged[idx] = data[weight].to_numpy(copy=True)[i]

                for i, t in enumerate(data_fut_vals):
                    if t in current_time_indices and self.weight in future_data.columns:
                        idx = current_time_indices[t]
                        # PyTorch wants writeable arrays
                        weights_merged[idx] = future_data[weight].to_numpy(copy=True)[i]

                result["weights"] = torch.tensor(weights_merged, dtype=torch.float32)
            else:
                result["weights"] = torch.tensor(
                    # PyTorch wants writeable arrays
                    data[self.weight].to_numpy(copy=True),
                    dtype=torch.float32,
                )

        return result

    def get_metadata(self) -> TimeSeriesMetadata:
        """Return metadata about the dataset.

        Returns
        -------
        TimeSeriesMetadata
            Schema containing:
            - cols: column names for y, x, and static features
            - col_type: mapping of columns to their types (F/C)
            - col_known: mapping of columns to their future known status (K/U)
            - is_prediction: whether the data are predictions
        """
        return self.metadata

    def to_pandas(self) -> pd.DataFrame:
        """Return the data as a single data frame.

        Returns
        -------
        pd.DataFrame
            ``data``, with ``data_future`` appended if it was passed.
        """
        if self.data_future is None:
            return self.data
        return pd.concat([self.data, self.data_future], ignore_index=True)

    @classmethod
    def from_tensors(
        cls,
        tensors: dict[str, torch.Tensor],
        metadata: TimeSeriesMetadata | dict | None = None,
        group_starts: list[int] | None = None,
        groups: np.ndarray | list | None = None,
    ) -> "TimeSeries":
        """Construct a ``TimeSeries`` of predictions from model output tensors.

        Parameters
        ----------
        tensors : dict of str to torch.Tensor
            must contain ``"y"``, of shape ``(n, n_targets)`` or ``(n,)``.
            May contain ``"t"``, of shape ``(n,)``; defaults to ``arange(n)``.
        metadata : TimeSeriesMetadata or dict, optional
            schema the tensors were produced against; only ``cols["y"]`` is used,
            to name the target columns. Without
            usable names, targets are called ``y0``, ``y1``, ...
        group_starts : list of int, optional
            row offsets at which each series begins.
        groups : np.ndarray or list, optional
            per-row series labels. Takes precedence over ``group_starts``.
            With neither, all rows are one series.

        Returns
        -------
        TimeSeries
            with columns ``_series``, ``_time_idx`` and one per target, and
            ``metadata.is_prediction`` set to ``True``.

        Raises
        ------
        ValueError
            If ``"y"`` is missing, is not 1- or 2-dimensional, or if ``t`` or
            ``groups`` does not have one entry per row of ``y``.
        """
        if "y" not in tensors:
            raise ValueError(
                '`tensors` must contain "y", the predicted values. Got keys: '
                f"{sorted(tensors)}."
            )

        y = tensors["y"]
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        elif y.ndim != 2:
            raise ValueError(
                "`tensors['y']` must have shape (n,) or (n, n_targets), got "
                f"shape {tuple(y.shape)}. Reshape it first, e.g. "
                "y.reshape(-1, 1)."
            )
        y = y.detach().cpu().numpy()
        n_rows, n_targets = y.shape

        for name, value in (("t", tensors.get("t")), ("groups", groups)):
            if value is not None and len(value) != n_rows:
                raise ValueError(
                    f"`{name}` must have one entry per row of `y` ({n_rows}), "
                    f"got {len(value)}."
                )

        target = list(metadata["cols"]["y"]) if metadata is not None else []
        if len(target) != n_targets:
            target = [f"y{i}" for i in range(n_targets)]

        t = tensors.get("t")
        if t is None:
            t = np.arange(n_rows)
        else:
            t = t.detach().cpu().numpy().reshape(-1)

        if groups is not None:
            series = np.asarray(groups).reshape(-1)
        elif group_starts is not None:
            series = np.zeros(n_rows, dtype=int)
            series[np.asarray(group_starts, dtype=int)[1:]] = 1
            series = np.cumsum(series)
        else:
            series = np.zeros(n_rows, dtype=int)

        df = pd.DataFrame({"_series": series, "_time_idx": t})
        for i, col in enumerate(target):
            df[col] = y[:, i]

        obj = cls(df, time="_time_idx", target=target, group=["_series"])
        obj.metadata = replace(obj.metadata, is_prediction=True)
        return obj
