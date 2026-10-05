"""
``TimeSeries`` datatype - prototype for the v2 redesign.
"""

from dataclasses import dataclass, field, fields, replace
from typing import Any
from warnings import warn

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting.utils._coerce import _coerce_to_list

__all__ = ["TimeSeriesMetadata", "TimeSeries_datatype"]


@dataclass(frozen=True)
class TimeSeriesMetadata:
    """Schema of a :class:`TimeSeries_datatype`.

    Same content as the ``metadata`` dict of ``TimeSeries``, as a dataclass.
    Mapping-style access is kept so that code written against the dict version
    keeps working.

    Parameters
    ----------
    cols : dict { 'y': list[str], 'x': list[str], 'st': list[str] }
        Names of columns for y, x, and static features, in the same order as
        the column dimensions of the corresponding tensors.
    col_type : dict[str, str]
        Maps column names to "F" (numerical) or "C" (categorical).
    col_known : dict[str, str]
        Maps column names to "K" (future known) or "U" (future unknown).
    is_prediction : bool, default=False
        Whether the object holds predictions. Predictions have no static
        features and no known/unknown split.
    """

    cols: dict[str, list[str]]
    col_type: dict[str, str] = field(default_factory=dict)
    col_known: dict[str, str] = field(default_factory=dict)
    is_prediction: bool = False

    def __getitem__(self, key: str) -> Any:
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self.keys():
            return default
        return getattr(self, key)

    def keys(self) -> list[str]:
        return [f.name for f in fields(self)]

    def __contains__(self, key: object) -> bool:
        return key in self.keys()

    def __iter__(self):
        return iter(self.keys())


class TimeSeries_datatype:
    """Container for time series data stored in a pandas DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        data frame with sequence data.
        Column names must all be str, and contain str as referred to below.
    data_future : pd.DataFrame, optional, default=None
        data frame with future data.
        May contain only columns that are in time, group, weight, known,
        or static.
    time : str, optional, default = first col not in group_ids, weight, target,
        static.
        integer typed column denoting the time index within ``data``.
    target : str or List[str], optional, default = last column (at iloc -1)
        column(s) in ``data`` denoting the forecasting target.
    group : List[str], optional, default = None
        list of column names identifying a time series instance within ``data``.
    weight : str, optional, default=None
        column name for weights.
    num : list of str, optional, default = all columns with dtype in "fi"
        list of numerical variables in ``data``.
    cat : list of str, optional, default = all columns with dtype in "Obc"
        list of categorical variables in ``data``.
    known : list of str, optional, default = all variables
        list of variables that change over time and are known in the future.
    unknown : list of str, optional, default = no variables
        list of variables that are not known in the future.
    static : list of str, optional, default = all variables not in known, unknown
        list of variables that do not change over time.
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
            "TimeSeries_datatype is a prototype of the reworked "
            "pytorch-forecasting data layer, for design testing only. "
            "It is not part of the public API and may change or disappear "
            "without warning. Feedback and suggestions are very welcome in "
            "pytorch-forecasting issue 1736, "
            "https://github.com/sktime/pytorch-forecasting/issues/1736",
            UserWarning,
        )

        # handle defaults, coercion, and derived attributes
        self._target = _coerce_to_list(target)
        self._group = _coerce_to_list(group)
        self._num = _coerce_to_list(num)
        self._cat = _coerce_to_list(cat)
        self._known = _coerce_to_list(known)
        self._unknown = _coerce_to_list(unknown)
        self._static = _coerce_to_list(static)

        self._infer_columns()

        self.feature_cols = [
            col
            for col in data.columns
            if col not in [self.time] + self._group + [self.weight] + self._target
        ]
        if self._group:
            group_arg = (
                self._group[0]
                if isinstance(self._group, (list, tuple)) and len(self._group) == 1
                else self._group
            )
            self._groups = self.data.groupby(group_arg).groups
            self._group_ids = list(self._groups.keys())
        else:
            self._groups = {"_single_group": self.data.index}
            self._group_ids = ["_single_group"]
        # create mapping from group id to index for efficient lookup
        self._group_to_idx = {gid: i for i, gid in enumerate(self._group_ids)}

        self._prepare_metadata()

        # overwrite __init__ params for upwards compatibility with AS PRs
        self.group = self._group
        self.target = self._target
        self.num = self._num
        self.cat = self._cat
        self.known = self._known
        self.unknown = self._unknown
        self.static = self._static

    def _infer_columns(self):
        """Fill in ``time``, ``target``, ``num`` and ``cat`` from the data frame.

        Applies the defaults promised by the docstring: target is the last
        column, time is the first column not otherwise spoken for, and the
        num/cat split follows the column dtypes.
        """
        cols = list(self.data.columns)

        if not self._target:
            self._target = [cols[-1]]

        if self.time is None:
            taken = set(self._group) | set(self._target) | set(self._static)
            taken.add(self.weight)
            self.time = next((col for col in cols if col not in taken), None)
            if self.time is None:
                raise ValueError(
                    "no column is left to use as the time index: every column "
                    f"of `data` ({cols}) is already used as group, target, "
                    "static or weight. Pass `time` explicitly."
                )

        index_cols = set(self._group) | {self.time, self.weight}
        typed = [col for col in cols if col not in index_cols]

        if not self._num and not self._cat:
            self._num = [c for c in typed if self.data[c].dtype.kind in "fi"]
            self._cat = [c for c in typed if self.data[c].dtype.kind in "Obc"]

    def _prepare_metadata(self):
        """Prepare metadata for the dataset.

        Same content as ``TimeSeries._prepare_metadata``, but stored as a
        :class:`TimeSeriesMetadata` instead of a dict.
        """
        col_type = {}
        col_known = {}

        for col in self._target + self.feature_cols + self._static:
            col_type[col] = "C" if col in self._cat else "F"
            col_known[col] = "K" if col in self._known else "U"

        self.metadata = TimeSeriesMetadata(
            cols={
                "y": self._target,
                "x": self.feature_cols,
                "st": self._static,
            },
            col_type=col_type,
            col_known=col_known,
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
        time = self.time
        feature_cols = self.feature_cols
        _target = self._target
        _known = self._known
        _static = self._static
        _group = self._group
        _groups = self._groups
        _group_ids = self._group_ids
        weight = self.weight
        data_future = self.data_future

        group_id = _group_ids[index]

        if _group:
            mask = _groups[group_id]
            data = self.data.loc[mask]
        else:
            data = self.data

        cutoff_time = data[time].max()

        # PyTorch wants writeable arrays
        data_vals = data[time].to_numpy(copy=True)
        data_tgt_vals = data[_target].to_numpy(copy=True)
        data_feat_vals = data[feature_cols].to_numpy(copy=True)

        result = {
            "t": data_vals,
            "y": torch.tensor(data_tgt_vals),
            "x": torch.tensor(data_feat_vals),
            "group": torch.tensor([self._group_to_idx[group_id]], dtype=torch.long),
            # PyTorch wants writeable arrays
            "st": torch.tensor(
                data[_static].iloc[0].to_numpy(copy=True) if _static else []
            ),
            "cutoff_time": cutoff_time,
        }

        if data_future is not None:
            if _group:
                group_arg = (
                    self._group[0]
                    if isinstance(self._group, (list, tuple)) and len(self._group) == 1
                    else self._group
                )
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
            y_merged = np.full((num_timepoints, len(_target)), np.nan)

            current_time_indices = {t: i for i, t in enumerate(combined_times)}
            for i, t in enumerate(data_vals):
                idx = current_time_indices[t]
                x_merged[idx] = data_feat_vals[i]
                y_merged[idx] = data_tgt_vals[i]

            for i, t in enumerate(data_fut_vals):
                if t in current_time_indices:
                    idx = current_time_indices[t]
                    for j, col in enumerate(_known):
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
            Dataclass containing:
            - cols: column names for y, x, and static features
            - col_type: mapping of columns to their types (F/C)
            - col_known: mapping of columns to their future known status (K/U)
        """
        return self.metadata

    def to_pandas(self) -> pd.DataFrame:
        """Return the data as a data frame, for users and the sktime adapter.

        Returns
        -------
        pd.DataFrame
            ``data``, with ``data_future`` appended if present.
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
    ) -> "TimeSeries_datatype":
        """Create a ``TimeSeries_datatype`` from tensors, e.g. model predictions.

        Predictions are already tensors and carry no static features and no
        known/unknown split, so the resulting metadata has ``is_prediction=True``
        and empty ``st``.

        Parameters
        ----------
        tensors : dict of str to torch.Tensor
            Must contain ``"y"``, of shape (n_timepoints, n_targets) or
            (n_timepoints,). May contain ``"t"`` of shape (n_timepoints,).
        metadata : TimeSeriesMetadata or dict, optional
            Metadata of the data the predictions were made for, either a
            ``TimeSeriesMetadata`` or the dict of the D1 ``TimeSeries``. Only
            the target names are taken from it; if not given, targets are
            named ``y0, y1, ...``.
        group_starts : list of int, optional
            Row offsets at which the individual series start. Defaults to
            ``[0]``, i.e. a single series. Ignored if ``groups`` is given.
        groups : array-like of shape (n_timepoints,), optional
            Group label of each row, for callers that know which series each
            prediction belongs to. Takes precedence over ``group_starts``.

        Returns
        -------
        TimeSeries_datatype
            with ``metadata.is_prediction`` set to True.
        """
        y = torch.as_tensor(tensors["y"]).detach().cpu().float()
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        n_rows = y.shape[0]

        if metadata is not None:
            # mapping access, so that both TimeSeriesMetadata and the plain
            # dict returned by the D1 ``TimeSeries`` work here
            target = list(metadata["cols"]["y"])
        else:
            target = [f"y{i}" for i in range(y.shape[-1])]

        t = tensors.get("t")
        t = np.arange(n_rows) if t is None else torch.as_tensor(t).cpu().numpy()

        if groups is None:
            starts = [0] if group_starts is None else [int(s) for s in group_starts]
            lengths = [b - a for a, b in zip(starts, starts[1:] + [n_rows])]
            groups = np.repeat(np.arange(len(starts)), lengths)
        else:
            groups = np.asarray(groups)

        df = pd.DataFrame(y.numpy(), columns=target)
        df.insert(0, "_series", groups)
        df.insert(1, "_time_idx", t)

        obj = cls(df, time="_time_idx", target=target, group=["_series"])
        obj.metadata = replace(obj.metadata, is_prediction=True)
        return obj
