"""Typed schemas of the data layer."""

from dataclasses import dataclass, fields


class _DictLike:
    """Mapping protocol for the metadata dataclasses."""

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key) if key in self else default

    def keys(self):
        return [f.name for f in fields(self)]

    def __contains__(self, key):
        return key in self.keys()

    def __iter__(self):
        return iter(self.keys())


@dataclass(frozen=True)
class TimeSeriesMetadata(_DictLike):
    """Schema of a :class:`~pytorch_forecasting.data.TimeSeries`.

    Parameters
    ----------
    cols : dict
        ``{"y": [...], "x": [...], "st": [...]}``, names of the target, feature
        and static columns. List order is the column order of the corresponding
        tensor dimension, and must not be reordered.
    col_type : dict
        maps column name to ``"F"`` (numerical) or ``"C"`` (categorical).
    col_known : dict
        maps column name to ``"K"`` (known in the future) or ``"U"`` (unknown).
    is_prediction : bool, default=False
        whether the described object holds predictions rather than input data.
    """

    cols: dict[str, list[str]]
    col_type: dict[str, str]
    col_known: dict[str, str]
    is_prediction: bool = False

    def __post_init__(self):
        """Validation checks.

        Raises
        ------
        ValueError
            If ``cols`` does not have exactly the keys ``"y"``, ``"x"``, ``"st"``.
        """
        missing = {"y", "x", "st"} - set(self.cols)
        if missing:
            raise ValueError(
                f"`cols` is missing required keys: {sorted(missing)}. It must "
                'map "y", "x" and "st" to lists of column names.'
            )
