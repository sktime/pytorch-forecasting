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


@dataclass(frozen=True)
class EncDecDataModuleMetadata(_DictLike):
    """Shapes an encoder-decoder model needs in order to be constructed.

    Parameters
    ----------
    encoder_cat : int
        Number of categorical variables in the encoder.
    encoder_cont : int
        Number of continuous variables in the encoder.
    decoder_cat : int
        Number of categorical variables known in advance to the decoder.
    decoder_cont : int
        Number of continuous variables known in advance to the decoder.
    target : int
        Number of target variables.
    static_categorical_features : int, default=0
        Number of static categorical features.
    static_continuous_features : int, default=0
        Number of static continuous features.
    max_encoder_length : int, default=0
        Maximum encoder length.
    max_prediction_length : int, default=0
        Maximum prediction length.
    min_encoder_length : int, default=0
        Minimum encoder length.
    min_prediction_length : int, default=0
        Minimum prediction length.
    """

    encoder_cat: int
    encoder_cont: int
    decoder_cat: int
    decoder_cont: int
    target: int
    static_categorical_features: int = 0
    static_continuous_features: int = 0
    max_encoder_length: int = 0
    max_prediction_length: int = 0
    min_encoder_length: int = 0
    min_prediction_length: int = 0


@dataclass(frozen=True)
class TslibDataModuleMetadata(_DictLike):
    """Shapes a tslib-style model needs in order to be constructed.

    Parameters
    ----------
    feature_names : dict
        Maps a feature group - ``"categorical"``, ``"continuous"``,
        ``"static"``, ``"known"``, ``"unknown"``, ``"target"``, ``"all"`` - to
        its column names.
    feature_indices : dict
        Maps the same groups, except ``"all"``, to their positions within the
        feature tensor.
    n_features : dict
        Number of features in each group of ``feature_names``.
    context_length : int
        Length of the context window.
    prediction_length : int
        Length of the prediction window.
    freq : str or None, default=None
        Frequency of the time series.
    features : str, default="MS"
        Feature combination mode, one of ``"S"``, ``"M"``, ``"MS"``.
    """

    feature_names: dict[str, list[str]]
    feature_indices: dict[str, list[int]]
    n_features: dict[str, int]
    context_length: int
    prediction_length: int
    freq: str | None = None
    features: str = "MS"
