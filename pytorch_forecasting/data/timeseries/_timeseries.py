"""
Timeseries datasets.

Timeseries data is special and has to be processed and passed in a special way.
This module defines TimeSeriesDataSet,
a class that is able to handle a wide variety of timeseries data problems.
"""

from copy import copy as _copy, deepcopy
from functools import lru_cache
import inspect
from typing import Any, Callable, Optional, TypeVar, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted
import torch
from torch.distributions import Beta
from torch.nn.utils import rnn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler, SequentialSampler

from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    GroupNormalizer,
    MultiNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)
from pytorch_forecasting.data.samplers import TimeSynchronizedBatchSampler
from pytorch_forecasting.utils import repr_class
from pytorch_forecasting.utils._coerce import _coerce_to_dict, _coerce_to_list
from pytorch_forecasting.utils._dependencies import _check_matplotlib


def _find_end_indices(
    diffs: np.ndarray, max_lengths: np.ndarray, min_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identify end indices in series even if some values are missing.

    Parameters
    ----------
    diffs : np.ndarray
        array of differences to next time step. nans should be filled up with ones
    max_lengths : np.ndarray
        maximum length of sequence by position.
    min_length : int
        minimum length of sequence.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        tuple of arrays where first is end indices and second is list of start
        and end indices that are currently missing.
    """
    missing_start_ends = []
    end_indices = []
    length = 1
    start_idx = 0
    max_idx = len(diffs) - 1
    max_length = max_lengths[start_idx]

    for idx, diff in enumerate(diffs):
        if length >= max_length:
            while length >= max_length:
                if length == max_length:
                    end_indices.append(idx)
                else:
                    end_indices.append(idx - 1)
                length -= diffs[start_idx]
                if start_idx < max_idx:
                    start_idx += 1
                max_length = max_lengths[start_idx]
        elif length >= min_length:
            missing_start_ends.append([start_idx, idx])
        length += diff
    if len(missing_start_ends) > 0:  # required for numba compliance
        return np.asarray(end_indices), np.asarray(missing_start_ends)
    else:
        return np.asarray(end_indices), np.empty((0, 2), dtype=np.int64)


try:
    import numba

    _find_end_indices = numba.jit(nopython=True)(_find_end_indices)
except ImportError:
    pass


def check_for_nonfinite(
    tensor: torch.Tensor, names: Union[str, list[str]]
) -> torch.Tensor:
    """Check if tensor contains NAs or infinite values and has correct dimension.

    Checks:

    * whether tensor is finite, otherwise raises ValueError
    * checks whether dimension of tensor is correct. If tensor is a str,
      tensor.ndim has to be 1, and if tensor is a list, tensor.ndim has to be 2.
      Otherwise raises AssertionError.

    Parameters
    ----------
    names : str or list of str
        name(s) of column(s) to check
    tensor : torch.Tensor
        tensor to check

    Returns
    -------
    torch.Tensor
        returns tensor unchanged, if checks yield no issues

    Raises
    ------
    ValueError
        if tensor contains NAs or infinite values
    AssertionError
        if tensor has incorrect dimension, see above
    """
    if isinstance(names, str):
        names = [names]
        assert tensor.ndim == 1, names
        nans = (~torch.isfinite(tensor).unsqueeze(-1)).sum(0)
    else:
        assert tensor.ndim == 2, names
        nans = (~torch.isfinite(tensor)).sum(0)
    for name, na in zip(names, nans):
        if na > 0:
            raise ValueError(
                f"{na} ({na / tensor.size(0):.2%}) of {name} "
                "values were found to be NA or infinite (even after encoding). "
                "NA values are not allowed "
                "`allow_missing_timesteps` refers to missing rows, not to missing "
                "values. Possible strategies to "
                f"fix the issue are (a) dropping the variable {name}, "
                "(b) using `NaNLabelEncoder(add_nan=True)` for categorical variables, "
                "(c) filling missing values and/or (d) optionally adding a variable "
                "indicating filled values"
            )
    return tensor


NORMALIZER = Union[TorchNormalizer, NaNLabelEncoder, EncoderNormalizer]

Columns = list[str]
TargetType = list[str, str]
TargetPositive = list[str, bool]
TargetSkew = list[str, float]

DataProperties = dict[str, Union[Columns, TargetType, TargetPositive, TargetSkew]]
TimeSeriesDataType = TypeVar("TimeSeriesType", bound="TimeSeriesDataSet")


class TimeSeriesDataSet(Dataset):
    """PyTorch Dataset for fitting timeseries models.

    The dataset automates common tasks such as

    * scaling and encoding of variables
    * normalizing the target variable
    * efficiently converting timeseries in pandas dataframes to torch tensors
    * holding information about static and time-varying variables known and unknown in
      the future
    * holding information about related categories (such as holidays)
    * downsampling for data augmentation
    * generating inference, validation and test datasets

    The :ref:`tutorial on passing data to models <passing-data>` is helpful to
    understand the output of the dataset
    and how it is coupled to models.

    Each sample is a subsequence of a full time series. The subsequence consists of
    encoder and decoder/prediction
    timepoints for a given time series. This class constructs an index which defined
    which subsequences exists and
    can be samples from (``index`` attribute). The samples in the index are defined
    by the various parameters.
    to the class (encoder and prediction lengths, minimum prediction length, randomize
    length and predict keywords).
    How samples are
    sampled into batches for training, is determined by the DataLoader.
    The class provides the
    :py:meth:`~TimeSeriesDataSet.to_dataloader` method
    to convert the dataset into a dataloader.

    Large datasets:

    Currently the class is limited to in-memory operations (that can be sped up by an
    existing installation of `numba <https://pypi.org/project/numba/>`_).
    If you have extremely large data,
    however, you can pass prefitted encoders and and scalers to it and a subset of
    sequences to the class to
    construct a valid dataset (plus, likely the EncoderNormalizer should be used to
    normalize targets).
    when fitting a network, you would then to create a custom DataLoader that rotates
    through the datasets.
    There are currently no in-built methods to do this.

    Parameters
    ----------
    data : pd.DataFrame
        dataframe with sequence data - each row can be identified with
        ``time_idx`` and the ``group_ids``

    time_idx : str
        integer typed column denoting the time index within ``data``.
        This columns is used to determine the sequence of samples.
        If there are no missings observations,
        the time index should increase by ``+1`` for each subsequent sample.
        The first time_idx for each series does not necessarily
        have to be ``0`` but any value is allowed.

    target : Union[str, list[str]]
        column(s) in ``data`` denoting the forecasting target.
        Can be categorical or continous dtype.

    group_ids : list[str]
        list of column names identifying a time series instance within ``data``
        This means that the ``group_ids``
        identify a sample together with the ``time_idx``.
        If you have only one timeseries, set this to the
        name of column that is constant.

    weight : str, optional, default=None
        column name for weights. Defaults to None.

    max_encoder_length : int, optional, default=30
        maximum length to encode.
        This is the maximum history length used by the time series dataset.

    min_encoder_length : int, optional, default=max_encoder_length
        minimum allowed length to encode. Defaults to max_encoder_length.

    min_prediction_idx : int, optional, default = first time_idx in data
        minimum ``time_idx`` from where to start predictions.
        This parameter can be useful to create a validation or test set.

    max_prediction_length : int, optional, default=1
        maximum prediction/decoder length
        (choose this not too short as it can help convergence)

    min_prediction_length : int, optional, default=max_prediction_length
        minimum prediction/decoder length

    static_categoricals : list of str, optional, default=None
        list of categorical variables that do not change over time, in ``data``,
        entries can be also lists which are then encoded together
        (e.g. useful for product categories)

    static_reals : list of str, optional, default=None
        list of continuous variables that do not change over time

    time_varying_known_categoricals : list of str, optional, default=None
        list of categorical variables that change over time and are known in the future,
        entries can be also lists which are then encoded together
        (e.g. useful for special days or promotion categories)

    time_varying_known_reals : list of str, optional, default=None
        list of continuous variables that change over time and are known in the future
        (e.g. price of a product, but not demand of a product)

    time_varying_unknown_categoricals : list of str, optional, default=None
        list of categorical variables that are not known in the future
        and change over time.
        entries can be also lists which are then encoded together
        (e.g. useful for weather categories).
        Target variables should be included here, if categorical.

    time_varying_unknown_reals : list of str, optional, default=None
        list of continuous variables that are not known in the future
        and change over time.
        Target variables should be included here, if real.

    variable_groups : Dict[str, list[str]], optional, default=None
        dictionary mapping a name to a list of columns in the data.
        The name should be present
        in a categorical or real class argument, to be able to encode or scale the
        columns by group.
        This will effectively combine categorical variables is particularly useful
        if a categorical variable can have multiple values at the same time.
        An example are holidays which can be overlapping.

    constant_fill_strategy : dict, optional, default=None
        Keys must be str, values can be str, float, int or bool.
        Dictionary of column names with constants to fill in missing values if there
        are gaps in the sequence (by default forward fill strategy is used).
        The values will be only used if ``allow_missing_timesteps=True``.
        A common use case is to denote that demand was 0 if the sample is not in the
        dataset.

    allow_missing_timesteps : bool, optional, default=False
        whether to allow missing timesteps that are automatically filled up.
        Missing values refer to gaps in the ``time_idx``, e.g. if a specific
        timeseries has only samples for 1, 2, 4, 5, the sample for 3 will be
        generated on-the-fly.
        Allow missings does not deal with ``NA`` values. You should fill NA values
        before passing the dataframe to the TimeSeriesDataSet.

    lags : dict[str, list[int]], optional, default=None
        dictionary of variable names mapped to list of time steps by which the
        variable should be lagged.
        Lags can be useful to indicate seasonality to the models.
        Useful to add if seasonalit(ies) of the data are known.,
        In this case, it is recommended to add the target variables
        with the corresponding lags to improve performance.
        Lags must be at not larger than the shortest time series as all time series
        will be cut by the largest lag value to prevent NA values.
        A lagged variable has to appear in the time-varying variables.
        If you only want the lagged but not the current value, lag it manually in
        your input data using
        ``data[lagged_varname] = ``
        ``data.sort_values(time_idx).groupby(group_ids, observed=True).shift(lag)``.

    add_relative_time_idx : bool, optional, default=False
        whether to add a relative time index as feature, i.e.,
        for each sampled sequence, the index will range from -encoder_length to
        prediction_length.

    add_target_scales : bool, optional, default=False
        whether to add scales for target to static real features, i.e., add the
        center and scale of the unnormalized timeseries as features.

    add_encoder_length : Union[bool, str], optional, default="auto"
        whether to add encoder length to list of static real variables.
        Defaults to "auto", iwhich is same as
        ``True`` iff ``min_encoder_length != max_encoder_length``.

    target_normalizer : torch transformer, str, list, tuple, optional, default="auto"
        Transformer that takes group_ids, target and time_idx to normalize targets.
        You can choose from
        :py:class:`~pytorch_forecasting.data.encoders.TorchNormalizer`,
        :py:class:`~pytorch_forecasting.data.encoders.GroupNormalizer`,
        :py:class:`~pytorch_forecasting.data.encoders.NaNLabelEncoder`,
        :py:class:`~pytorch_forecasting.data.encoders.EncoderNormalizer`
        (on which overfitting tests will fail)
        or ``None`` for using no normalizer. For multiple targets, use a
        :py:class`~pytorch_forecasting.data.encoders.MultiNormalizer`.
        By default an appropriate normalizer is chosen automatically.

    categorical_encoders : dict[str, BaseEstimator]
        dictionary of scikit learn label transformers.
        If you have unobserved categories in
        the future  / a cold-start problem, you can use the
        :py:class:`~pytorch_forecasting.data.encoders.NaNLabelEncoder` with
        ``add_nan=True``.
        Defaults effectively to sklearn's ``LabelEncoder()``.
        Prefitted encoders will not be fit again.

    scalers : optional, dict with str keys and torch or sklearn scalers as values
        dictionary of scikit-learn or torch scalers.
        Defaults to sklearn's ``StandardScaler()``.
        Other options
        are :py:class:`~pytorch_forecasting.data.encoders.EncoderNormalizer`,
        :py:class:`~pytorch_forecasting.data.encoders.GroupNormalizer`
        or scikit-learn's ``StandarScaler()``,
        ``RobustScaler()`` or ``None`` for using no normalizer / normalizer
        with ``center=0`` and ``scale=1``
        (``method="identity"``).
        Prefittet encoders will not be fit again (with the exception of the
        :py:class:`~pytorch_forecasting.data.encoders.EncoderNormalizer` that is
        fit on every encoder sequence).

    randomize_length : optional, None, bool, or tuple of float.
        None or False if not to randomize lengths.
        Tuple of beta distribution concentrations from which
        probabilities are sampled that are used to sample new sequence lengths
        with a binomial distribution.
        If True, defaults to (0.2, 0.05), i.e. ~1/4 of samples
        around minimum encoder length.
        Defaults to False otherwise.

    predict_mode : bool
        If True, the TimeSeriesDataSet will only create one sequence
        per time series (i.e. only from the latest provided samples).
        Effectively, this will select each time series identified by ``group_ids``
        the last ``max_prediction_length`` samples of each time series as
        prediction samples and everthing previous up to ``max_encoder_length``
        samples as encoder samples.
        If False, the TimeSeriesDataSet will create subsequences by sliding a
        window over the data samples.
        For training use cases, it's preferable to set predict_mode=False
        to get all subseries.
        On the other hand, predict_mode = True is ideal for validation cases.
    """

    # todo: refactor:
    # - creating base class with minimal functionality
    # - "outsource" transformations -> use pytorch transformations as default

    # todo: integrate graphs
    # - add option to pass networkx graph to the dataset -> clearly defined
    # - create method to create networkx graph for hierachies -> clearly defined
    # - convert networkx graph to pytorch geometric graph
    # - create sampler to sample from the graph
    # - create option in `to_dataloader` method to use a graph sampler
    #     -> automatically changing collate function which returns graphs
    #     -> should incorporate entire dataset but be compatible with current approach
    # - integrate hierachical loss somehow into loss metrics

    # how to get there:
    # - add networkx and pytorch_geometric to requirements BUT as extras
    #     -> do we also need torch_sparse, etc.? -> can we avoid this? probably not
    # - networkx graph: define what makes sense from user perspective
    # - define conversion into pytorch geometric graph? is this a two-step process of
    #     - encoding networkx graph and converting it into "unfilled" pytorch geometric
    #       graph
    #     - then creating full graph in collate function on the fly?
    #     - or is data already stored in pytorch geometric graph, only cut through it?
    #     - dataformat would change? Is is all timeseries data? + mask when valid?
    #     - then making cuts through the graph in sampling?
    #     - would it be best in this case to re-think the timeseries class and design it
    #       as series of transformations?
    #     - what is the new master data? very off current state or very similar?
    #     - current approach is storing data in long format which is memory efficient
    #       and using the index object to
    #       make sense of it when accessing. graphs would require wide format?
    # - do NOT overengineer, i.e. support only usecase of single static graph,
    #   but only subset might be relevant
    #     -> however, should think what happens if we want a dynamic graph. would this
    #        completely change the
    #        data format?

    # decisions:
    # - stay with long format and create graph on the fly even if hampering
    #   efficiency and performance
    # - go with pytorch_geometric approach for future proofing
    # - directly convert networkx into pytorch_geometric graph
    # - sampling: support only time-synchronized.
    #     - sample randomly an instance from index as now.
    #     - then get additional samples as per graph (that has been created) and
    #       available data
    #     - then collate into graph object

    def __init__(
        self,
        data: pd.DataFrame,
        time_idx: str,
        target: Union[str, list[str]],
        group_ids: list[str],
        weight: Union[str, None] = None,
        max_encoder_length: int = 30,
        min_encoder_length: int = None,
        min_prediction_idx: int = None,
        min_prediction_length: int = None,
        max_prediction_length: int = 1,
        static_categoricals: Optional[list[str]] = None,
        static_reals: Optional[list[str]] = None,
        time_varying_known_categoricals: Optional[list[str]] = None,
        time_varying_known_reals: Optional[list[str]] = None,
        time_varying_unknown_categoricals: Optional[list[str]] = None,
        time_varying_unknown_reals: Optional[list[str]] = None,
        variable_groups: Optional[dict[str, list[int]]] = None,
        constant_fill_strategy: Optional[
            dict[str, Union[str, float, int, bool]]
        ] = None,
        allow_missing_timesteps: bool = False,
        lags: Optional[dict[str, list[int]]] = None,
        add_relative_time_idx: bool = False,
        add_target_scales: bool = False,
        add_encoder_length: Union[bool, str] = "auto",
        target_normalizer: Union[
            NORMALIZER, str, list[NORMALIZER], tuple[NORMALIZER], None
        ] = "auto",
        categorical_encoders: Optional[dict[str, NaNLabelEncoder]] = None,
        scalers: Optional[
            dict[
                str,
                Union[StandardScaler, RobustScaler, TorchNormalizer, EncoderNormalizer],
            ]
        ] = None,
        randomize_length: Union[None, tuple[float, float], bool] = False,
        predict_mode: bool = False,
    ):
        """Timeseries dataset holding data for models."""
        super().__init__()

        # write variables to self and handle defaults
        # -------------------------------------------
        self.max_encoder_length = max_encoder_length
        if min_encoder_length is None:
            min_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        self.max_prediction_length = max_prediction_length
        if min_prediction_length is None:
            min_prediction_length = max_prediction_length
        self.min_prediction_length = min_prediction_length

        self.target = target
        self.weight = weight
        self.time_idx = time_idx
        self.group_ids = _coerce_to_list(group_ids)

        self.static_categoricals = static_categoricals
        self._static_categoricals = _coerce_to_list(static_categoricals)

        self.static_reals = static_reals
        self._static_reals = _coerce_to_list(static_reals)

        self.time_varying_known_categoricals = time_varying_known_categoricals
        self._time_varying_known_categoricals = _coerce_to_list(
            time_varying_known_categoricals
        )

        self.time_varying_known_reals = time_varying_known_reals
        self._time_varying_known_reals = _coerce_to_list(time_varying_known_reals)

        self.time_varying_unknown_categoricals = time_varying_unknown_categoricals
        self._time_varying_unknown_categoricals = _coerce_to_list(
            time_varying_unknown_categoricals
        )

        self.time_varying_unknown_reals = time_varying_unknown_reals
        self._time_varying_unknown_reals = _coerce_to_list(time_varying_unknown_reals)

        self.add_relative_time_idx = add_relative_time_idx

        # set automatic defaults
        if isinstance(randomize_length, bool):
            if not randomize_length:
                randomize_length = None
            else:
                randomize_length = (0.2, 0.05)

        self.randomize_length = randomize_length

        if min_prediction_idx is None:
            min_prediction_idx = data[self.time_idx].min()
        self.min_prediction_idx = min_prediction_idx

        self.constant_fill_strategy = constant_fill_strategy
        self._constant_fill_strategy = _coerce_to_dict(constant_fill_strategy)

        self.predict_mode = predict_mode
        self.allow_missing_timesteps = allow_missing_timesteps
        self.target_normalizer = target_normalizer

        self.categorical_encoders = categorical_encoders
        self._categorical_encoders = _coerce_to_dict(categorical_encoders)

        self.scalers = scalers
        self._scalers = _coerce_to_dict(scalers)

        self.add_target_scales = add_target_scales
        self.variable_groups = variable_groups
        self._variable_groups = _coerce_to_dict(variable_groups)

        self.lags = lags
        self._lags = _coerce_to_dict(lags)

        # add_encoder_length
        if isinstance(add_encoder_length, str):
            msg = (
                f"Only 'auto' allowed for add_encoder_length "
                f"but found {add_encoder_length}"
            )
            assert add_encoder_length == "auto", msg
            add_encoder_length = self.min_encoder_length != self.max_encoder_length
        self.add_encoder_length = add_encoder_length

        # overwrite values
        self.reset_overwrite_values()

        # check parameters
        self._check_params()

        # data preprocessing in pandas
        # ----------------------------

        # get metadata from data
        self._data_properties = self._data_properties(data)

        # target normalizer
        self.target_normalizer = self._set_target_normalizer(
            self._data_properties, self.target_normalizer
        )

        # add time index relative to prediction position
        if self.add_relative_time_idx:
            assert (
                "relative_time_idx" not in data.columns
            ), "relative_time_idx is a protected column and must not be present in data"
            if (
                "relative_time_idx" not in self._time_varying_known_reals
                and "relative_time_idx" not in self.reals
            ):
                self._time_varying_known_reals.append("relative_time_idx")

        # add decoder length to static real variables
        if self.add_encoder_length:
            assert (
                "encoder_length" not in data.columns
            ), "encoder_length is a protected column and must not be present in data"
            if (
                "encoder_length" not in self._time_varying_known_reals
                and "encoder_length" not in self.reals
            ):
                self._static_reals.append("encoder_length")

        # add columns for additional features
        if self.add_relative_time_idx or self.add_encoder_length:
            data = data.copy()  # only copies indices (underlying data is NOT copied)
        if self.add_relative_time_idx:
            data.loc[:, "relative_time_idx"] = (
                0.0  # dummy - real value will be set dynamically in __getitem__()
            )
        if self.add_encoder_length:
            data.loc[:, "encoder_length"] = (
                0  # dummy - real value will be set dynamically in __getitem__()
            )

        # validate
        self._validate_data(data)

        # add lags
        if len(self._lags) > 0:
            self._set_lagged_variables()

        # filter data
        if min_prediction_idx is not None:
            # filtering for min_prediction_idx will be done on subsequence level,
            # ensuring that minimal decoder index is always >= min_prediction_idx
            data = data[
                lambda x: x[self.time_idx]
                >= self.min_prediction_idx - self.max_encoder_length - self.max_lag
            ]
        data = data.sort_values(self.group_ids + [self.time_idx])

        # preprocess data
        data = self._preprocess_data(data)

        msg = "Target normalizer is separate and not in scalers."
        for target in self.target_names:
            assert target not in self._scalers, msg

        # index for getitem based resampling
        # ----------------------------------
        # NOTE: this should be refactored and probably in a DataLoader

        # create index
        self.index = self._construct_index(data, predict_mode=self.predict_mode)

        # data conversion to torch tensors
        # --------------------------------

        # convert to torch tensor for high performance data loading later
        self.data = self._data_to_tensors(data)

        # check that all tensors are finite
        self._check_tensors(self.data)

    def _check_params(self):
        """Check parameters of self against assumptions."""
        assert isinstance(
            self.max_encoder_length, int
        ), "max encoder length must be integer"
        assert (
            self.min_encoder_length <= self.max_encoder_length
        ), "max encoder length has to be larger equals min encoder length"
        assert isinstance(
            self.min_encoder_length, int
        ), "min encoder length must be integer"
        assert isinstance(
            self.max_prediction_length, int
        ), "max prediction length must be integer"
        assert (
            self.min_prediction_length <= self.max_prediction_length
        ), "max prediction length has to be larger equals min prediction length"
        assert (
            self.min_prediction_length > 0
        ), "min prediction length must be larger than 0"
        assert isinstance(
            self.min_prediction_length, int
        ), "min prediction length must be integer"
        msg = (
            f"add_encoder_length should be boolean or 'auto' "
            f"but found {self.add_encoder_length}"
        )
        assert isinstance(self.add_encoder_length, bool), msg

        for target in self.target_names:
            assert (
                target not in self._time_varying_known_reals
            ), f"target {target} should be an unknown continuous variable in the future"

        assert self.min_lag > 0, "lags should be positive"

    def _data_properties(self, data: pd.DataFrame) -> DataProperties:
        """Returns a dict with properties of the data used later.

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        dict
            dictionary with properties of the data.
            The following fields are returned:

            * columns : list[str]
                list of column names in the data
            * target_type : dict[str, str]
                type of target variable, categorial or real.
                Keys are target variable names in self.target_names.
                Value is either "categorical" or "real".
            * target_positive : dict[str, bool]
                whether target variable is positive.
                Keys are target variable names in self.target_names that are real.
                Value is True if all values of the target variable are positive.
                Computed and returned only if target_normalizer is "auto".
            * target_skew : dict[str, float]
                skew of target variable.
                Keys are target variable names in self.target_names that are
                real and positive. Value is the skew of the target variable.
                Computed and returned only if target_normalizer is "auto".
        """
        target_norm = self.target_normalizer
        details_required = isinstance(target_norm, str) and target_norm == "auto"

        props = {"target_type": {}, "target_skew": {}, "target_positive": {}}
        props["columns"] = data.columns.tolist()
        for target in self.target_names:
            if data[target].dtype.kind != "f":  # category
                props["target_type"][target] = "categorical"
            else:
                props["target_type"][target] = "real"

                if details_required:
                    props["target_positive"][target] = (data[target] > 0).all()
                    if props["target_positive"][target]:
                        props["target_skew"][target] = data[target].skew()
        return props

    def _set_lagged_variables(self) -> None:
        """Add lagged variables to lists of variables.

        * generates lagged variable names and adds them to the appropriate lists
          of time-varying variables, typed by known/unknown and categorical/real
        * checks that all lagged variables passed by user adhere to the
          naming convention of lags
        """
        var_name_dict = {
            ("real", "known"): "_time_varying_known_reals",
            ("real", "unknown"): "_time_varying_unknown_reals",
            ("cat", "known"): "_time_varying_known_categoricals",
            ("cat", "unknown"): "_time_varying_unknown_categoricals",
        }

        def _attr(realcat, known):
            return getattr(self, var_name_dict[(realcat, known)])

        def _append_if_new(lst, x):
            if x not in lst:
                lst.append(x)

        # check that all names passed in self._lags appear as variables
        all_time_varying_var_names = [x for kw in var_name_dict for x in _attr(*kw)]
        for name in self._lags:
            if name not in all_time_varying_var_names:
                raise KeyError(
                    f"lagged variable {name} is not a known "
                    "nor unknown time-varying variable"
                )

        # add lagged variables to type indicators
        for name in self._lags:
            lagged_names = self._get_lagged_names(name)

            # add lags
            for realcat, known in var_name_dict:
                var_names = _attr(realcat, known)

                if name in var_names:
                    for lagged_name, lag in lagged_names.items():
                        # if lag is longer than horizon, lagged var becomes future-known
                        if known or lag < self.max_prediction_length:
                            _append_if_new(var_names, lagged_name)
                        elif lag < self.max_prediction_length:
                            _append_if_new(_attr(realcat, "known"), lagged_name)

    @property
    def dropout_categoricals(self) -> list[str]:
        """
        list of categorical variables that are unknown when making a
        forecast without observed history
        """
        return [
            name
            for name, encoder in self._categorical_encoders.items()
            if encoder.add_nan
        ]

    def _get_lagged_names(self, name: str) -> dict[str, int]:
        """
        Generate names for lagged variables

        Parameters
        ----------
        name : str
            name of variable to lag

        Returns
        -------
        dict[str, int]
            dictionary mapping new variable names to lags
        """
        return {f"{name}_lagged_by_{lag}": lag for lag in self._lags.get(name, [])}

    @property
    @lru_cache(None)
    def lagged_variables(self) -> dict[str, str]:
        """Lagged variables.

        Parameters
        ----------
        dict[str, str]
            dictionary of variable names corresponding to lagged variables,
            mapped to variable that is lagged
        """
        vars = {}
        for name in self._lags:
            vars.update({lag_name: name for lag_name in self._get_lagged_names(name)})
        return vars

    @property
    @lru_cache(None)
    def lagged_targets(self) -> dict[str, str]:
        """Subset of lagged_variables to variables that are lagged targets.

        Parameters
        ----------
        dict[str, str]
            dictionary of variable names corresponding to lagged variables,
            mapped to variable that is lagged
        """
        vars = {}
        for name in self._lags:
            vars.update(
                {
                    lag_name: name
                    for lag_name in self._get_lagged_names(name)
                    if name in self.target_names
                }
            )
        return vars

    @property
    @lru_cache(None)
    def min_lag(self) -> int:
        """
        Minimum number of time steps variables are lagged.

        Returns
        -------
        int: minimum lag
        """
        if len(self._lags) == 0:
            return 1e9
        else:
            return min([min(lag) for lag in self._lags.values()])

    @property
    @lru_cache(None)
    def max_lag(self) -> int:
        """
        Maximum number of time steps variables are lagged.

        Returns
        -------
        int: maximum lag
        """
        if len(self._lags) == 0:
            return 0
        else:
            return max([max(lag) for lag in self._lags.values()])

    def _set_target_normalizer(
        self,
        data_properties: DataProperties,
        target_normalizer: Union[NORMALIZER, str, list, tuple],
    ) -> TorchNormalizer:
        """Determine target normalizer.

        Determines normalizers for variables based on self.target_normalizer setting.

        Coerces normalizers to torch normalizer, and deals with the "auto" setting.

        In the auto case, the normalizer for a variable x is determined as follows:

        * if x is categorical, a NaNLabelEncoder is used
        * if x is real and max_encoder_length > 20 and min_encoder_length > 1,
            an EncoderNormalizer is used, otherwise a GroupNormalizer is used.
            The transformation used in it is determined as follows:
        * if x is real and positive, a log transformation is used if the skew of x is
            larger than 2.5, otherwise a ReLU transformation is used
        * if x is real and not positive, no transformation is used

        The "auto" case uses metadata from the data passed in ``data_properties``,
        otherwise the ``data_properties`` are not used.

        Parameters
        ----------
        data_properties : dict
            Dictionary of data properties as returned by self._data_properties(data)
        target_normalizer : Union[NORMALIZER, str, list, tuple, None]
            Normalizer for target variable. If "auto", the normalizer is determined
            as above.

        Returns
        -------
        TorchNormalizer
            Normalizer for target variable, determined as above.
        """
        if isinstance(target_normalizer, str) and target_normalizer == "auto":
            target_normalizer = self._get_auto_normalizer(data_properties)
        elif isinstance(target_normalizer, (tuple, list)):
            target_normalizer = MultiNormalizer(self.target_normalizer)
        elif target_normalizer is None:
            target_normalizer = TorchNormalizer(method="identity")

        # validation
        assert (
            not isinstance(target_normalizer, EncoderNormalizer)
            or self.min_encoder_length >= target_normalizer.min_length
        ), "EncoderNormalizer is only allowed if min_encoder_length > 1"
        assert isinstance(target_normalizer, (TorchNormalizer, NaNLabelEncoder)), (
            f"target_normalizer has to be either None or of "
            f"class TorchNormalizer but found {target_normalizer}"
        )
        assert not self.multi_target or isinstance(
            target_normalizer, MultiNormalizer
        ), (
            "multiple targets / list of targets requires MultiNormalizer as "
            f"target_normalizer but found {target_normalizer}"
        )
        return target_normalizer

    def _get_auto_normalizer(self, data_properties: DataProperties) -> TorchNormalizer:
        """Get normalizer for auto setting, using data_properties.

        See docstring of _set_target_normalizer for details.

        Parameters
        ----------
        data_properties : dict
            Dictionary of data properties as returned by self._data_properties(data)

        Returns
        -------
        TorchNormalizer
            Normalizer for target variable
        """
        normalizers = []
        for target in self.target_names:
            if data_properties["target_type"][target] == "categorical":
                normalizers.append(NaNLabelEncoder())
                if self.add_target_scales:
                    warnings.warn(
                        "Target scales will be only added for continous targets",
                        UserWarning,
                    )
            else:  # real
                if data_properties["target_positive"][target]:
                    if data_properties["target_skew"][target] > 2.5:
                        transformer = "log"
                    else:
                        transformer = "relu"
                else:
                    transformer = None
                if self.max_encoder_length > 20 and self.min_encoder_length > 1:
                    normalizers.append(EncoderNormalizer(transformation=transformer))
                else:
                    normalizers.append(GroupNormalizer(transformation=transformer))
        if self.multi_target:
            target_normalizer = MultiNormalizer(normalizers)
        else:
            target_normalizer = normalizers[0]
        return target_normalizer

    @property
    @lru_cache(None)
    def _group_ids_mapping(self) -> dict[str, str]:
        """
        Mapping of group id names to group ids used to identify series in dataset -
        group ids can also be used for target normalizer.

        The former can change from training to validation and test dataset
        while the later must not.
        """
        return {name: f"__group_id__{name}" for name in self.group_ids}

    @property
    @lru_cache(None)
    def _group_ids(self) -> list[str]:
        """
        Group ids used to identify series in dataset.

        See :py:meth:`~TimeSeriesDataSet._group_ids_mapping` for details.
        """
        return list(self._group_ids_mapping.values())

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate assumptions on data.."""
        assert (
            data[self.time_idx].dtype.kind == "i"
        ), "Timeseries index should be of type integer"
        # numeric categoricals which can cause issues in tensorborad logging
        category_columns = data.head(1).select_dtypes("category").columns
        object_columns = data.head(1).select_dtypes(object).columns
        for name in self.flat_categoricals:
            if name not in data.columns:
                raise KeyError(f"variable {name} specified but not found in data")
            if not (
                name in object_columns
                or (
                    name in category_columns
                    and data[name].cat.categories.dtype.kind not in "bifc"
                )
            ):
                raise ValueError(
                    f"Data type of category {name} was found to be numeric"
                    " - use a string type / categorified string"
                )
        # check for "." in column names
        columns_with_dot = data.columns[data.columns.str.contains(r"\.")]
        if len(columns_with_dot) > 0:
            raise ValueError(
                f"column names must not contain '.' characters. "
                f"Names {columns_with_dot.tolist()} are invalid"
            )

        assert data.index.is_unique, "data index has to be unique"

        if len(self._lags) > 0:
            for name in self._lags:
                lagged_names = self._get_lagged_names(name)
                for lagged_name in lagged_names:
                    assert lagged_name not in data.columns, (
                        f"{lagged_name} is a protected column and must not be "
                        "present in data"
                    )

    def save(self, fname: str) -> None:
        """
        Save dataset to disk

        Args:
            fname (str): filename to save to
        """
        torch.save(self, fname)

    @classmethod
    def load(cls: type[TimeSeriesDataType], fname: str) -> TimeSeriesDataType:
        """
        Load dataset from disk

        Args:
            fname (str): filename to load from

        Returns:
            TimeSeriesDataSet
        """
        obj = torch.load(fname)
        assert isinstance(obj, cls), f"Loaded file is not of class {cls}"
        return obj

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scale continuous variables, encode categories and set aside target and weight.

        Args:
            data (pd.DataFrame): original data

        Returns:
            pd.DataFrame: pre-processed dataframe
        """
        # add lags to data
        for name in self._lags:
            # todo: add support for variable groups
            msg = (
                f"lagged variables that are in {self._variable_groups} "
                "are not supported yet"
            )
            assert name not in self._variable_groups, msg

            for lagged_name, lag in self._get_lagged_names(name).items():
                data[lagged_name] = data.groupby(self.group_ids, observed=True)[
                    name
                ].shift(lag)

        # encode group ids - this encoding
        for name, group_name in self._group_ids_mapping.items():
            # use existing encoder - but a copy of it not too loose current encodings
            encoder = deepcopy(
                self._categorical_encoders.get(group_name, NaNLabelEncoder())
            )
            self._categorical_encoders[group_name] = encoder.fit(
                data[name].to_numpy().reshape(-1), overwrite=False
            )
            data[group_name] = self.transform_values(
                name, data[name], inverse=False, group_id=True
            )

        # encode categoricals first to ensure
        # that group normalizer relies on encoded categories
        if isinstance(
            self.target_normalizer, (GroupNormalizer, MultiNormalizer)
        ):  # if we use a group normalizer, group_ids must be encoded as well
            group_ids_to_encode = self.group_ids
        else:
            group_ids_to_encode = []
        for name in dict.fromkeys(group_ids_to_encode + self.categoricals):
            if name in self.lagged_variables:
                continue  # do not encode here but only in transform
            if name in self._variable_groups:  # fit groups
                columns = self._variable_groups[name]
                if name not in self._categorical_encoders:
                    self._categorical_encoders[name] = NaNLabelEncoder().fit(
                        data[columns].to_numpy().reshape(-1)
                    )
                elif self._categorical_encoders[name] is not None:
                    try:
                        check_is_fitted(self._categorical_encoders[name])
                    except NotFittedError:
                        self._categorical_encoders[name] = self._categorical_encoders[
                            name
                        ].fit(data[columns].to_numpy().reshape(-1))
            else:
                if name not in self._categorical_encoders:
                    self._categorical_encoders[name] = NaNLabelEncoder().fit(data[name])
                elif (
                    self._categorical_encoders[name] is not None
                    and name not in self.target_names
                ):
                    try:
                        check_is_fitted(self._categorical_encoders[name])
                    except NotFittedError:
                        self._categorical_encoders[name] = self._categorical_encoders[
                            name
                        ].fit(data[name])

        # encode them
        for name in dict.fromkeys(group_ids_to_encode + self.flat_categoricals):
            # targets and its lagged versions are handled separetely
            if name not in self.target_names and name not in self.lagged_targets:
                data[name] = self.transform_values(
                    name,
                    data[name],
                    inverse=False,
                    ignore_na=name in self.lagged_variables,
                )

        # save special variables
        assert (
            "__time_idx__" not in data.columns
        ), "__time_idx__ is a protected column and must not be present in data"
        data["__time_idx__"] = data[self.time_idx]  # save unscaled
        for target in self.target_names:
            msg = (
                f"__target__{target} is a protected column "
                "and must not be present in data"
            )
            assert f"__target__{target}" not in data.columns, msg
            data[f"__target__{target}"] = data[target]
        if self.weight is not None:
            data["__weight__"] = data[self.weight]

        # train target normalizer
        if self.target_normalizer is not None:
            # fit target normalizer
            try:
                check_is_fitted(self.target_normalizer)
            except NotFittedError:
                if isinstance(self.target_normalizer, EncoderNormalizer):
                    self.target_normalizer.fit(data[self.target])
                elif isinstance(
                    self.target_normalizer, (GroupNormalizer, MultiNormalizer)
                ):
                    self.target_normalizer.fit(data[self.target], data)
                else:
                    self.target_normalizer.fit(data[self.target])

            # transform target
            if isinstance(self.target_normalizer, EncoderNormalizer):
                # we approximate the scales and target transformation by assuming one
                # transformation over the entire time range but by each group
                common_init_args = [
                    name
                    for name in inspect.signature(
                        GroupNormalizer.__init__
                    ).parameters.keys()
                    if name
                    in inspect.signature(EncoderNormalizer.__init__).parameters.keys()
                    and name not in ["data", "self"]
                ]
                copy_kwargs = {
                    name: getattr(self.target_normalizer, name)
                    for name in common_init_args
                }
                normalizer = GroupNormalizer(groups=self.group_ids, **copy_kwargs)
                data[self.target], scales = normalizer.fit_transform(
                    data[self.target], data, return_norm=True
                )

            elif isinstance(self.target_normalizer, GroupNormalizer):
                data[self.target], scales = self.target_normalizer.transform(
                    data[self.target], data, return_norm=True
                )

            elif isinstance(self.target_normalizer, MultiNormalizer):
                transformed, scales = self.target_normalizer.transform(
                    data[self.target], data, return_norm=True
                )

                for idx, target in enumerate(self.target_names):
                    data[target] = transformed[idx]

                    if isinstance(self.target_normalizer[idx], NaNLabelEncoder):
                        # overwrite target because it requires encoding
                        # (continuous targets should not be normalized)
                        data[f"__target__{target}"] = data[target]

            elif isinstance(self.target_normalizer, NaNLabelEncoder):
                data[self.target] = self.target_normalizer.transform(data[self.target])
                # overwrite target because it requires encoding
                # (continuous targets should not be normalized)
                data[f"__target__{self.target}"] = data[self.target]
                scales = None

            else:
                data[self.target], scales = self.target_normalizer.transform(
                    data[self.target], return_norm=True
                )

            # add target scales
            if self.add_target_scales:
                if not isinstance(self.target_normalizer, MultiNormalizer):
                    scales = [scales]
                for target_idx, target in enumerate(self.target_names):
                    if not isinstance(
                        self.target_normalizers[target_idx], NaNLabelEncoder
                    ):
                        for scale_idx, name in enumerate(["center", "scale"]):
                            feature_name = f"{target}_{name}"
                            msg = (
                                f"{feature_name} is a protected column "
                                "and must not be present in data"
                            )
                            assert feature_name not in data.columns, msg

                            data[feature_name] = scales[target_idx][
                                :, scale_idx
                            ].squeeze()
                            if feature_name not in self.reals:
                                self._static_reals.append(feature_name)

        # rescale continuous variables apart from target
        for name in self.reals:
            if name in self.target_names or name in self.lagged_variables:
                # lagged variables are only transformed - not fitted
                continue
            elif name not in self._scalers:
                self._scalers[name] = StandardScaler().fit(data[[name]])
            elif self._scalers[name] is not None:
                try:
                    check_is_fitted(self._scalers[name])
                except NotFittedError:
                    if isinstance(self._scalers[name], GroupNormalizer):
                        self._scalers[name] = self._scalers[name].fit(data[name], data)
                    else:
                        self._scalers[name] = self._scalers[name].fit(data[[name]])

        # encode after fitting
        for name in self.reals:
            # targets are handled separately
            transformer = self.get_transformer(name)
            if (
                name not in self.target_names
                and transformer is not None
                and not isinstance(transformer, EncoderNormalizer)
            ):
                data[name] = self.transform_values(
                    name, data[name], data=data, inverse=False
                )

        # encode lagged categorical targets
        for name in self.lagged_targets:
            # normalizer only now available
            if name in self.flat_categoricals:
                data[name] = self.transform_values(
                    name, data[name], inverse=False, ignore_na=True
                )

        # encode constant values
        self.encoded_constant_fill_strategy = {}
        for name, value in self._constant_fill_strategy.items():
            if name in self.target_names:
                self.encoded_constant_fill_strategy[f"__target__{name}"] = value
            self.encoded_constant_fill_strategy[name] = self.transform_values(
                name, np.array([value]), data=data, inverse=False
            )[0]

        # shorten data by maximum of lagged sequences to avoid NA values -
        # shorten only after encoding
        if self.max_lag > 0:
            # negative tail implementation as .groupby().tail(-self.max_lag)
            # is not implemented in pandas
            g = data.groupby(self._group_ids, observed=True)
            data = g._selected_obj[g.cumcount() >= self.max_lag]
        return data

    def get_transformer(
        self, name: str, group_id: bool = False
    ) -> Union[NORMALIZER, Any, None]:
        """Get transformer for variable.

        Parameters
        ----------
        name : str
            variable name
        group_id : bool, optional, default=False
            Whether the passed name refers to a group id,
            different encoders are used for these.

        Returns
        -------
        transformer: Union[NORMALIZER, Any, None]
            transformer for variable, None if no transformer is available
        """
        if group_id:
            name = self._group_ids_mapping[name]
        elif (
            name in self.lagged_variables
        ):  # recover transformer fitted on non-lagged variable
            name = self.lagged_variables[name]

        if name in self.flat_categoricals + self.group_ids + self._group_ids:
            name = self.variable_to_group_mapping.get(name, name)  # map name to encoder

            # take target normalizer if required
            if name in self.target_names:
                transformer = self.target_normalizers[self.target_names.index(name)]
            else:
                transformer = self._categorical_encoders.get(name, None)
            return transformer

        elif name in self.reals:
            # take target normalizer if required
            if name in self.target_names:
                transformer = self.target_normalizers[self.target_names.index(name)]
            else:
                transformer = self._scalers.get(name, None)
            return transformer
        else:
            return None

    def transform_values(
        self,
        name: str,
        values: Union[pd.Series, torch.Tensor, np.ndarray],
        data: pd.DataFrame = None,
        inverse=False,
        group_id: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Scale and encode values.

        Parameters
        ----------
        name : str
            name of variable
        values : Union[pd.Series, torch.Tensor, np.ndarray]
            values to encode/scale
        data : pd.DataFrame, optional, default=None
            extra data used for scaling (e.g. dataframe with groups columns)
        inverse : bool, optional, default=False
            whether transform is plain (True), or inverse (False)
        group_id : bool, optional, default=False
            whether the passed name refers to a group id -
            different encoders are used for these
        **kwargs: additional arguments for transform/inverse_transform method

        Returns
        -------
        np.ndarray
            (de/en)coded/(de)scaled values
        """
        transformer = self.get_transformer(name, group_id=group_id)
        if transformer is None:
            return values
        if inverse:
            transform = transformer.inverse_transform
        else:
            transform = transformer.transform

        if group_id:
            name = self._group_ids_mapping[name]
        # remaining categories
        if name in self.flat_categoricals + self.group_ids + self._group_ids:
            return transform(values, **kwargs)

        # reals
        elif name in self.reals:
            if isinstance(transformer, GroupNormalizer):
                return transform(values, data, **kwargs)
            elif isinstance(transformer, EncoderNormalizer):
                return transform(values, **kwargs)
            else:
                if isinstance(values, pd.Series):
                    values = values.to_frame()
                    return np.asarray(transform(values, **kwargs)).reshape(-1)
                else:
                    values = values.reshape(-1, 1)
                    return transform(values, **kwargs).reshape(-1)
        else:
            return values

    def _data_to_tensors(self, data: pd.DataFrame) -> dict[str, torch.Tensor]:
        """Convert data to tensors for faster access with :py:meth:`~__getitem__`.

        Parameters
        ----------
        data : pd.DataFrame
            preprocessed data

        Returns
        -------
        dict[str, torch.Tensor]
            dictionary of tensors for continous, categorical data, groups, target and
            time index
        """

        def _to_tensor(cols, long=True) -> torch.Tensor:
            """Convert data[cols] to torch tensor.

            Converts sub-frames to numpy and then to torch tensor.
            Makes the following choices for types:

            * float columns are converted to torch.float
            * integer columns are converted to torch.int64 or torch.long,
              depending on the long argument
            """
            if not isinstance(cols, list) and cols not in data.columns:
                return None
            if isinstance(cols, list) and len(cols) == 0:
                dtypekind = "f"
            elif isinstance(cols, list):  # and len(cols) > 0
                dtypekind = data.dtypes[cols[0]].kind
            else:
                dtypekind = data.dtypes[cols].kind
            if not long:
                return torch.tensor(data[cols].to_numpy(np.int64), dtype=torch.int64)
            elif dtypekind in "bi":
                return torch.tensor(data[cols].to_numpy(np.int64), dtype=torch.long)
            else:
                return torch.tensor(data[cols].to_numpy(np.float64), dtype=torch.float)

        index = _to_tensor(self._group_ids, long=False)
        time = _to_tensor("__time_idx__", long=False)
        categorical = _to_tensor(self.flat_categoricals, long=False)

        weight = _to_tensor("__weight__")

        # get target
        if isinstance(self.target_normalizer, NaNLabelEncoder):
            target = [_to_tensor(f"__target__{self.target}")]
        else:
            if not isinstance(self.target, str):  # multi-target
                target = [_to_tensor(f"__target__{name}") for name in self.target_names]
            else:
                target = [_to_tensor(f"__target__{self.target}")]

        # continuous covariates
        continuous = _to_tensor(self.reals)

        tensors = dict(
            reals=continuous,
            categoricals=categorical,
            groups=index,
            target=target,
            weight=weight,
            time=time,
        )
        return tensors

    def _check_tensors(self, tensors):
        """Check for non-finite values in tensors."""
        var_names_dict = {
            "reals": self.reals,
            "categoricals": self.flat_categoricals,
            "groups": self.group_ids,
            "target": self.target_names,
            "weight": self.weight,
            "time": self.time_idx,
        }

        for key, tensor in tensors.items():
            var_names = var_names_dict[key]
            if tensor is not None:
                if isinstance(tensor, list):
                    for idx, target_tensor in enumerate(tensor):
                        check_for_nonfinite(target_tensor, var_names[idx])
                else:
                    check_for_nonfinite(tensor, var_names)

    @property
    def categoricals(self) -> list[str]:
        """
        Categorical variables as used for modelling.

        Returns:
            list[str]: list of variables
        """
        return (
            self._static_categoricals
            + self._time_varying_known_categoricals
            + self._time_varying_unknown_categoricals
        )

    @property
    def flat_categoricals(self) -> list[str]:
        """
        Categorical variables as defined in input data.

        Returns:
            list[str]: list of variables
        """
        categories = []
        for name in self.categoricals:
            if name in self._variable_groups:
                categories.extend(self._variable_groups[name])
            else:
                categories.append(name)
        return categories

    @property
    def variable_to_group_mapping(self) -> dict[str, str]:
        """
        Mapping from categorical variables to variables in input data.

        Returns
        -------
        dict[str, str]
            dictionary, maps :py:meth:`~categorical` to :py:meth:`~flat_categoricals`.
        """
        groups = {}
        for group_name, sublist in self._variable_groups.items():
            groups.update({name: group_name for name in sublist})
        return groups

    @property
    def reals(self) -> list[str]:
        """
        Continous variables as used for modelling.

        Returns:
            list[str]: list of variables
        """
        return (
            self._static_reals
            + self._time_varying_known_reals
            + self._time_varying_unknown_reals
        )

    @property
    @lru_cache(None)
    def target_names(self) -> list[str]:
        """
        List of targets.

        Returns:
            list[str]: list of targets
        """
        if self.multi_target:
            return self.target
        else:
            return [self.target]

    @property
    def multi_target(self) -> bool:
        """
        If dataset encodes one or multiple targets.

        Returns:
            bool: true if multiple targets
        """
        return isinstance(self.target, (list, tuple))

    @property
    def target_normalizers(self) -> list[TorchNormalizer]:
        """
        List of target normalizers aligned with ``target_names``.

        Returns:
            list[TorchNormalizer]: list of target normalizers
        """
        if isinstance(self.target_normalizer, MultiNormalizer):
            target_normalizers = self.target_normalizer.normalizers
        else:
            target_normalizers = [self.target_normalizer]
        return target_normalizers

    def get_parameters(self) -> dict[str, Any]:
        """Get parameters of self as dict.

        These can be used with :py:meth:`~from_parameters`
        to create a new dataset with the same scalers.

        Returns
        -------
        dict[str, Any]: dictionary of parameters
        """
        kwargs = {
            name: getattr(self, name)
            for name in inspect.signature(self.__class__.__init__).parameters.keys()
            if name not in ["data", "self"]
        }
        kwargs["categorical_encoders"] = self._categorical_encoders
        kwargs["scalers"] = self._scalers
        return kwargs

    @classmethod
    def from_dataset(
        cls: type[TimeSeriesDataType],
        dataset: TimeSeriesDataType,
        data: pd.DataFrame,
        stop_randomization: bool = False,
        predict: bool = False,
        **update_kwargs,
    ) -> TimeSeriesDataType:
        """Construct dataset with different data, same variable encoders, scalers, etc.

        Calls :py:meth:`~from_parameters` under the hood.

        May override parameters with update_kwargs.

        Parameters
        ----------
        dataset : TimeSeriesDataSet
            dataset from which to copy parameters
        data : pd.DataFrame
            data from which new dataset will be generated
        stop_randomization : bool, optional, default=None
            Whether to stop randomizing encoder and decoder lengths,
            useful for validation set.
        predict : bool, optional, default=False
            Whether to predict the decoder length on the last entries in the
            time index (i.e. one prediction per group only).
        **update_kwargs
            keyword arguments overrides, passed to constructor of the new dataset

        Returns
        -------
        TimeSeriesDataSet
            new dataset
        """
        return cls.from_parameters(
            dataset.get_parameters(),
            data,
            stop_randomization=stop_randomization,
            predict=predict,
            **update_kwargs,
        )

    @classmethod
    def from_parameters(
        cls: type[TimeSeriesDataType],
        parameters: dict[str, Any],
        data: pd.DataFrame,
        stop_randomization: bool = None,
        predict: bool = False,
        **update_kwargs,
    ) -> TimeSeriesDataType:
        """Construct dataset with different data, same variable encoders, scalers, etc.

        Returns TimeSeriesDataSet with same parameters as self, but different data.
        May override parameters with update_kwargs.

        Parameters
        ----------
        parameters : dict[str, Any]
            dataset parameters which to use for the new dataset
        data : pd.DataFrame
            data from which new dataset will be generated
        stop_randomization : bool, optional, default=None
            Whether to stop randomizing encoder and decoder lengths,
            useful for validation set.
        predict : bool, optional, default=False
            Whether to predict the decoder length on the last entries in the
            time index (i.e. one prediction per group only).
        **update_kwargs
            keyword arguments overrides, passed to constructor of the new dataset

        Returns
        -------
        TimeSeriesDataType
            new dataset
        """
        parameters = deepcopy(parameters)

        if predict:
            if isinstance(stop_randomization, bool) and not stop_randomization:
                warnings.warn(
                    "If predicting, no randomization should be possible - "
                    "setting stop_randomization=True",
                    UserWarning,
                )
            parameters["min_prediction_length"] = parameters["max_prediction_length"]
            parameters["predict_mode"] = True

        # this treats cases for randomize_length randomization:
        # if predict mode, always turned off, i.e., always stop_ransomization=True
        # otherwise, None defaults to False
        stop_randomization = predict or stop_randomization
        if stop_randomization:
            parameters["randomize_length"] = None
        parameters.update(update_kwargs)

        new = cls(data, **parameters)
        return new

    def _construct_index(self, data: pd.DataFrame, predict_mode: bool) -> pd.DataFrame:
        """Create index of samples returned by getitem dunder.

        Parameters
        ----------
        data : pd.DataFrame
            preprocessed data
        predict_mode : bool
            whether to create one sample per group
            with prediction length equals ``max_decoder_length``

        Returns
        -------
        pd.DataFrame
            index dataframe for timesteps and index dataframe for groups.
            It contains a list of all possible subsequences.
        """
        g = data.groupby(self._group_ids, observed=True)

        df_index_first = g["__time_idx__"].transform("first").to_frame("time_first")
        df_index_last = g["__time_idx__"].transform("last").to_frame("time_last")
        df_index_diff_to_next = (
            -g["__time_idx__"]
            .diff(-1)
            .fillna(-1)
            .astype(int)
            .to_frame("time_diff_to_next")
        )
        df_index = pd.concat(
            [df_index_first, df_index_last, df_index_diff_to_next], axis=1
        )
        df_index["index_start"] = np.arange(len(df_index))
        df_index["time"] = data["__time_idx__"]
        df_index["count"] = (df_index["time_last"] - df_index["time_first"]).astype(
            int
        ) + 1
        sequence_ids = g.ngroup()
        df_index["sequence_id"] = sequence_ids

        min_sequence_length = self.min_prediction_length + self.min_encoder_length
        max_sequence_length = self.max_prediction_length + self.max_encoder_length

        # calculate maximum index to include from current index_start
        max_time = (df_index["time"] + max_sequence_length - 1).clip(
            upper=df_index["count"] + df_index.time_first - 1
        )

        # if there are missing timesteps, we cannot say directly what
        # is the last timestep to include
        # therefore we iterate until it is found
        if (df_index["time_diff_to_next"] != 1).any():
            msg = (
                "Time difference between steps has been idenfied as larger than 1 - "
                "set allow_missing_timesteps=True"
            )
            assert self.allow_missing_timesteps, msg

        df_index["index_end"], missing_sequences = _find_end_indices(
            diffs=df_index.time_diff_to_next.to_numpy(),
            max_lengths=(max_time - df_index.time).to_numpy() + 1,
            min_length=min_sequence_length,
        )
        # add duplicates but mostly with shorter sequence length for start of timeseries
        # while the previous steps have ensured that we start a sequence on every time
        # step, the missing_sequences
        # ensure that there is a sequence that finishes on every timestep
        if len(missing_sequences) > 0:
            shortened_sequences = df_index.iloc[missing_sequences[:, 0]].assign(
                index_end=missing_sequences[:, 1]
            )

            # concatenate shortened sequences
            df_index = pd.concat(
                [df_index, shortened_sequences], axis=0, ignore_index=True
            )

        # filter out where encode and decode length are not satisfied
        df_index["sequence_length"] = (
            df_index["time"].iloc[df_index["index_end"]].to_numpy()
            - df_index["time"]
            + 1
        )

        # filter too short sequences
        df_index = df_index[
            # sequence must be at least of minimal prediction length
            lambda x: (x.sequence_length >= min_sequence_length)
            &
            # prediction must be for minimal prediction index + length of prediction
            (
                x["sequence_length"] + x["time"]
                >= self.min_prediction_idx + self.min_prediction_length
            )
        ]

        if predict_mode:
            # keep longest element per series
            # (i.e., the first element that spans to the end of the series)
            # filter all elements that are longer
            # than the allowed maximum sequence length
            df_index = df_index[
                lambda x: (x["time_last"] - x["time"] + 1 <= max_sequence_length)
                & (x["sequence_length"] >= min_sequence_length)
            ]
            # choose longest sequence
            df_index = df_index.loc[
                df_index.groupby("sequence_id").sequence_length.idxmax()
            ]

        # check that all groups/series have at least one entry in the index
        if not sequence_ids.isin(df_index.sequence_id).all():
            missing_groups = data.loc[
                ~sequence_ids.isin(df_index.sequence_id), self._group_ids
            ].drop_duplicates()
            # decode values
            for name, id in self._group_ids_mapping.items():
                missing_groups[id] = self.transform_values(
                    name, missing_groups[id], inverse=True, group_id=True
                )
            warnings.warn(
                "Min encoder length and/or min_prediction_idx and/or min "
                "prediction length and/or lags are too large for "
                f"{len(missing_groups)} series/groups which therefore are not present"
                " in the dataset index. "
                "This means no predictions can be made for those series. "
                f"First 10 removed groups: "
                f"{list(missing_groups.iloc[:10].to_dict(orient='index').values())}",
                UserWarning,
            )
        msg = (
            "filters should not remove entries all entries - "
            "check encoder/decoder lengths and lags"
        )
        assert len(df_index) > 0, msg

        return df_index

    def filter(self, filter_func: Callable, copy: bool = True) -> TimeSeriesDataType:
        """Filter subsequences in dataset.

        Uses interpretable version of index :py:meth:`~decoded_index`
        to filter subsequences in dataset.

        Parameters
        ----------
        filter_func : Callable
            function to filter. Should take :py:meth:`~decoded_index`
            dataframe as only argument which contains group ids and time index columns.
        copy : bool, optional, default=True
            whether to return copy of dataset (True) or filter inplace (False).

        Returns
        -------
        TimeSeriesDataSet
            filtered dataset
        """
        # calculate filter
        filtered_index = self.index[np.asarray(filter_func(self.decoded_index))]
        # raise error if filter removes all entries
        if len(filtered_index) == 0:
            raise ValueError("After applying filter no sub-sequences left in dataset")
        if copy:
            dataset = _copy(self)
            dataset.index = filtered_index
            return dataset
        else:
            self.index = filtered_index
            return self

    @property
    def decoded_index(self) -> pd.DataFrame:
        """
        Get interpretable version of index.

        DataFrame contains
        - group_id columns in original encoding
        - time_idx_first column: first time index of subsequence
        - time_idx_last columns: last time index of subsequence
        - time_idx_first_prediction columns: first time index which is in decoder

        Returns:
            pd.DataFrame: index that can be understood in terms of original data
        """
        # get dataframe to filter
        index_start = self.index["index_start"].to_numpy()
        index_last = self.index["index_end"].to_numpy()
        index = (
            # get group ids in order of index
            pd.DataFrame(
                self.data["groups"][index_start].numpy(), columns=self.group_ids
            )
            # to original values
            .apply(
                lambda x: self.transform_values(
                    name=x.name, values=x, group_id=True, inverse=True
                )
            )
            # add time index
            .assign(
                time_idx_first=self.data["time"][index_start].numpy(),
                time_idx_last=self.data["time"][index_last].numpy(),
                # prediction index is last time index - decoder length + 1
                time_idx_first_prediction=lambda x: x.time_idx_last
                - self.calculate_decoder_length(
                    time_last=x.time_idx_last,
                    sequence_length=x.time_idx_last - x.time_idx_first + 1,
                )
                + 1,
            )
        )
        return index

    def plot_randomization(
        self,
        betas: tuple[float, float] = None,
        length: int = None,
        min_length: int = None,
    ):
        """Plot expected randomized length distribution.

        Parameters
        ----------
        betas : tuple[float, float], optional, default=randomize_length of dataset
            Tuple of betas, e.g. ``(0.2, 0.05)`` to use for randomization.
        length : int, optional, default=max_encoder_length of dataset
            Length of sequence to plot.
        min_length : int, optional, default=min_encoder_length of dataset
            Minimum length of sequence to plot.

        Returns
        -------
        tuple[plt.Figure, torch.Tensor]
            tuple of figure and histogram based on 1000 samples
        """
        _check_matplotlib("plot_randomization")

        import matplotlib.pyplot as plt

        if betas is None:
            betas = self.randomize_length
        if length is None:
            length = self.max_encoder_length
        if min_length is None:
            min_length = self.min_encoder_length
        probabilities = Beta(betas[0], betas[1]).sample((1000,))

        lengths = ((length - min_length) * probabilities).round() + min_length

        fig, ax = plt.subplots()
        ax.hist(lengths)
        return fig, lengths

    def __len__(self) -> int:
        """
        Length of dataset.

        Returns:
            int: length
        """
        return self.index.shape[0]

    def set_overwrite_values(
        self,
        values: Union[float, torch.Tensor],
        variable: str,
        target: Union[str, slice] = "decoder",
    ) -> None:
        """Overwrite values in decoder or encoder (or both) for a specific variable.

        Parameters
        ----------
        values : Union[float, torch.Tensor]
            values to use for overwrite.
        variable : str
            variable whose values should be overwritten.
        target : Union[str, slice], optional)
            positions to overwrite. One of "decoder", "encoder" or "all" or
            a slice object which is directly used to overwrite indices,
            e.g., ``slice(-5, None)`` will overwrite
            the last 5 values. Defaults to "decoder".
        """
        values = torch.tensor(
            self.transform_values(
                variable, np.asarray(values).reshape(-1), inverse=False
            )
        ).squeeze()
        msg = (
            f"target has be one of 'all', 'decoder' or 'encoder' "
            f"but got target={target} instead"
        )
        assert target in ["all", "decoder", "encoder"], msg

        if variable in self._static_categoricals or variable in self._static_reals:
            target = "all"

        if variable in self.target_names:
            raise NotImplementedError("Target variable is not supported")
        if self.weight is not None and self.weight == variable:
            raise NotImplementedError("Weight variable is not supported")
        if isinstance(
            self._scalers.get(variable, self._categorical_encoders.get(variable)),
            TorchNormalizer,
        ):
            raise NotImplementedError(
                "TorchNormalizer (e.g. GroupNormalizer) is not supported"
            )

        if self._overwrite_values is None:
            self._overwrite_values = {}
        self._overwrite_values.update(
            dict(values=values, variable=variable, target=target)
        )

    def reset_overwrite_values(self) -> None:
        """
        Reset values used to override sample features.
        """
        self._overwrite_values = None

    def calculate_decoder_length(
        self,
        time_last: Union[int, pd.Series, np.ndarray],
        sequence_length: Union[int, pd.Series, np.ndarray],
    ) -> Union[int, pd.Series, np.ndarray]:
        """Calculate length of decoder.

        Parameters
        ----------
        time_last : Union[int, pd.Series, np.ndarray]
            last time index of the sequence
        sequence_length : Union[int, pd.Series, np.ndarray]
            total length of the sequence

        Returns
        -------
        Union[int, pd.Series, np.ndarray]
            decoder length(s)
        """
        if isinstance(time_last, int):
            decoder_length = min(
                time_last
                - (self.min_prediction_idx - 1),  # not going beyond min prediction idx
                self.max_prediction_length,  # maximum prediction length
                sequence_length
                - self.min_encoder_length,  # sequence length - min decoder length
            )
        else:
            decoder_length = np.min(
                [
                    time_last - (self.min_prediction_idx - 1),
                    sequence_length - self.min_encoder_length,
                ],
                axis=0,
            ).clip(max=self.max_prediction_length)
        return decoder_length

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Get sample for model

        Args:
            idx (int): index of prediction (between ``0`` and ``len(dataset) - 1``)

        Returns:
            tuple[dict[str, torch.Tensor], torch.Tensor]: x and y for model
        """
        index = self.index.iloc[idx]

        # slice data based on index
        idx_slice = slice(index.index_start, index.index_end + 1)

        data_cont = self.data["reals"][idx_slice].clone()
        data_cat = self.data["categoricals"][idx_slice].clone()
        time = self.data["time"][idx_slice].clone()
        target = [d[idx_slice].clone() for d in self.data["target"]]
        groups = self.data["groups"][index.index_start].clone()
        if self.data["weight"] is None:
            weight = None
        else:
            weight = self.data["weight"][idx_slice].clone()
        # get target scale in the form of a list
        target_scale = self.target_normalizer.get_parameters(groups, self.group_ids)
        if not isinstance(self.target_normalizer, MultiNormalizer):
            target_scale = [target_scale]

        # fill in missing values (if not all time indices are specified)
        sequence_length = len(time)
        if sequence_length < index.sequence_length:
            assert (
                self.allow_missing_timesteps
            ), "allow_missing_timesteps should be True if sequences have gaps"
            repetitions = torch.cat(
                [time[1:] - time[:-1], torch.ones(1, dtype=time.dtype)]
            )
            indices = torch.repeat_interleave(torch.arange(len(time)), repetitions)
            repetition_indices = torch.cat(
                [torch.tensor([False], dtype=torch.bool), indices[1:] == indices[:-1]]
            )

            # select data
            data_cat = data_cat[indices]
            data_cont = data_cont[indices]
            target = [d[indices] for d in target]
            if weight is not None:
                weight = weight[indices]

            # reset index
            if self.time_idx in self.reals:
                time_idx = self.reals.index(self.time_idx)
                data_cont[:, time_idx] = torch.linspace(
                    data_cont[0, time_idx],
                    data_cont[-1, time_idx],
                    len(target[0]),
                    dtype=data_cont.dtype,
                )

            # make replacements to fill in categories
            for name, value in self.encoded_constant_fill_strategy.items():
                if name in self.reals:
                    data_cont[repetition_indices, self.reals.index(name)] = value
                elif name in [
                    f"__target__{target_name}" for target_name in self.target_names
                ]:
                    target_pos = self.target_names.index(name[len("__target__") :])
                    target[target_pos][repetition_indices] = value
                elif name in self.flat_categoricals:
                    data_cat[repetition_indices, self.flat_categoricals.index(name)] = (
                        value
                    )
                elif name in self.target_names:  # target is just not an input value
                    pass
                else:
                    raise KeyError(
                        f"Variable {name} is not known and thus cannot be filled in"
                    )

            sequence_length = len(target[0])

        # determine data window
        assert (
            sequence_length >= self.min_prediction_length
        ), "Sequence length should be at least minimum prediction length"
        # determine prediction/decode length and encode length
        decoder_length = self.calculate_decoder_length(time[-1], sequence_length)
        encoder_length = sequence_length - decoder_length
        assert (
            decoder_length >= self.min_prediction_length
        ), "Decoder length should be at least minimum prediction length"
        assert (
            encoder_length >= self.min_encoder_length
        ), "Encoder length should be at least minimum encoder length"

        if self.randomize_length is not None:  # randomization improves generalization
            # modify encode and decode lengths
            modifiable_encoder_length = encoder_length - self.min_encoder_length
            encoder_length_probability = Beta(
                self.randomize_length[0], self.randomize_length[1]
            ).sample()

            # subsample a new/smaller encode length
            new_encoder_length = self.min_encoder_length + int(
                (modifiable_encoder_length * encoder_length_probability).round()
            )

            # extend decode length if possible
            new_decoder_length = min(
                decoder_length + (encoder_length - new_encoder_length),
                self.max_prediction_length,
            )

            # select subset of sequence of new sequence
            if new_encoder_length + new_decoder_length < len(target[0]):
                data_cat = data_cat[
                    encoder_length - new_encoder_length : encoder_length
                    + new_decoder_length
                ]
                data_cont = data_cont[
                    encoder_length - new_encoder_length : encoder_length
                    + new_decoder_length
                ]
                target = [
                    t[
                        encoder_length - new_encoder_length : encoder_length
                        + new_decoder_length
                    ]
                    for t in target
                ]
                if weight is not None:
                    weight = weight[
                        encoder_length - new_encoder_length : encoder_length
                        + new_decoder_length
                    ]
                encoder_length = new_encoder_length
                decoder_length = new_decoder_length

            # switch some variables to nan if encode length is 0
            if encoder_length == 0 and len(self.dropout_categoricals) > 0:
                data_cat[
                    :,
                    [
                        self.flat_categoricals.index(c)
                        for c in self.dropout_categoricals
                    ],
                ] = 0  # zero is encoded nan

        assert decoder_length > 0, "Decoder length should be greater than 0"
        assert encoder_length >= 0, "Encoder length should be at least 0"

        if self.add_relative_time_idx:
            data_cont[:, self.reals.index("relative_time_idx")] = (
                torch.arange(-encoder_length, decoder_length, dtype=data_cont.dtype)
                / self.max_encoder_length
            )

        if self.add_encoder_length:
            data_cont[:, self.reals.index("encoder_length")] = (
                (encoder_length - 0.5 * self.max_encoder_length)
                / self.max_encoder_length
                * 2.0
            )

        # rescale target
        for idx, target_normalizer in enumerate(self.target_normalizers):
            if isinstance(target_normalizer, EncoderNormalizer):
                target_name = self.target_names[idx]
                # fit and transform
                target_normalizer.fit(target[idx][:encoder_length])
                # get new scale
                single_target_scale = target_normalizer.get_parameters()
                # modify input data
                if target_name in self.reals:
                    data_cont[:, self.reals.index(target_name)] = (
                        target_normalizer.transform(target[idx])
                    )
                if self.add_target_scales:
                    data_cont[:, self.reals.index(f"{target_name}_center")] = (
                        self.transform_values(
                            f"{target_name}_center", single_target_scale[0]
                        )[0]
                    )
                    data_cont[:, self.reals.index(f"{target_name}_scale")] = (
                        self.transform_values(
                            f"{target_name}_scale", single_target_scale[1]
                        )[0]
                    )
                # scale needs to be numpy to be consistent with GroupNormalizer
                target_scale[idx] = single_target_scale.numpy()

        # rescale covariates
        for name in self.reals:
            if name not in self.target_names and name not in self.lagged_variables:
                normalizer = self.get_transformer(name)
                if isinstance(normalizer, EncoderNormalizer):
                    # fit and transform
                    pos = self.reals.index(name)
                    normalizer.fit(data_cont[:encoder_length, pos])
                    # transform
                    data_cont[:, pos] = normalizer.transform(data_cont[:, pos])

        # also normalize lagged variables
        for name in self.reals:
            if name in self.lagged_variables:
                normalizer = self.get_transformer(name)
                if isinstance(normalizer, EncoderNormalizer):
                    pos = self.reals.index(name)
                    data_cont[:, pos] = normalizer.transform(data_cont[:, pos])

        # overwrite values
        if self._overwrite_values is not None:
            if isinstance(self._overwrite_values["target"], slice):
                positions = self._overwrite_values["target"]
            elif self._overwrite_values["target"] == "all":
                positions = slice(None)
            elif self._overwrite_values["target"] == "encoder":
                positions = slice(None, encoder_length)
            else:  # decoder
                positions = slice(encoder_length, None)

            if self._overwrite_values["variable"] in self.reals:
                idx = self.reals.index(self._overwrite_values["variable"])
                data_cont[positions, idx] = self._overwrite_values["values"]
            else:
                msg = (
                    "overwrite values variable has to be "
                    "either in real or categorical variables"
                )
                assert self._overwrite_values["variable"] in self.flat_categoricals, msg
                idx = self.flat_categoricals.index(self._overwrite_values["variable"])
                data_cat[positions, idx] = self._overwrite_values["values"]

        # weight is only required for decoder
        if weight is not None:
            weight = weight[encoder_length:]

        # if user defined target as list, output should be list, otherwise tensor
        if self.multi_target:
            encoder_target = [t[:encoder_length] for t in target]
            target = [t[encoder_length:] for t in target]
        else:
            encoder_target = target[0][:encoder_length]
            target = target[0][encoder_length:]
            target_scale = target_scale[0]

        return (
            dict(
                x_cat=data_cat,
                x_cont=data_cont,
                encoder_length=encoder_length,
                decoder_length=decoder_length,
                encoder_target=encoder_target,
                encoder_time_idx_start=time[0],
                groups=groups,
                target_scale=target_scale,
            ),
            (target, weight),
        )

    @staticmethod
    def _collate_fn(
        batches: list[tuple[dict[str, torch.Tensor], torch.Tensor]],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Collate function to combine items into mini-batch for dataloader.

        Parameters
        ----------
        batches (list[tuple[dict[str, torch.Tensor], torch.Tensor]]):
            List of samples generated with :py:meth:`~__getitem__`.

        Returns
        -------
        dict[str, torch.Tensor]
            dictionary of minibatches with keys:

            * encoder_cat: (batch_size, encoder_length, num_categorical),
                categorical variables for encoder
            * encoder_cont: (batch_size, encoder_length, num_real),
                continuous variables for encoder
            * encoder_target: (batch_size, encoder_length, num_target),
                target variables for encoder
            * encoder_lengths: (batch_size), length of encoder
            * decoder_cat: (batch_size, decoder_length, num_categorical),
                categorical variables for decoder
            * decoder_cont: (batch_size, decoder_length, num_real),
                continuous variables for decoder
            * decoder_target: (batch_size, decoder_length, num_target),
                target variables for decoder
            * decoder_lengths: (batch_size), length of decoder
            * decoder_time_idx: (batch_size, decoder_length),
                time index for decoder
            * groups: (batch_size), group ids
            * target_scale: (batch_size, num_target),
                scale of target variables

        tuple[torch.Tensor, torch.Tensor]
            minibatch, 2-tuple with entries:

            * target: (batch_size, decoder_length, num_target),
                target variables
            * weight: (batch_size, decoder_length),
                weights for target variables
        """
        # collate function for dataloader
        # lengths
        encoder_lengths = torch.tensor(
            [batch[0]["encoder_length"] for batch in batches], dtype=torch.long
        )
        decoder_lengths = torch.tensor(
            [batch[0]["decoder_length"] for batch in batches], dtype=torch.long
        )

        # ids
        decoder_time_idx_start = (
            torch.tensor(
                [batch[0]["encoder_time_idx_start"] for batch in batches],
                dtype=torch.long,
            )
            + encoder_lengths
        )
        decoder_time_idx = decoder_time_idx_start.unsqueeze(1) + torch.arange(
            decoder_lengths.max()
        ).unsqueeze(0)
        groups = torch.stack([batch[0]["groups"] for batch in batches])

        # features
        encoder_cont = rnn.pad_sequence(
            [
                batch[0]["x_cont"][:length]
                for length, batch in zip(encoder_lengths, batches)
            ],
            batch_first=True,
        )
        encoder_cat = rnn.pad_sequence(
            [
                batch[0]["x_cat"][:length]
                for length, batch in zip(encoder_lengths, batches)
            ],
            batch_first=True,
        )

        decoder_cont = rnn.pad_sequence(
            [
                batch[0]["x_cont"][length:]
                for length, batch in zip(encoder_lengths, batches)
            ],
            batch_first=True,
        )
        decoder_cat = rnn.pad_sequence(
            [
                batch[0]["x_cat"][length:]
                for length, batch in zip(encoder_lengths, batches)
            ],
            batch_first=True,
        )

        # target scale
        if isinstance(batches[0][0]["target_scale"], torch.Tensor):  # stack tensor
            target_scale = torch.stack([batch[0]["target_scale"] for batch in batches])
        elif isinstance(batches[0][0]["target_scale"], (list, tuple)):
            target_scale = []
            for idx in range(len(batches[0][0]["target_scale"])):
                if isinstance(
                    batches[0][0]["target_scale"][idx], torch.Tensor
                ):  # stack tensor
                    scale = torch.stack(
                        [batch[0]["target_scale"][idx] for batch in batches]
                    )
                else:
                    scale = torch.from_numpy(
                        np.array(
                            [batch[0]["target_scale"][idx] for batch in batches],
                            dtype=np.float32,
                        ),
                    )
                target_scale.append(scale)
        else:  # convert to tensor
            target_scale = torch.from_numpy(
                np.array(
                    [batch[0]["target_scale"] for batch in batches], dtype=np.float32
                ),
            )

        # target and weight
        if isinstance(batches[0][1][0], (tuple, list)):
            target = [
                rnn.pad_sequence(
                    [batch[1][0][idx] for batch in batches], batch_first=True
                )
                for idx in range(len(batches[0][1][0]))
            ]
            encoder_target = [
                rnn.pad_sequence(
                    [batch[0]["encoder_target"][idx] for batch in batches],
                    batch_first=True,
                )
                for idx in range(len(batches[0][1][0]))
            ]
        else:
            target = rnn.pad_sequence(
                [batch[1][0] for batch in batches], batch_first=True
            )
            encoder_target = rnn.pad_sequence(
                [batch[0]["encoder_target"] for batch in batches], batch_first=True
            )

        if batches[0][1][1] is not None:
            weight = rnn.pad_sequence(
                [batch[1][1] for batch in batches], batch_first=True
            )
        else:
            weight = None

        return (
            dict(
                encoder_cat=encoder_cat,
                encoder_cont=encoder_cont,
                encoder_target=encoder_target,
                encoder_lengths=encoder_lengths,
                decoder_cat=decoder_cat,
                decoder_cont=decoder_cont,
                decoder_target=target,
                decoder_lengths=decoder_lengths,
                decoder_time_idx=decoder_time_idx,
                groups=groups,
                target_scale=target_scale,
            ),
            (target, weight),
        )

    def to_dataloader(
        self,
        train: bool = True,
        batch_size: int = 64,
        batch_sampler: Union[Sampler, str] = None,
        **kwargs,
    ) -> DataLoader:
        """Construct dataloader from dataset, for use in models.

        Parameters
        ----------
        train : bool, optional, default=Trze
            whether dataloader is used for training (True) or prediction (False).
            Will shuffle and drop last batch if True. Defaults to True.
        batch_size : int, optional, default=64
            batch size for training model. Defaults to 64.
        batch_sampler : Sampler, str, or None, optional, default=None
            torch batch sampler or string. One of

            * "synchronized": ensure that samples in decoder are aligned in time.
                Does not support missing values in dataset.
                This makes only sense if the underlying algorithm makes use of
                values aligned in time.
            * PyTorch Sampler instance: any PyTorch sampler,
                e.g., ``the WeightedRandomSampler()``
            * None: samples are taken randomly from times series.

        **kwargs: additional arguments passed to ``DataLoader`` constructor

        Returns
        -------
        DataLoader: dataloader that returns Tuple.
            First entry is ``x``, a dictionary of tensors with the entries,
            and shapes in brackets.

            * encoder_cat : long (batch_size x n_encoder_time_steps x n_features)
                long tensor of encoded categoricals for encoder
            * encoder_cont : float (batch_size x n_encoder_time_steps x n_features)
                float tensor of scaled continuous variables for encoder
            * encoder_target : float (batch_size x n_encoder_time_steps) or list thereof
                if list, each entry for a different target.
                float tensor with unscaled continous target
                or encoded categorical target,
                list of tensors for multiple targets
            * encoder_lengths : long (batch_size)
                long tensor with lengths of the encoder time series. No entry will
                be greater than n_encoder_time_steps
            * decoder_cat : long (batch_size x n_decoder_time_steps x n_features)
                long tensor of encoded categoricals for decoder
            * decoder_cont : float (batch_size x n_decoder_time_steps x n_features)
                float tensor of scaled continuous variables for decoder
            * decoder_target : float (batch_size x n_decoder_time_steps) or list thereof
                if list, with each entry for a different target.
                float tensor with unscaled continous target or encoded categorical
                target for decoder
                - this corresponds to first entry of ``y``,
                list of tensors for multiple targets
            * decoder_lengths : long (batch_size)
                long tensor with lengths of the decoder time series. No entry will
                be greater than n_decoder_time_steps
            * group_ids : float (batch_size x number_of_ids)
                encoded group ids that identify a time series in the dataset
            * target_scale : float (batch_size x scale_size) or list thereof.
                if list, with each entry for a different target.
                parameters used to normalize the target.
                Typically these are mean and standard deviation.
                Is list of tensors for multiple targets.

            Second entry is ``y``, a tuple of the form (``target``, `weight`)

            * target : float (batch_size x n_decoder_time_steps) or list thereof
                if list, with each entry for a different target.
                unscaled (continuous) or encoded (categories) targets,
                list of tensors for multiple targets
            * weight : None or float (batch_size x n_decoder_time_steps)
                weights for each target, None if no weight is used (= equal weights)

        Example
        -------
        Weight by samples for training:

        .. code-block:: python

            from torch.utils.data import WeightedRandomSampler

            # length of probabilties for sampler have to be equal to the length of index
            probabilities = np.sqrt(1 + data.loc[dataset.index, "target"])
            sampler = WeightedRandomSampler(probabilities, len(probabilities))
            dataset.to_dataloader(train=True, sampler=sampler, shuffle=False)
        """
        default_kwargs = dict(
            shuffle=train,
            drop_last=train and len(self) > batch_size,
            collate_fn=self._collate_fn,
            batch_size=batch_size,
            batch_sampler=batch_sampler,
        )
        default_kwargs.update(kwargs)
        kwargs = default_kwargs
        if kwargs["batch_sampler"] is not None:
            sampler = kwargs["batch_sampler"]
            if isinstance(sampler, str):
                if sampler == "synchronized":
                    kwargs["batch_sampler"] = TimeSynchronizedBatchSampler(
                        SequentialSampler(self),
                        batch_size=kwargs["batch_size"],
                        shuffle=kwargs["shuffle"],
                        drop_last=kwargs["drop_last"],
                    )
                else:
                    raise ValueError(
                        f"batch_sampler {sampler} unknown - "
                        "see docstring for valid batch_sampler"
                    )
            del kwargs["batch_size"]
            del kwargs["shuffle"]
            del kwargs["drop_last"]

        return DataLoader(
            self,
            **kwargs,
        )

    def x_to_index(self, x: dict[str, torch.Tensor]) -> pd.DataFrame:
        """
        Decode dataframe index from x.

        Returns:
            dataframe with time index column for first prediction and group ids
        """
        index_data = {self.time_idx: x["decoder_time_idx"][:, 0].cpu()}
        for id in self.group_ids:
            index_data[id] = x["groups"][:, self.group_ids.index(id)].cpu()
            # decode if possible
            index_data[id] = self.transform_values(
                id, index_data[id], inverse=True, group_id=True
            )
        index = pd.DataFrame(index_data)
        return index

    def __repr__(self) -> str:
        return repr_class(
            self,
            attributes=self.get_parameters(),
            extra_attributes=dict(length=len(self)),
        )
