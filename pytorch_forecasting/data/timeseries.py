"""
Timeseries datasets.

Timeseries data is special and has to be processed and fed to algorithms in a special way. This module
defines a class that is able to handle a wide variety of timeseries data problems.
"""
from copy import deepcopy
from functools import lru_cache
import inspect
from typing import Any, Dict, List, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.type_check import nan_to_num
import pandas as pd
from pandas.core.algorithms import isin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.validation import check_is_fitted
import torch
from torch.distributions import Beta
from torch.nn.utils import rnn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    GroupNormalizer,
    MultiNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)


def _find_end_indices(diffs: np.ndarray, max_lengths: np.ndarray, min_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify end indices in series even if some values are missing.

    Args:
        diffs (np.ndarray): array of differences to next time step. nans should be filled up with ones
        max_lengths (np.ndarray): maximum length of sequence by position.
        min_length (int): minimum length of sequence.

    Returns:
        Tuple[np.ndarray, np.ndarray]: tuple of arrays where first is end indices and second is list of start
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


def check_for_nonfinite(tensor: torch.Tensor, names: Union[str, List[str]]) -> torch.Tensor:
    """
    Check if 2D tensor contains NAs or inifinite values.

    Args:
        names (Union[str, List[str]]): name(s) of column(s) (used for error messages)
        tensor (torch.Tensor): tensor to check

    Returns:
        torch.Tensor: returns tensor if checks yield no issues
    """
    if isinstance(names, str):
        names = [names]
        assert tensor.ndim == 1
        nans = (~torch.isfinite(tensor).unsqueeze(-1)).sum(0)
    else:
        assert tensor.ndim == 2
        nans = (~torch.isfinite(tensor)).sum(0)
    for name, na in zip(names, nans):
        if na > 0:
            raise ValueError(
                f"{na} ({na/tensor.size(0):.2%}) of {name} "
                "values were found to be NA or infinite (even after encoding). NA values are not allowed "
                "`allow_missings` refers to missing rows, not to missing values. Possible strategies to "
                f"fix the issue are (a) dropping the variable {name}, "
                "(b) using `NaNLabelEncoder(add_nan=True)` for categorical variables, "
                "(c) filling missing values and/or (d) optionally adding a variable indicating filled values"
            )
    return tensor


class TimeSeriesDataSet(Dataset):
    """
    PyTorch Dataset for fitting timeseries models.

    The dataset automates common tasks such as

    * scaling and encoding of variables
    * normalizing the target variable
    * efficiently converting timeseries in pandas dataframes to torch tensors
    * holding information about static and time-varying variables known and unknown in the future
    * holiding information about related categories (such as holidays)
    * downsampling for data augmentation
    * generating inference, validation and test datasets
    * etc.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        time_idx: str,
        target: Union[str, List[str]],
        group_ids: List[str],
        weight: Union[str, None] = None,
        max_encoder_length: int = 30,
        min_encoder_length: int = None,
        min_prediction_idx: int = None,
        min_prediction_length: int = None,
        max_prediction_length: int = 1,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_known_categoricals: List[str] = [],
        time_varying_known_reals: List[str] = [],
        time_varying_unknown_categoricals: List[str] = [],
        time_varying_unknown_reals: List[str] = [],
        variable_groups: Dict[str, List[int]] = {},
        dropout_categoricals: List[str] = [],
        constant_fill_strategy: Dict[str, Union[str, float, int, bool]] = {},
        allow_missings: bool = False,
        lags: Dict[str, List[int]] = {},
        add_relative_time_idx: bool = False,
        add_target_scales: bool = False,
        add_encoder_length: Union[bool, str] = "auto",
        target_normalizer: Union[TorchNormalizer, NaNLabelEncoder, EncoderNormalizer, str] = "auto",
        categorical_encoders: Dict[str, NaNLabelEncoder] = {},
        scalers: Dict[str, Union[StandardScaler, RobustScaler, TorchNormalizer, EncoderNormalizer]] = {},
        randomize_length: Union[None, Tuple[float, float], bool] = False,
        predict_mode: bool = False,
    ):
        """
        Timeseries dataset holding data for models.

        The :ref:`tutorial on passing data to models <passing-data>` is helpful to understand the output of the dataset
        and how it is coupled to models.

        Each sample is a subsequence of a full time series. The subsequence consists of encoder and decoder/prediction
        timepoints for a given time series. This class constructs an index which defined which subsequences exists and
        can be samples from (``index`` attribute). The samples in the index are defined by by the various parameters.
        to the class (encoder and prediction lengths, minimum prediction length, randomize length and predict keywords).
        How samples are
        sampled into batches for training, is determined by the DataLoader. The class provides the
        :py:meth:`~TimeSeriesDataSet.to_dataloader` method to convert the dataset into a dataloader.

        Large datasets:

            Currently the class is limited to in-memory operations (that can be sped up by an
            existing installation of `numba <https://pypi.org/project/numba/>`_). If you have extremely large data,
            however, you can pass prefitted encoders and and scalers to it and a subset of sequences to the class to
            construct a valid dataset (plus, likely the EncoderNormalizer should be used to normalize targets).
            when fitting a network, you would then to create a custom DataLoader that rotates through the datasets.
            There is currently no in-built methods to do this.

        Args:
            data (pd.DataFrame): dataframe with sequence data - each row can be identified with
                ``time_idx`` and the ``group_ids``
            time_idx (str): integer column denoting the time index. This columns is used to determine
                the sequence of samples.
                If there no missings observations, the time index should increase by ``+1`` for each subsequent sample.
                The first time_idx for each series does not necessarily have to be ``0`` but any value is allowed.
            target (Union[str, List[str]]): column denoting the target or list of columns denoting the target -
                categorical or continous.
            group_ids (List[str]): list of column names identifying a time series. This means that the ``group_ids``
                identify a sample together with the ``time_idx``. If you have only one timeseries, set this to the
                name of column that is constant.
            weight (str): column name for weights. Defaults to None.
            max_encoder_length (int): maximum length to encode
            min_encoder_length (int): minimum allowed length to encode. Defaults to max_encoder_length.
            min_prediction_idx (int): minimum ``time_idx`` from where to start predictions. This parameter
                can be useful to create a validation or test set.
            max_prediction_length (int): maximum prediction/decoder length (choose this not too short as it can help
                convergence)
            min_prediction_length (int): minimum prediction/decoder length. Defaults to max_prediction_length
            static_categoricals (List[str]): list of categorical variables that do not change over time,
                entries can be also lists which are then encoded together
                (e.g. useful for product categories)
            static_reals (List[str]): list of continuous variables that do not change over time
            time_varying_known_categoricals (List[str]): list of categorical variables that change over
                time and are know in the future, entries can be also lists which are then encoded together
                (e.g. useful for special days or promotion categories)
            time_varying_known_reals (List[str]): list of continuous variables that change over
                time and are know in the future
            time_varying_unknown_categoricals (List[str]): list of categorical variables that change over
                time and are not know in the future, entries can be also lists which are then encoded together
                (e.g. useful for weather categories)
            time_varying_unknown_reals (List[str]): list of continuous variables that change over
                time and are not know in the future
            variable_groups (Dict[str, List[str]]): dictionary mapping a name to a list of columns in the data.
                The name should be present
                in a categorical or real class argument, to be able to encode or scale the columns by group.
            dropout_categoricals (List[str]): list of categorical variables that are unknown when making a
                forecast without observed history
            constant_fill_strategy (Dict[str, Union[str, float, int, bool]]): dictionary of column names with
                constants to fill in missing values if there are
                gaps in the sequence (by default forward fill strategy is used). The values will be only used if
                ``allow_missings=True``. A common use case is to denote that demand was 0 if the sample is not in
                the dataset.
            allow_missings (bool): if to allow missing timesteps that are automatically filled up. Missing values
                refer to gaps in the ``time_idx``, e.g. if a specific timeseries has only samples for
                1, 2, 4, 5, the sample for 3 will be generated on-the-fly.
                Allow missings does not deal with ``NA`` values. You should fill NA values before
                passing the dataframe to the TimeSeriesDataSet.
            lags (Dict[str, List[int]]): dictionary of variable names mapped to list of time steps by
                which the variable should be lagged.
                Lags can be useful to indicate seasonality to the models. If you know the seasonalit(ies) of your data,
                add at least the target variables with the corresponding lags to improve performance.
                Lags must be at not larger than the shortest time series as all time series will be cut by the largest
                lag value to prevent NA values.
                Defaults to no lags.
            add_relative_time_idx (bool): if to add a relative time index as feature (i.e. for each sampled sequence,
                the index will range from -encoder_length to prediction_length)
            add_target_scales (bool): if to add scales for target to static real features (i.e. add the center and scale
                of the unnormalized timeseries as features)
            add_encoder_length (bool): if to add decoder length to list of static real variables.
                Defaults to "auto", i.e. yes if ``min_encoder_length != max_encoder_length``.
            target_normalizer (Union[TorchNormalizer, NaNLabelEncoder, EncoderNormalizer, str]): transformer that take
                group_ids, target and time_idx to return normalized targets.
                You can choose from :py:class:`~TorchNormalizer`, :py:class:`~NaNLabelEncoder`,
                :py:class:`~EncoderNormalizer` or `None` for using not normalizer.
                By default an appropriate normalizer is chosen automatically.
            categorical_encoders (Dict[str, NaNLabelEncoder]): dictionary of scikit learn label transformers.
                If you have unobserved categories in
                the future, you can use the :py:class:`~pytorch_forecasting.encoders.NaNLabelEncoder` with
                ``add_nan=True``. Defaults effectively to sklearn's ``LabelEncoder()``. Prefittet encoders will not
                be fit again.
            scalers (Dict[str, Union[StandardScaler, RobustScaler, TorchNormalizer, EncoderNormalizer]]): dictionary of
                scikit-learn scalers. Defaults to sklearn's ``StandardScaler()``.
                Other options are :py:class:`~pytorch_forecasting.data.encoders.EncoderNormalizer`,
                :py:class:`~pytorch_forecasting.data.encoders.GroupNormalizer` or scikit-learn's ``StandarScaler()``,
                ``RobustScaler()`` or `None` for using no normalizer / normalizer with `center=0` and `scale=1`
                (`method="identity"`).
                Prefittet encoders will not be fit again (with the exception of the
                :py:class:`~pytorch_forecasting.data.encoders.EncoderNormalizer` that is fit on every encoder sequence).
            randomize_length (Union[None, Tuple[float, float], bool]): None or False if not to randomize lengths.
                Tuple of beta distribution concentrations from which
                probabilities are sampled that are used to sample new sequence lengths with a binomial
                distribution.
                If True, defaults to (0.2, 0.05), i.e. ~1/4 of samples around minimum encoder length.
                Defaults to False otherwise.
            predict_mode (bool): if to only iterate over each timeseries once (only the last provided samples).
                Effectively, this will take choose for each time series identified by ``group_ids``
                the last ``max_prediction_length`` samples of each time series as
                prediction samples and everthing previous up to ``max_encoder_length`` samples as encoder samples.
        """
        super().__init__()
        self.max_encoder_length = max_encoder_length
        assert isinstance(self.max_encoder_length, int), "max encoder length must be integer"
        if min_encoder_length is None:
            min_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        assert (
            self.min_encoder_length <= self.max_encoder_length
        ), "max encoder length has to be larger equals min encoder length"
        assert isinstance(self.min_encoder_length, int), "min encoder length must be integer"
        self.max_prediction_length = max_prediction_length
        assert isinstance(self.max_prediction_length, int), "max prediction length must be integer"
        if min_prediction_length is None:
            min_prediction_length = max_prediction_length
        self.min_prediction_length = min_prediction_length
        assert (
            self.min_prediction_length <= self.max_prediction_length
        ), "max prediction length has to be larger equals min prediction length"
        assert self.min_prediction_length > 0, "min prediction length must be larger than 0"
        assert isinstance(self.min_prediction_length, int), "min prediction length must be integer"
        self.target = target
        self.weight = weight
        self.time_idx = time_idx
        self.group_ids = [] + group_ids
        self.static_categoricals = [] + static_categoricals
        self.static_reals = [] + static_reals
        self.time_varying_known_categoricals = [] + time_varying_known_categoricals
        self.time_varying_known_reals = [] + time_varying_known_reals
        self.time_varying_unknown_categoricals = [] + time_varying_unknown_categoricals
        self.time_varying_unknown_reals = [] + time_varying_unknown_reals
        self.dropout_categoricals = [] + dropout_categoricals
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
        self.constant_fill_strategy = {} if len(constant_fill_strategy) == 0 else constant_fill_strategy
        self.predict_mode = predict_mode
        self.allow_missings = allow_missings
        self.target_normalizer = target_normalizer
        self.categorical_encoders = {} if len(categorical_encoders) == 0 else categorical_encoders
        self.scalers = {} if len(scalers) == 0 else scalers
        self.add_target_scales = add_target_scales
        self.variable_groups = {} if len(variable_groups) == 0 else variable_groups
        self.lags = {} if len(lags) == 0 else lags

        # add_encoder_length
        if isinstance(add_encoder_length, str):
            assert (
                add_encoder_length == "auto"
            ), f"Only 'auto' allowed for add_encoder_length but found {add_encoder_length}"
            add_encoder_length = self.min_encoder_length != self.max_encoder_length
        assert isinstance(
            add_encoder_length, bool
        ), f"add_encoder_length should be boolean or 'auto' but found {add_encoder_length}"
        self.add_encoder_length = add_encoder_length

        # target normalizer
        self._set_target_normalizer(data)

        # overwrite values
        self.reset_overwrite_values()

        for target in self.target_names:
            assert (
                target not in self.time_varying_known_reals
            ), f"target {target} should be an unknown continuous variable in the future"

        # validate
        self._validate_data(data)
        assert data.index.is_unique, "data index has to be unique"

        # add time index relative to prediction position
        if self.add_relative_time_idx:
            assert (
                "relative_time_idx" not in data.columns
            ), "relative_time_idx is a protected column and must not be present in data"
            if "relative_time_idx" not in self.time_varying_known_reals and "relative_time_idx" not in self.reals:
                self.time_varying_known_reals.append("relative_time_idx")
            data["relative_time_idx"] = 0.0  # dummy - real value will be set dynamiclly in __getitem__()

        # add decoder length to static real variables
        if self.add_encoder_length:
            assert (
                "encoder_length" not in data.columns
            ), "encoder_length is a protected column and must not be present in data"
            if "encoder_length" not in self.time_varying_known_reals and "encoder_length" not in self.reals:
                self.static_reals.append("encoder_length")
            data["encoder_length"] = 0  # dummy - real value will be set dynamiclly in __getitem__()

        # add lags
        assert self.min_lag > 0, "lags should be positive"
        if len(self.lags) > 0:
            # add variables
            for name in self.lags:
                lagged_names = self._get_lagged_names(name)
                for lagged_name in lagged_names:
                    assert (
                        lagged_name not in data.columns
                    ), f"{lagged_name} is a protected column and must not be present in data"
                # add lags
                if name in self.static_reals:
                    for lagged_name in lagged_names:
                        if lagged_name not in self.static_reals:
                            self.static_reals.append(lagged_name)
                elif name in self.static_categoricals:
                    for lagged_name in lagged_names:
                        if lagged_name not in self.static_categoricals:
                            self.static_categoricals.append(lagged_name)
                elif name in self.time_varying_known_reals:
                    for lagged_name in lagged_names:
                        if lagged_name not in self.time_varying_known_reals:
                            self.time_varying_known_reals.append(lagged_name)
                elif name in self.time_varying_known_categoricals:
                    for lagged_name in lagged_names:
                        if lagged_name not in self.time_varying_known_categoricals:
                            self.time_varying_known_categoricals.append(lagged_name)
                elif name in self.time_varying_unknown_reals:
                    for lagged_name, lag in lagged_names.items():
                        if lag < self.max_prediction_length:  # keep in unknown as if lag is too small
                            if lagged_name not in self.time_varying_unknown_reals:
                                self.time_varying_unknown_reals.append(lagged_name)
                        else:
                            if lagged_name not in self.time_varying_known_reals:
                                # switch to known so that lag can be used in decoder directly
                                self.time_varying_known_reals.append(lagged_name)
                elif name in self.time_varying_unknown_categoricals:
                    for lagged_name, lag in lagged_names.items():
                        if lag < self.max_prediction_length:  # keep in unknown as if lag is too small
                            if lagged_name not in self.time_varying_unknown_categoricals:
                                self.time_varying_unknown_categoricals.append(lagged_name)
                        if lagged_name not in self.time_varying_known_categoricals:
                            # switch to known so that lag can be used in decoder directly
                            self.time_varying_known_categoricals.append(lagged_name)
                else:
                    raise KeyError(f"lagged variable {name} is not a static, nor encoder or decoder variable")

        # filter data
        if min_prediction_idx is not None:
            # filtering for min_prediction_idx will be done on subsequence level ensuring
            # minimal decoder index is always >= min_prediction_idx
            data = data[lambda x: x[self.time_idx] >= self.min_prediction_idx - self.max_encoder_length - self.max_lag]
        data = data.sort_values(self.group_ids + [self.time_idx])

        # preprocess data
        data = self._preprocess_data(data)
        for target in self.target_names:
            assert target not in self.scalers, "Target normalizer is separate and not in scalers."

        # create index
        self.index = self._construct_index(data, predict_mode=predict_mode)

        # convert to torch tensor for high performance data loading later
        self.data = self._data_to_tensors(data)

    def _get_lagged_names(self, name: str) -> Dict[str, int]:
        """
        Generate names for lagged variables

        Args:
            name (str): name of variable to lag

        Returns:
            Dict[str, int]: dictionary mapping new variable names to lags
        """
        return {f"{name}_lagged_by_{lag}": lag for lag in self.lags.get(name, [])}

    @property
    @lru_cache(None)
    def lagged_variables(self) -> Dict[str, str]:
        """
        Lagged variables.

        Returns:
            Dict[str, str]: dictionary of variable names corresponding to lagged variables
                mapped to variable that is lagged
        """
        vars = {}
        for name in self.lags:
            vars.update({lag_name: name for lag_name in self._get_lagged_names(name)})
        return vars

    @property
    @lru_cache(None)
    def lagged_targets(self) -> Dict[str, str]:
        """Subset of `lagged_variables` but only includes variables that are lagged targets."""
        vars = {}
        for name in self.lags:
            vars.update({lag_name: name for lag_name in self._get_lagged_names(name) if name in self.target_names})
        return vars

    @property
    @lru_cache(None)
    def min_lag(self) -> int:
        """
        Minimum number of time steps variables are lagged.

        Returns:
            int: minimum lag
        """
        if len(self.lags) == 0:
            return 1e9
        else:
            return min([min(lag) for lag in self.lags.values()])

    @property
    @lru_cache(None)
    def max_lag(self) -> int:
        """
        Maximum number of time steps variables are lagged.

        Returns:
            int: maximum lag
        """
        if len(self.lags) == 0:
            return 0
        else:
            return max([max(lag) for lag in self.lags.values()])

    def _set_target_normalizer(self, data: pd.DataFrame):
        """
        Determine target normalizer.

        Args:
            data (pd.DataFrame): input data
        """
        if isinstance(self.target_normalizer, str) and self.target_normalizer == "auto":
            normalizers = []
            for target in self.target_names:
                if data[target].dtype.kind != "f":  # category
                    normalizers.append(NaNLabelEncoder())
                    if self.add_target_scales:
                        warnings.warn("Target scales will be only added for continous targets", UserWarning)
                else:
                    data_positive = (data[target] > 0).all()
                    if data_positive:
                        if data[target].skew() > 2.5:
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
                self.target_normalizer = MultiNormalizer(normalizers)
            else:
                self.target_normalizer = normalizers[0]
        elif self.target_normalizer is None:
            self.target_normalizer = TorchNormalizer(method="identity")
        assert self.min_encoder_length > 1 or not isinstance(
            self.target_normalizer, EncoderNormalizer
        ), "EncoderNormalizer is only allowed if min_encoder_length > 1"
        assert isinstance(
            self.target_normalizer, (TorchNormalizer, NaNLabelEncoder)
        ), f"target_normalizer has to be either None or of class TorchNormalizer but found {self.target_normalizer}"

    @property
    @lru_cache(None)
    def _group_ids_mapping(self) -> Dict[str, str]:
        """
        Mapping of group id names to group ids used to identify series in dataset -
        group ids can also be used for target normalizer.
        The former can change from training to validation and test dataset while the later must not.
        """
        return {name: f"__group_id__{name}" for name in self.group_ids}

    @property
    @lru_cache(None)
    def _group_ids(self) -> List[str]:
        """
        Group ids used to identify series in dataset.

        See :py:meth:`~TimeSeriesDataSet._group_ids_mapping` for details.
        """
        return list(self._group_ids_mapping.values())

    def _validate_data(self, data: pd.DataFrame):
        """
        Validate that data will not cause hick-ups later on.
        """
        # check for numeric categoricals which can cause hick-ups in logging in tensorboard
        category_columns = data.head(1).select_dtypes("category").columns
        object_columns = data.head(1).select_dtypes(object).columns
        for name in self.flat_categoricals:
            if not (
                name in object_columns
                or (name in category_columns and data[name].cat.categories.dtype.kind not in "bifc")
            ):
                raise ValueError(
                    f"Data type of category {name} was found to be numeric - use a string type / categorified string"
                )
        # check for "." in column names
        columns_with_dot = data.columns[data.columns.str.contains(r"\.")]
        if len(columns_with_dot) > 0:
            raise ValueError(
                f"column names must not contain '.' characters. Names {columns_with_dot.tolist()} are invalid"
            )

    def save(self, fname: str) -> None:
        """
        Save dataset to disk

        Args:
            fname (str): filename to save to
        """
        torch.save(self, fname)

    @classmethod
    def load(cls, fname: str):
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
        for name in self.lags:
            # todo: add support for variable groups
            assert (
                name not in self.variable_groups
            ), f"lagged variables that are in {self.variable_groups} are not supported yet"
            for lagged_name, lag in self._get_lagged_names(name).items():
                data[lagged_name] = data.groupby(self.group_ids, observed=True)[name].shift(lag)

        # encode group ids - this encoding
        for name, group_name in self._group_ids_mapping.items():
            self.categorical_encoders[group_name] = NaNLabelEncoder().fit(data[name].to_numpy().reshape(-1))
            data[group_name] = self.transform_values(name, data[name], inverse=False, group_id=True)

        # encode categoricals first to ensure that group normalizer for relies on encoded categories
        if isinstance(
            self.target_normalizer, (GroupNormalizer, MultiNormalizer)
        ):  # if we use a group normalizer, group_ids must be encoded as well
            group_ids_to_encode = self.group_ids
        else:
            group_ids_to_encode = []
        for name in dict.fromkeys(group_ids_to_encode + self.categoricals):
            if name in self.lagged_variables:
                continue  # do not encode here but only in transform
            allow_nans = name in self.dropout_categoricals
            if name in self.variable_groups:  # fit groups
                columns = self.variable_groups[name]
                if name not in self.categorical_encoders:
                    self.categorical_encoders[name] = NaNLabelEncoder(add_nan=allow_nans).fit(
                        data[columns].to_numpy().reshape(-1)
                    )
                elif self.categorical_encoders[name] is not None:
                    try:
                        check_is_fitted(self.categorical_encoders[name])
                    except NotFittedError:
                        self.categorical_encoders[name] = self.categorical_encoders[name].fit(
                            data[columns].to_numpy().reshape(-1)
                        )
            else:
                if name not in self.categorical_encoders:
                    self.categorical_encoders[name] = NaNLabelEncoder(add_nan=allow_nans).fit(data[name])
                elif self.categorical_encoders[name] is not None and name not in self.target_names:
                    try:
                        check_is_fitted(self.categorical_encoders[name])
                    except NotFittedError:
                        self.categorical_encoders[name] = self.categorical_encoders[name].fit(data[name])

        # encode them
        for name in dict.fromkeys(group_ids_to_encode + self.flat_categoricals):
            # targets and its lagged versions are handled separetely
            if name not in self.target_names and name not in self.lagged_targets:
                data[name] = self.transform_values(
                    name, data[name], inverse=False, ignore_na=name in self.lagged_variables
                )

        # save special variables
        assert "__time_idx__" not in data.columns, "__time_idx__ is a protected column and must not be present in data"
        data["__time_idx__"] = data[self.time_idx]  # save unscaled
        for target in self.target_names:
            assert (
                f"__target__{target}" not in data.columns
            ), f"__target__{target} is a protected column and must not be present in data"
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
                elif isinstance(self.target_normalizer, (GroupNormalizer, MultiNormalizer)):
                    self.target_normalizer.fit(data[self.target], data)
                else:
                    self.target_normalizer.fit(data[self.target])

            # transform target
            if isinstance(self.target_normalizer, EncoderNormalizer):
                # we approximate the scales and target transformation by assuming one
                # transformation over the entire time range but by each group
                common_init_args = [
                    name
                    for name in inspect.signature(GroupNormalizer.__init__).parameters.keys()
                    if name in inspect.signature(EncoderNormalizer.__init__).parameters.keys()
                    and name not in ["data", "self"]
                ]
                copy_kwargs = {name: getattr(self.target_normalizer, name) for name in common_init_args}
                normalizer = GroupNormalizer(groups=self.group_ids, **copy_kwargs)
                data[self.target], scales = normalizer.fit_transform(data[self.target], data, return_norm=True)

            elif isinstance(self.target_normalizer, GroupNormalizer):
                data[self.target], scales = self.target_normalizer.transform(data[self.target], data, return_norm=True)

            elif isinstance(self.target_normalizer, MultiNormalizer):
                transformed, scales = self.target_normalizer.transform(data[self.target], data, return_norm=True)

                for idx, target in enumerate(self.target_names):
                    data[target] = transformed[idx]

                    if isinstance(self.target_normalizer[idx], NaNLabelEncoder):
                        # overwrite target because it requires encoding (continuous targets should not be normalized)
                        data[f"__target__{target}"] = data[target]

            elif isinstance(self.target_normalizer, NaNLabelEncoder):
                data[self.target] = self.target_normalizer.transform(data[self.target])
                # overwrite target because it requires encoding (continuous targets should not be normalized)
                data[f"__target__{self.target}"] = data[self.target]
                scales = None

            else:
                data[self.target], scales = self.target_normalizer.transform(data[self.target], return_norm=True)

            # add target scales
            if self.add_target_scales:
                if not isinstance(self.target_normalizer, MultiNormalizer):
                    scales = [scales]
                for target_idx, target in enumerate(self.target_names):
                    if not isinstance(self.target_normalizers[target_idx], NaNLabelEncoder):
                        for scale_idx, name in enumerate(["center", "scale"]):
                            feature_name = f"{target}_{name}"
                            assert (
                                feature_name not in data.columns
                            ), f"{feature_name} is a protected column and must not be present in data"
                            data[feature_name] = scales[target_idx][:, scale_idx].squeeze()
                            if feature_name not in self.reals:
                                self.static_reals.append(feature_name)

        # rescale continuous variables apart from target
        for name in self.reals:
            if name in self.target_names or name in self.lagged_variables:
                # lagged variables are only transformed - not fitted
                continue
            elif name not in self.scalers:
                self.scalers[name] = StandardScaler().fit(data[[name]])
            elif self.scalers[name] is not None:
                try:
                    check_is_fitted(self.scalers[name])
                except NotFittedError:
                    if isinstance(self.scalers[name], GroupNormalizer):
                        self.scalers[name] = self.scalers[name].fit(data[[name]], data)
                    else:
                        self.scalers[name] = self.scalers[name].fit(data[[name]])

        # encode after fitting
        for name in self.reals:
            # targets are handled separately
            transformer = self.get_transformer(name)
            if (
                name not in self.target_names
                and transformer is not None
                and not isinstance(transformer, EncoderNormalizer)
            ):
                data[name] = self.transform_values(name, data[name], data=data, inverse=False)

        # encode lagged categorical targets
        for name in self.lagged_targets:
            # normalizer only now available
            if name in self.flat_categoricals:
                data[name] = self.transform_values(name, data[name], inverse=False, ignore_na=True)

        # encode constant values
        self.encoded_constant_fill_strategy = {}
        for name, value in self.constant_fill_strategy.items():
            if name in self.target_names:
                self.encoded_constant_fill_strategy[f"__target__{name}"] = value
            self.encoded_constant_fill_strategy[name] = self.transform_values(
                name, np.array([value]), data=data, inverse=False
            )[0]

        # shorten data by maximum of lagged sequences to avoid NA values - shorten only after encoding
        if self.max_lag > 0:
            # negative tail implementation as .groupby().tail(-self.max_lag) is not implemented in pandas
            g = data.groupby(self._group_ids, observed=True)
            data = g._selected_obj[g.cumcount() >= self.max_lag]
        return data

    def get_transformer(self, name: str, group_id: bool = False):
        """
        Get transformer for variable.

        Args:
            name (str): variable name
            group_id (bool, optional): If the passed name refers to a group id (different encoders are used for these).
                Defaults to False.

        Returns:
            transformer
        """
        if group_id:
            name = self._group_ids_mapping[name]
        elif name in self.lagged_variables:  # recover transformer fitted on non-lagged variable
            name = self.lagged_variables[name]

        if name in self.flat_categoricals + self.group_ids + self._group_ids:
            name = self.variable_to_group_mapping.get(name, name)  # map name to encoder

            # take target normalizer if required
            if name in self.target_names:
                transformer = self.target_normalizers[self.target_names.index(name)]
            else:
                transformer = self.categorical_encoders.get(name, None)
            return transformer

        elif name in self.reals:
            # take target normalizer if required
            if name in self.target_names:
                transformer = self.target_normalizers[self.target_names.index(name)]
            else:
                transformer = self.scalers.get(name, None)
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
        """
        Scale and encode values.

        Args:
            name (str): name of variable
            values (Union[pd.Series, torch.Tensor, np.ndarray]): values to encode/scale
            data (pd.DataFrame, optional): extra data used for scaling (e.g. dataframe with groups columns).
                Defaults to None.
            inverse (bool, optional): if to conduct inverse transformation. Defaults to False.
            group_id (bool, optional): If the passed name refers to a group id (different encoders are used for these).
                Defaults to False.
            **kwargs: additional arguments for transform/inverse_transform method

        Returns:
            np.ndarray: (de/en)coded/(de)scaled values
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

    def _data_to_tensors(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Convert data to tensors for faster access with :py:meth:`~__getitem__`.

        Args:
            data (pd.DataFrame): preprocessed data

        Returns:
            Dict[str, torch.Tensor]: dictionary of tensors for continous, categorical data, groups, target and
                time index
        """

        index = check_for_nonfinite(
            torch.tensor(data[self._group_ids].to_numpy(np.long), dtype=torch.long), self.group_ids
        )
        time = check_for_nonfinite(
            torch.tensor(data["__time_idx__"].to_numpy(np.long), dtype=torch.long), self.time_idx
        )

        # categorical covariates
        categorical = check_for_nonfinite(
            torch.tensor(data[self.flat_categoricals].to_numpy(np.long), dtype=torch.long), self.flat_categoricals
        )

        # get weight
        if self.weight is not None:
            weight = check_for_nonfinite(
                torch.tensor(
                    data["__weight__"].to_numpy(dtype=np.float),
                    dtype=torch.float,
                ),
                self.weight,
            )
        else:
            weight = None

        # get target
        if isinstance(self.target_normalizer, NaNLabelEncoder):
            target = [
                check_for_nonfinite(
                    torch.tensor(data[f"__target__{self.target}"].to_numpy(dtype=np.long), dtype=torch.long),
                    self.target,
                )
            ]
        else:
            if not isinstance(self.target, str):  # multi-target
                target = [
                    check_for_nonfinite(
                        torch.tensor(
                            data[f"__target__{name}"].to_numpy(
                                dtype=[np.float, np.long][data[name].dtype.kind in "bi"]
                            ),
                            dtype=[torch.float, torch.long][data[name].dtype.kind in "bi"],
                        ),
                        name,
                    )
                    for name in self.target_names
                ]
            else:
                target = [
                    check_for_nonfinite(
                        torch.tensor(data[f"__target__{self.target}"].to_numpy(dtype=np.float), dtype=torch.float),
                        self.target,
                    )
                ]

        # continuous covariates
        continuous = check_for_nonfinite(
            torch.tensor(data[self.reals].to_numpy(dtype=np.float), dtype=torch.float), self.reals
        )

        tensors = dict(
            reals=continuous, categoricals=categorical, groups=index, target=target, weight=weight, time=time
        )

        return tensors

    @property
    def categoricals(self) -> List[str]:
        """
        Categorical variables as used for modelling.

        Returns:
            List[str]: list of variables
        """
        return self.static_categoricals + self.time_varying_known_categoricals + self.time_varying_unknown_categoricals

    @property
    def flat_categoricals(self) -> List[str]:
        """
        Categorical variables as defined in input data.

        Returns:
            List[str]: list of variables
        """
        categories = []
        for name in self.categoricals:
            if name in self.variable_groups:
                categories.extend(self.variable_groups[name])
            else:
                categories.append(name)
        return categories

    @property
    def variable_to_group_mapping(self) -> Dict[str, str]:
        """
        Mapping from categorical variables to variables in input data.

        Returns:
            Dict[str, str]: dictionary mapping from :py:meth:`~categorical` to :py:meth:`~flat_categoricals`.
        """
        groups = {}
        for group_name, sublist in self.variable_groups.items():
            groups.update({name: group_name for name in sublist})
        return groups

    @property
    def reals(self) -> List[str]:
        """
        Continous variables as used for modelling.

        Returns:
            List[str]: list of variables
        """
        return self.static_reals + self.time_varying_known_reals + self.time_varying_unknown_reals

    @property
    @lru_cache(None)
    def target_names(self) -> List[str]:
        """
        List of targets.

        Returns:
            List[str]: list of targets
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
    def target_normalizers(self) -> List[TorchNormalizer]:
        """
        List of target normalizers aligned with ``target_names``.

        Returns:
            List[TorchNormalizer]: list of target normalizers
        """
        if isinstance(self.target_normalizer, MultiNormalizer):
            target_normalizers = self.target_normalizer.normalizers
        else:
            target_normalizers = [self.target_normalizer]
        return target_normalizers

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters that can be used with :py:meth:`~from_parameters` to create a new dataset with the same scalers.

        Returns:
            Dict[str, Any]: dictionary of parameters
        """
        kwargs = {
            name: getattr(self, name)
            for name in inspect.signature(self.__class__.__init__).parameters.keys()
            if name not in ["data", "self"]
        }
        kwargs["categorical_encoders"] = self.categorical_encoders
        kwargs["scalers"] = self.scalers
        return kwargs

    @classmethod
    def from_dataset(
        cls, dataset, data: pd.DataFrame, stop_randomization: bool = False, predict: bool = False, **update_kwargs
    ):
        """
        Generate dataset with different underlying data but same variable encoders and scalers, etc.

        Calls :py:meth:`~from_parameters` under the hood.

        Args:
            dataset (TimeSeriesDataSet): dataset from which to copy parameters
            data (pd.DataFrame): data from which new dataset will be generated
            stop_randomization (bool, optional): If to stop randomizing encoder and decoder lengths,
                e.g. useful for validation set. Defaults to False.
            predict (bool, optional): If to predict the decoder length on the last entries in the
                time index (i.e. one prediction per group only). Defaults to False.
            **kwargs: keyword arguments overriding parameters in the original dataset

        Returns:
            TimeSeriesDataSet: new dataset
        """
        return cls.from_parameters(
            dataset.get_parameters(), data, stop_randomization=stop_randomization, predict=predict, **update_kwargs
        )

    @classmethod
    def from_parameters(
        cls,
        parameters: Dict[str, Any],
        data: pd.DataFrame,
        stop_randomization: bool = False,
        predict: bool = False,
        **update_kwargs,
    ):
        """
        Generate dataset with different underlying data but same variable encoders and scalers, etc.

        Args:
            parameters (Dict[str, Any]): dataset parameters which to use for the new dataset
            data (pd.DataFrame): data from which new dataset will be generated
            stop_randomization (bool, optional): If to stop randomizing encoder and decoder lengths,
                e.g. useful for validation set. Defaults to False.
            predict (bool, optional): If to predict the decoder length on the last entries in the
                time index (i.e. one prediction per group only). Defaults to False.
            **kwargs: keyword arguments overriding parameters

        Returns:
            TimeSeriesDataSet: new dataset
        """
        parameters = deepcopy(parameters)
        if predict:
            if not stop_randomization:
                warnings.warn(
                    "If predicting, no randomization should be possible - setting stop_randomization=True", UserWarning
                )
                stop_randomization = True
            parameters["min_prediction_length"] = parameters["max_prediction_length"]
            parameters["predict_mode"] = True
        if stop_randomization:
            parameters["randomize_length"] = None
        parameters.update(update_kwargs)

        new = cls(data, **parameters)
        return new

    def _construct_index(self, data: pd.DataFrame, predict_mode: bool) -> pd.DataFrame:
        """
        Create index of samples.

        Args:
            data (pd.DataFrame): preprocessed data
            predict_mode (bool): if to create one same per group with prediction length equals ``max_decoder_length``

        Returns:
            pd.DataFrame: index dataframe
        """
        g = data.groupby(self._group_ids, observed=True)

        df_index_first = g["__time_idx__"].transform("nth", 0).to_frame("time_first")
        df_index_last = g["__time_idx__"].transform("nth", -1).to_frame("time_last")
        df_index_diff_to_next = -g["__time_idx__"].diff(-1).fillna(-1).astype(int).to_frame("time_diff_to_next")
        df_index = pd.concat([df_index_first, df_index_last, df_index_diff_to_next], axis=1)
        df_index["index_start"] = np.arange(len(df_index))
        df_index["time"] = data["__time_idx__"]
        df_index["count"] = (df_index["time_last"] - df_index["time_first"]).astype(int) + 1
        group_ids = g.ngroup()
        df_index["group_id"] = group_ids

        min_sequence_length = self.min_prediction_length + self.min_encoder_length
        max_sequence_length = self.max_prediction_length + self.max_encoder_length

        # calculate maximum index to include from current index_start
        max_time = (df_index["time"] + max_sequence_length - 1).clip(upper=df_index["count"] + df_index.time_first - 1)

        # if there are missing timesteps, we cannot say directly what is the last timestep to include
        # therefore we iterate until it is found
        if (df_index["time_diff_to_next"] != 1).any():
            assert (
                self.allow_missings
            ), "Time difference between steps has been idenfied as larger than 1 - set allow_missings=True"

        df_index["index_end"], missing_sequences = _find_end_indices(
            diffs=df_index.time_diff_to_next.to_numpy(),
            max_lengths=(max_time - df_index.time).to_numpy() + 1,
            min_length=min_sequence_length,
        )
        # add duplicates but mostly with shorter sequence length for start of timeseries
        # while the previous steps have ensured that we start a sequence on every time step, the missing_sequences
        # ensure that there is a sequence that finishes on every timestep
        if len(missing_sequences) > 0:
            shortened_sequences = df_index.iloc[missing_sequences[:, 0]].assign(index_end=missing_sequences[:, 1])

            # concatenate shortened sequences
            df_index = pd.concat([df_index, shortened_sequences], axis=0, ignore_index=True)

        # filter out where encode and decode length are not satisfied
        df_index["sequence_length"] = df_index["time"].iloc[df_index["index_end"]].to_numpy() - df_index["time"] + 1

        # filter too short sequences
        df_index = df_index[
            # sequence must be at least of minimal prediction length
            lambda x: (x.sequence_length >= min_sequence_length)
            &
            # prediction must be for after minimal prediction index + length of prediction
            (x["sequence_length"] + x["time"] >= self.min_prediction_idx + self.min_prediction_length)
        ]

        if predict_mode:  # keep longest element per series (i.e. the first element that spans to the end of the series)
            # filter all elements that are longer than the allowed maximum sequence length
            df_index = df_index[
                lambda x: (x["time_last"] - x["time"] + 1 <= max_sequence_length)
                & (x["sequence_length"] >= min_sequence_length)
            ]
            # choose longest sequence
            df_index = df_index.loc[df_index.groupby("group_id").sequence_length.idxmax()]

        # check that all groups/series have at least one entry in the index
        if not group_ids.isin(df_index.group_id).all():
            missing_groups = data.loc[~group_ids.isin(df_index.group_id), self._group_ids].drop_duplicates()
            # decode values
            for name, id in self._group_ids_mapping.items():
                missing_groups[id] = self.transform_values(name, missing_groups[id], inverse=True, group_id=True)
            warnings.warn(
                "Min encoder length and/or min_prediction_idx and/or min prediction length and/or lags are "
                "too large for "
                f"{len(missing_groups)} series/groups which therefore are not present in the dataset index. "
                "This means no predictions can be made for those series. "
                f"First 10 removed groups: {list(missing_groups.iloc[:10].to_dict(orient='index').values())}",
                UserWarning,
            )
        assert (
            len(df_index) > 0
        ), "filters should not remove entries all entries - check encoder/decoder lengths and lags"

        return df_index

    def plot_randomization(
        self, betas: Tuple[float, float] = None, length: int = None, min_length: int = None
    ) -> Tuple[plt.Figure, torch.Tensor]:
        """
        Plot expected randomized length distribution.

        Args:
            betas (Tuple[float, float], optional): Tuple of betas, e.g. ``(0.2, 0.05)`` to use for randomization.
                Defaults to ``randomize_length`` of dataset.
            length (int, optional): . Defaults to ``max_encoder_length``.
            min_length (int, optional): [description]. Defaults to ``min_encoder_length``.

        Returns:
            Tuple[plt.Figure, torch.Tensor]: tuple of figure and histogram based on 1000 samples
        """
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
        self, values: Union[float, torch.Tensor], variable: str, target: Union[str, slice] = "decoder"
    ) -> None:
        """
        Convenience method to quickly overwrite values in decoder or encoder (or both) for a specific variable.

        Args:
            values (Union[float, torch.Tensor]): values to use for overwrite.
            variable (str): variable whose values should be overwritten.
            target (Union[str, slice], optional): positions to overwrite. One of "decoder", "encoder" or "all" or
                a slice object which is directly used to overwrite indices, e.g. ``slice(-5, None)`` will overwrite
                the last 5 values. Defaults to "decoder".
        """
        values = torch.tensor(self.transform_values(variable, np.asarray(values).reshape(-1), inverse=False)).squeeze()
        assert target in [
            "all",
            "decoder",
            "encoder",
        ], f"target has be one of 'all', 'decoder' or 'encoder' but target={target} instead"

        if variable in self.static_categoricals or variable in self.static_categoricals:
            target = "all"

        if variable in self.target_names:
            raise NotImplementedError("Target variable is not supported")
        if self.weight is not None and self.weight == variable:
            raise NotImplementedError("Weight variable is not supported")
        if isinstance(self.scalers.get(variable, self.categorical_encoders.get(variable)), TorchNormalizer):
            raise NotImplementedError("TorchNormalizer (e.g. GroupNormalizer) is not supported")

        if self._overwrite_values is None:
            self._overwrite_values = {}
        self._overwrite_values.update(dict(values=values, variable=variable, target=target))

    def reset_overwrite_values(self) -> None:
        """
        Reset values used to override sample features.
        """
        self._overwrite_values = None

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get sample for model

        Args:
            idx (int): index of prediction (between ``0`` and ``len(dataset) - 1``)

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: x and y for model
        """
        index = self.index.iloc[idx]
        # get index data
        data_cont = self.data["reals"][index.index_start : index.index_end + 1].clone()
        data_cat = self.data["categoricals"][index.index_start : index.index_end + 1].clone()
        time = self.data["time"][index.index_start : index.index_end + 1].clone()
        target = [d[index.index_start : index.index_end + 1].clone() for d in self.data["target"]]
        groups = self.data["groups"][index.index_start].clone()
        if self.data["weight"] is None:
            weight = None
        else:
            weight = self.data["weight"][index.index_start : index.index_end + 1].clone()

        # get target scale in the form of a list
        target_scale = self.target_normalizer.get_parameters(groups, self.group_ids)
        if not isinstance(self.target_normalizer, MultiNormalizer):
            target_scale = [target_scale]

        # fill in missing values (if not all time indices are specified
        sequence_length = len(time)
        if sequence_length < index.sequence_length:
            assert self.allow_missings, "allow_missings should be True if sequences have gaps"
            repetitions = torch.cat([time[1:] - time[:-1], torch.ones(1, dtype=time.dtype)])
            indices = torch.repeat_interleave(torch.arange(len(time)), repetitions)
            repetition_indices = torch.cat([torch.tensor([False], dtype=torch.bool), indices[1:] == indices[:-1]])

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
                    data_cont[0, time_idx], data_cont[-1, time_idx], len(target[0]), dtype=data_cont.dtype
                )

            # make replacements to fill in categories
            for name, value in self.encoded_constant_fill_strategy.items():
                if name in self.reals:
                    data_cont[repetition_indices, self.reals.index(name)] = value
                elif name in [f"__target__{target_name}" for target_name in self.target_names]:
                    target_pos = self.target_names.index(name[len("__target__") :])
                    target[target_pos][repetition_indices] = value
                elif name in self.flat_categoricals:
                    data_cat[repetition_indices, self.flat_categoricals.index(name)] = value
                elif name in self.target_names:  # target is just not an input value
                    pass
                else:
                    raise KeyError(f"Variable {name} is not known and thus cannot be filled in")

            sequence_length = len(target[0])

        # determine data window
        assert (
            sequence_length >= self.min_prediction_length
        ), "Sequence length should be at least minimum prediction length"
        # determine prediction/decode length and encode length
        decoder_length = min(
            time[-1] - (self.min_prediction_idx - 1),
            self.max_prediction_length,
            sequence_length - self.min_encoder_length,
        )
        encoder_length = sequence_length - decoder_length
        assert (
            decoder_length >= self.min_prediction_length
        ), "Decoder length should be at least minimum prediction length"
        assert encoder_length >= self.min_encoder_length, "Encoder length should be at least minimum encoder length"

        if self.randomize_length is not None:  # randomization improves generalization
            # modify encode and decode lengths
            modifiable_encoder_length = encoder_length - self.min_encoder_length
            encoder_length_probability = Beta(self.randomize_length[0], self.randomize_length[1]).sample()

            # subsample a new/smaller encode length
            new_encoder_length = self.min_encoder_length + int(
                (modifiable_encoder_length * encoder_length_probability).round()
            )

            # extend decode length if possible
            new_decoder_length = min(decoder_length + (encoder_length - new_encoder_length), self.max_prediction_length)

            # select subset of sequence of new sequence
            if new_encoder_length + new_decoder_length < len(target[0]):
                data_cat = data_cat[encoder_length - new_encoder_length : encoder_length + new_decoder_length]
                data_cont = data_cont[encoder_length - new_encoder_length : encoder_length + new_decoder_length]
                target = [t[encoder_length - new_encoder_length : encoder_length + new_decoder_length] for t in target]
                encoder_length = new_encoder_length
                decoder_length = new_decoder_length

            # switch some variables to nan if encode length is 0
            if encoder_length == 0 and len(self.dropout_categoricals) > 0:
                data_cat[
                    :, [self.flat_categoricals.index(c) for c in self.dropout_categoricals]
                ] = 0  # zero is encoded nan

        assert decoder_length > 0, "Decoder length should be greater than 0"
        assert encoder_length >= 0, "Encoder length should be at least 0"

        if self.add_relative_time_idx:
            data_cont[:, self.reals.index("relative_time_idx")] = (
                torch.arange(-encoder_length, decoder_length, dtype=data_cont.dtype) / self.max_encoder_length
            )

        if self.add_encoder_length:
            data_cont[:, self.reals.index("encoder_length")] = (
                (encoder_length - 0.5 * self.max_encoder_length) / self.max_encoder_length * 2.0
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
                    data_cont[:, self.reals.index(target_name)] = target_normalizer.transform(target[idx])
                if self.add_target_scales:
                    data_cont[:, self.reals.index(f"{target_name}_center")] = self.transform_values(
                        f"{target_name}_center", single_target_scale[0]
                    )[0]
                    data_cont[:, self.reals.index(f"{target_name}_scale")] = self.transform_values(
                        f"{target_name}_scale", single_target_scale[1]
                    )[0]
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
                assert (
                    self._overwrite_values["variable"] in self.flat_categoricals
                ), "overwrite values variable has to be either in real or categorical variables"
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

    def _collate_fn(
        self, batches: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Collate function to combine items into mini-batch for dataloader.

        Args:
            batches (List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]): List of samples generated with
                :py:meth:`~__getitem__`.

        Returns:
            Tuple[Dict[str, torch.Tensor], Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]: minibatch
        """
        # collate function for dataloader
        # lengths
        encoder_lengths = torch.tensor([batch[0]["encoder_length"] for batch in batches], dtype=torch.long)
        decoder_lengths = torch.tensor([batch[0]["decoder_length"] for batch in batches], dtype=torch.long)

        # ids
        decoder_time_idx_start = (
            torch.tensor([batch[0]["encoder_time_idx_start"] for batch in batches], dtype=torch.long) + encoder_lengths
        )
        decoder_time_idx = decoder_time_idx_start.unsqueeze(1) + torch.arange(decoder_lengths.max()).unsqueeze(0)
        groups = torch.stack([batch[0]["groups"] for batch in batches])

        # features
        encoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][:length] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )
        encoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][:length] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )

        decoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][length:] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )
        decoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][length:] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )

        # target scale
        if isinstance(batches[0][0]["target_scale"], torch.Tensor):  # stack tensor
            target_scale = torch.stack([batch[0]["target_scale"] for batch in batches])
        elif isinstance(batches[0][0]["target_scale"], (list, tuple)):
            target_scale = []
            for idx in range(len(batches[0][0]["target_scale"])):
                if isinstance(batches[0][0]["target_scale"][idx], torch.Tensor):  # stack tensor
                    scale = torch.stack([batch[0]["target_scale"][idx] for batch in batches])
                else:
                    scale = torch.tensor([batch[0]["target_scale"][idx] for batch in batches], dtype=torch.float)
                target_scale.append(scale)
        else:  # convert to tensor
            target_scale = torch.tensor([batch[0]["target_scale"] for batch in batches], dtype=torch.float)

        # target and weight
        if isinstance(batches[0][1][0], (tuple, list)):
            target = [
                rnn.pad_sequence([batch[1][0][idx] for batch in batches], batch_first=True)
                for idx in range(len(batches[0][1][0]))
            ]
            encoder_target = [
                rnn.pad_sequence([batch[0]["encoder_target"][idx] for batch in batches], batch_first=True)
                for idx in range(len(batches[0][1][0]))
            ]
        else:
            target = rnn.pad_sequence([batch[1][0] for batch in batches], batch_first=True)
            encoder_target = rnn.pad_sequence([batch[0]["encoder_target"] for batch in batches], batch_first=True)

        if batches[0][1][1] is not None:
            weight = rnn.pad_sequence([batch[1][1] for batch in batches], batch_first=True)
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
        self, train: bool = True, batch_size: int = 64, batch_sampler: Union[Sampler, str] = None, **kwargs
    ) -> DataLoader:
        """
        Get dataloader from dataset.

        The

        Args:
            train (bool, optional): if dataloader is used for training or prediction
                Will shuffle and drop last batch if True. Defaults to True.
            batch_size (int): batch size for training model. Defaults to 64.
            batch_sampler (Union[Sampler, str]): batch sampler or string. One of

                * "synchronized": ensure that samples in decoder are aligned in time. Does not support missing
                  values in dataset. This makes only sense if the underlying algorithm makes use of values aligned
                  in time.
                * PyTorch Sampler instance: any PyTorch sampler, e.g. the WeightedRandomSampler()
                * None: samples are taken randomly from times series.

            **kwargs: additional arguments to ``DataLoader()``

        Returns:
            DataLoader: dataloader that returns Tuple.
                First entry is ``x``, a dictionary of tensors with the entries (and shapes in brackets)

                * encoder_cat (batch_size x n_encoder_time_steps x n_features): long tensor of encoded
                  categoricals for encoder
                * encoder_cont (batch_size x n_encoder_time_steps x n_features): float tensor of scaled continuous
                  variables for encoder
                * encoder_target (batch_size x n_encoder_time_steps or list thereof with each entry for a different
                  target):
                  float tensor with unscaled continous target or encoded categorical target,
                  list of tensors for multiple targets
                * encoder_lengths (batch_size): long tensor with lengths of the encoder time series. No entry will
                  be greater than n_encoder_time_steps
                * decoder_cat (batch_size x n_decoder_time_steps x n_features): long tensor of encoded
                  categoricals for decoder
                * decoder_cont (batch_size x n_decoder_time_steps x n_features): float tensor of scaled continuous
                  variables for decoder
                * decoder_target (batch_size x n_decoder_time_steps or list thereof with each entry for a different
                  target):
                  float tensor with unscaled continous target or encoded categorical target for decoder
                  - this corresponds to first entry of ``y``, list of tensors for multiple targets
                * decoder_lengths (batch_size): long tensor with lengths of the decoder time series. No entry will
                  be greater than n_decoder_time_steps
                * group_ids (batch_size x number_of_ids): encoded group ids that identify a time series in the dataset
                * target_scale (batch_size x scale_size or list thereof with each entry for a different target):
                  parameters used to normalize the target.
                  Typically these are mean and standard deviation. Is list of tensors for multiple targets.


                Second entry is ``y``, a tuple of the form (``target``, `weight`)

                * target (batch_size x n_decoder_time_steps or list thereof with each entry for a different target):
                  unscaled (continuous) or encoded (categories) targets, list of tensors for multiple targets
                * weight (None or batch_size x n_decoder_time_steps): weight

        Example:

            Weight by samples for training:

            .. code-block:: python

                from torch.utils.data import WeightedRandomSampler

                # length of probabilties for sampler have to be equal to the length of the index
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
                        self, batch_size=kwargs["batch_size"], shuffle=kwargs["shuffle"], drop_last=kwargs["drop_last"]
                    )
                else:
                    raise ValueError(f"batch_sampler {sampler} unknown - see docstring for valid batch_sampler")
            del kwargs["batch_size"]
            del kwargs["shuffle"]
            del kwargs["drop_last"]

        return DataLoader(
            self,
            **kwargs,
        )

    def x_to_index(self, x: Dict[str, torch.Tensor]) -> pd.DataFrame:
        """
        Decode dataframe index from x.

        Returns:
            dataframe with time index column for first prediction and group ids
        """
        index_data = {self.time_idx: x["decoder_time_idx"][:, 0].cpu()}
        for id in self.group_ids:
            index_data[id] = x["groups"][:, self.group_ids.index(id)].cpu()
            # decode if possible
            index_data[id] = self.transform_values(id, index_data[id], inverse=True, group_id=True)
        index = pd.DataFrame(index_data)
        return index


class TimeSynchronizedBatchSampler(Sampler):
    """
    Samples mini-batches randomly but in a time-synchronised manner.

    Time-synchornisation means that the time index of the first decoder samples are aligned across the batch.
    This sampler does not support missing values in the dataset.
    """

    def __init__(
        self,
        data_source: TimeSeriesDataSet,
        batch_size: int = 64,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        """
        Initialize TimeSynchronizedBatchSampler.

        Args:
            data_source (TimeSeriesDataSet): timeseries dataset.
            drop_last (bool): if to drop last mini-batch from a group if it is smaller than batch_size.
                Defaults to False.
            shuffle (bool): if to shuffle dataset. Defaults to False.
            batch_size (int, optional): Number of samples in a mini-batch. This is rather the maximum number
                of samples. Because mini-batches are grouped by prediction time, chances are that there
                are multiple where batch size will be smaller than the maximum. Defaults to 64.
        """
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integer value, " "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got " "drop_last={}".format(drop_last))
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        assert not self.data_source.allow_missings, "allow_missings should be False for time-synchronized mini-batches"

        # construct index from which can be sampled
        self.construct_batch_groups()

    def construct_batch_groups(self):
        """
        Construct index of batches from which can be sampled
        """
        index = self.data_source.index
        # get groups, i.e. group all samples by first predict time
        decoder_lengths = np.min(
            [
                index.time_last - (self.data_source.min_prediction_idx - 1),
                index.sequence_length - self.data_source.min_encoder_length,
            ],
            axis=0,
        ).clip(max=self.data_source.max_prediction_length)
        first_prediction_time = index.time + index.sequence_length - decoder_lengths + 1
        self._groups = pd.RangeIndex(0, len(index.index)).groupby(first_prediction_time)

        # calculate sizes of groups
        self._group_sizes = {}
        warns = []
        for name, group in self._groups.items():  # iterate over groups
            if self.drop_last:
                self._group_sizes[name] = len(group) // self.batch_size
            else:
                self._group_sizes[name] = (len(group) + self.batch_size - 1) // self.batch_size
            if self._group_sizes[name] == 0:
                self._group_sizes[name] = 1
                warns.append(name)
        if len(warns) > 0:
            warnings.warn(
                f"Less than {self.batch_size} samples available for {len(warns)} prediction times. "
                f"Use batch size smaller than {self.batch_size}. "
                f"First 10 prediction times with small batch sizes: {warns[:10]}"
            )
        # create index from which can be sampled: index is equal to number of batches
        # associate index with prediction time
        self._group_index = np.repeat(list(self._group_sizes.keys()), list(self._group_sizes.values()))
        # associate index with batch within prediction time group
        self._sub_group_index = np.concatenate([np.arange(size) for size in self._group_sizes.values()])

    def __iter__(self):
        if self.shuffle:  # shuffle samples
            groups = {name: shuffle(group) for name, group in self._groups.items()}
        else:
            groups = self._groups

        batch_samples = np.random.permutation(len(self))
        for idx in batch_samples:
            name = self._group_index[idx]
            sub_group = self._sub_group_index[idx]
            sub_group_start = sub_group * self.batch_size
            sub_group_end = sub_group_start + self.batch_size
            batch = groups[name][sub_group_start:sub_group_end]
            yield batch

    def __len__(self):
        return len(self._group_index)
