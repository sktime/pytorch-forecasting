"""
Timeseries datasets.

Timeseries data is special and has to be processed and fed to algorithms in a special way. This module
defines a class that is able to handle a wide variety of timeseries data problems.
"""
from copy import deepcopy
import inspect
from typing import Any, Dict, List, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.type_check import nan_to_num
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.validation import check_is_fitted
import torch
from torch.distributions import Beta
from torch.nn.utils import rnn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from pytorch_forecasting.data.encoders import EncoderNormalizer, GroupNormalizer, NaNLabelEncoder, TorchNormalizer


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
        weight: Union[str, List[str], None] = None,
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
        constant_fill_strategy={},
        allow_missings: bool = False,
        add_relative_time_idx: bool = False,
        add_target_scales: bool = False,
        add_encoder_length: Union[bool, str] = "auto",
        target_normalizer: Union[TorchNormalizer, NaNLabelEncoder, EncoderNormalizer, str] = "auto",
        categorical_encoders={},
        scalers={},
        randomize_length: Union[None, Tuple[float, float], bool] = False,
        predict_mode: bool = False,
    ):
        """
        Timeseries dataset holding data for models.

        Each sample is a subsequence of a full time series. The subsequence consists of encoder and decoder/prediction
        timepoints for a given time series. This class constructs an index which defined which subsequences exists and
        can be samples from (``index`` attribute). The samples in the index are defined by by the various parameters.
        to the class (encoder and prediction lengths, minimum prediction length, randomize length and predict keywords).
        How samples are
        sampled into batches for training, is determined by the DataLoader. The class provides the
        :py:meth:`~TimeSeriesDataSet.to_dataloader` method to convert the dataset into a dataloader.

        Large datasets:

            Currently the class is limited to in-memory operations. If you have extremely large data,
            however, you can pass prefitted encoders and and scalers to it and a subset of sequences to the class to
            construct a valid dataset (plus, likely the EncoderNormalizer should be used to normalize targets).
            when fitting a network, you would then to create a custom DataLoader that rotates through the datasets.
            There is currently no in-built methods to do this.

        Args:
            data: dataframe with sequence data - each row can be identified with ``time_idx`` and the ``group_ids``
            time_idx: integer column denoting the time index. This columns is used to determine the sequence of samples.
                If there no missings observations, the time index should increase by ``+1`` for each subsequent sample.
                The first time_idx for each series does not necessarily have to be ``0`` but any value is allowed.
            target: column denoting the target or list of columns denoting the target - categorical or continous.
            group_ids: list of column names identifying a time series. This means that the ``group_ids`` identify
                a sample together with the ``time_idx``. If you have only one timeseries, set this to the
                name of column that is constant.
            weight: column name for weights or list of column names corresponding to each target
            max_encoder_length: maximum length to encode
            min_encoder_length: minimum allowed length to encode. Defaults to max_encoder_length.
            min_prediction_idx: minimum ``time_idx`` from where to start predictions. This parameter can be useful to
                create a validation or test set.
            max_prediction_length: maximum prediction/decoder length (choose this not too short as it can help
                convergence)
            min_prediction_length: minimum prediction/decoder length. Defaults to max_prediction_length
            static_categoricals: list of categorical variables that do not change over time,
                entries can be also lists which are then encoded together
                (e.g. useful for product categories)
            static_reals: list of continuous variables that do not change over time
            time_varying_known_categoricals: list of categorical variables that change over
                time and are know in the future, entries can be also lists which are then encoded together
                (e.g. useful for special days or promotion categories)
            time_varying_known_reals: list of continuous variables that change over
                time and are know in the future
            time_varying_unknown_categoricals: list of categorical variables that change over
                time and are not know in the future, entries can be also lists which are then encoded together
                (e.g. useful for weather categories)
            time_varying_unknown_reals: list of continuous variables that change over
                time and are not know in the future
            variable_groups: dictionary mapping a name to a list of columns in the data. The name should be present
                in a categorical or real class argument, to be able to encode or scale the columns by group.
            dropout_categoricals: list of categorical variables that are unknown when making a forecast without
                observed history
            constant_fill_strategy: dictionary of column names with constants to fill in missing values if there are
                gaps in the sequence (by default forward fill strategy is used). The values will be only used if
                ``allow_missings=True``. A common use case is to denote that demand was 0 if the sample is not in
                the dataset.
            allow_missings: if to allow missing timesteps that are automatically filled up. Missing values
                refer to gaps in the ``time_idx``, e.g. if a specific timeseries has only samples for
                1, 2, 4, 5, the sample for 3 will be generated on-the-fly.
                Allow missings does not deal with ``NA`` values. You should fill NA values before
                passing the dataframe to the TimeSeriesDataSet.
            add_relative_time_idx: if to add a relative time index as feature (i.e. for each sampled sequence, the index
                will range from -encoder_length to prediction_length)
            add_target_scales: if to add scales for target to static real features (i.e. add the center and scale
                of the unnormalized timeseries as features)
            add_encoder_length: if to add decoder length to list of static real variables. Defaults to "auto",
                i.e. yes if ``min_encoder_length != max_encoder_length``.
            target_normalizer: transformer that takes group_ids, target and time_idx to return normalized targets.
                You can choose from :py:class:`~TorchNormalizer`, :py:class:`~NaNLabelEncoder`,
                :py:class:`~EncoderNormalizer` or `None` for using not normalizer.
                By default an appropriate normalizer is chosen automatically.
            categorical_encoders: dictionary of scikit learn label transformers. If you have unobserved categories in
                the future, you can use the :py:class:`~pytorch_forecasting.encoders.NaNLabelEncoder` with
                ``add_nan=True``. Defaults effectively to sklearn's ``LabelEncoder()``. Prefittet encoders will not
                be fit again.
            scalers: dictionary of scikit learn scalers. Defaults to sklearn's ``StandardScaler()``.
                Other options are :py:class:`~pytorch_forecasting.data.EncoderNormalizer`,
                :py:class:`~pytorch_forecasting.data.GroupNormalizer` or scikit-learn's ``StandarScaler()``,
                ``RobustScaler()`` or `None` for using not normalizer.
                Prefittet encoders will not be fit again (with the exception of the
                :py:class:`~pytorch_forecasting.data.EncoderNormalizer`).
            randomize_length: None or False if not to randomize lengths. Tuple of beta distribution concentrations
                from which
                probabilities are sampled that are used to sample new sequence lengths with a binomial
                distribution.
                If True, defaults to (0.2, 0.05), i.e. ~1/4 of samples around minimum encoder length.
                Defaults to False otherwise.
            predict_mode: if to only iterate over each timeseries once (only the last provided samples).
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
        if isinstance(target, list) and weight is not None:
            assert isinstance(weight, list) and len(target) == len(weight), (
                "if target is a list, then weights should "
                "be either None or also a list of the same length corresponding to targets"
            )
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

        assert (
            self.target not in self.time_varying_known_reals
        ), "target should be an unknown continuous variable in the future"

        # set data
        assert data.index.is_unique, "data index has to be unique"
        if min_prediction_idx is not None:
            data = data[lambda x: data[self.time_idx] >= self.min_prediction_idx - self.max_encoder_length]
        data = data.sort_values(self.group_ids + [self.time_idx])

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

        # validate
        self._validate_data(data)

        # preprocess data
        data = self._preprocess_data(data)

        # create index
        self.index = self._construct_index(data, predict_mode=predict_mode)

        # convert to torch tensor for high performance data loading later
        self.data = self._data_to_tensors(data)

    def _set_target_normalizer(self, data: pd.DataFrame):
        """
        Determine target normalizer.

        Args:
            data (pd.DataFrame): input data
        """
        if isinstance(self.target_normalizer, str) and self.target_normalizer == "auto":
            if data[self.target].dtype.kind != "f":  # category
                self.target_normalizer = NaNLabelEncoder()
                assert not self.add_target_scales, "Target scales can be only added for continous targets"
            else:
                data_positive = (data[self.target] > 0).all()
                if data_positive:
                    if data[self.target].skew() > 2.5:
                        transformer = "log"
                    else:
                        transformer = "relu"
                else:
                    transformer = None
                if self.max_encoder_length > 20 and self.min_encoder_length > 1:
                    self.target_normalizer = EncoderNormalizer(transformation=transformer)
                else:
                    self.target_normalizer = GroupNormalizer(transformation=transformer)
        elif self.target_normalizer is None:
            self.target_normalizer = TorchNormalizer(method="identity")
        assert self.min_encoder_length > 1 or not isinstance(
            self.target_normalizer, EncoderNormalizer
        ), "EncoderNormalizer is only allowed if min_encoder_length > 1"
        assert isinstance(
            self.target_normalizer, (TorchNormalizer, NaNLabelEncoder)
        ), f"target_normalizer has to be either None or of class TorchNormalizer but found {self.target_normalizer}"

    @property
    def _group_ids_mapping(self) -> Dict[str, str]:
        """
        Mapping of group id names to group ids used to identify series in dataset -
        group ids can also be used for target normalizer.
        The former can change from training to validation and test dataset while the later must not.
        """
        return {name: f"__group_id__{name}" for name in self.group_ids}

    @property
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
        # encode group ids - this encoding
        for name, group_name in self._group_ids_mapping.items():
            self.categorical_encoders[group_name] = NaNLabelEncoder().fit(data[name].to_numpy().reshape(-1))
            data[group_name] = self.transform_values(name, data[name], inverse=False, group_id=True)

        # encode categoricals
        if isinstance(
            self.target_normalizer, GroupNormalizer
        ):  # if we use a group normalizer, group_ids must be encoded as well
            group_ids_to_encode = self.group_ids
        else:
            group_ids_to_encode = []
        for name in set(group_ids_to_encode + self.categoricals):
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
                elif self.categorical_encoders[name] is not None:
                    try:
                        check_is_fitted(self.categorical_encoders[name])
                    except NotFittedError:
                        self.categorical_encoders[name] = self.categorical_encoders[name].fit(data[name])

        # encode them
        for name in set(group_ids_to_encode + self.flat_categoricals):
            data[name] = self.transform_values(name, data[name], inverse=False)

        # save special variables
        assert "__time_idx__" not in data.columns, "__time_idx__ is a protected column and must not be present in data"
        data["__time_idx__"] = data[self.time_idx]  # save unscaled
        assert "__target__" not in data.columns, "__target__ is a protected column and must not be present in data"
        data["__target__"] = data[self.target]
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
                elif isinstance(self.target_normalizer, GroupNormalizer):
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
            elif isinstance(self.target_normalizer, NaNLabelEncoder):
                data[self.target] = self.target_normalizer.transform(data[self.target])
                data["__target__"] = data[
                    self.target
                ]  # overwrite target because it requires encoding (continuous targets should not be normalized)
                scales = "no target scales available for categorical target"
            else:
                data[self.target], scales = self.target_normalizer.transform(data[self.target], return_norm=True)

            # add target scales
            if self.add_target_scales:
                for idx, name in enumerate(["center", "scale"]):
                    feature_name = f"{self.target}_{name}"
                    assert (
                        feature_name not in data.columns
                    ), f"{feature_name} is a protected column and must not be present in data"
                    data[feature_name] = scales[:, idx].squeeze()
                    if feature_name not in self.reals:
                        self.static_reals.append(feature_name)

        if self.target in self.reals:
            self.scalers[self.target] = self.target_normalizer
        else:
            self.categorical_encoders[self.target] = self.target_normalizer

        # rescale continuous variables apart from target
        for name in self.reals:
            if name not in self.scalers:
                self.scalers[name] = StandardScaler().fit(data[[name]])
            elif self.scalers[name] is not None:
                try:
                    check_is_fitted(self.scalers[name])
                except NotFittedError:
                    if isinstance(self.scalers[name], GroupNormalizer):
                        self.scalers[name] = self.scalers[name].fit(data[[name]], data)
                    else:
                        self.scalers[name] = self.scalers[name].fit(data[[name]])
            if self.scalers[name] is not None and name != self.target:
                data[name] = self.transform_values(name, data[name], data=data, inverse=False)

        # encode constant values
        self.encoded_constant_fill_strategy = {}
        for name, value in self.constant_fill_strategy.items():
            if name == self.target:
                self.encoded_constant_fill_strategy["__target__"] = value
            self.encoded_constant_fill_strategy[name] = self.transform_values(
                name, np.array([value]), data=data, inverse=False
            )[0]
        return data

    def transform_values(
        self,
        name: str,
        values: Union[pd.Series, torch.Tensor, np.ndarray],
        data: pd.DataFrame = None,
        inverse=False,
        group_id: bool = False,
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

        Returns:
            np.ndarray: (de/en)coded/(de)scaled values
        """
        if group_id:
            name = self._group_ids_mapping[name]
        # remaining categories
        if name in set(self.flat_categoricals + self.group_ids + self._group_ids):
            name = self.variable_to_group_mapping.get(name, name)  # map name to encoder
            encoder = self.categorical_encoders[name]
            if encoder is None:
                return values
            elif inverse:
                return encoder.inverse_transform(values)
            else:
                return encoder.transform(values)

        # reals
        if name in self.scalers:
            scaler = self.scalers[name]
            if scaler is None:
                return values
            if inverse:
                transform = scaler.inverse_transform
            else:
                transform = scaler.transform
            if isinstance(scaler, GroupNormalizer):
                return transform(values, data)
            elif isinstance(scaler, EncoderNormalizer) and name == self.target:
                return transform(values)
            else:
                if isinstance(values, pd.Series):
                    values = values.to_frame()
                    return np.asarray(transform(values)).reshape(-1)
                else:
                    values = values.reshape(-1, 1)
                    return transform(values).reshape(-1)

        return values  # fallback

    def _data_to_tensors(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Convert data to tensors for faster access with :py:meth:`~__getitem__`.

        Args:
            data (pd.DataFrame): preprocessed data

        Returns:
            Dict[str, torch.Tensor]: dictionary of tensors for continous, categorical data, groups, target and
                time index
        """

        index = torch.tensor(data[self._group_ids].to_numpy(np.long), dtype=torch.long)
        time = torch.tensor(data["__time_idx__"].to_numpy(np.long), dtype=torch.long)

        categorical = torch.tensor(data[self.flat_categoricals].to_numpy(np.long), dtype=torch.long)

        if self.weight is None:
            target_names = "__target__"
        else:
            target_names = ["__target__", "__weight__"]
        if isinstance(self.target_normalizer, NaNLabelEncoder):
            target = torch.tensor(data[target_names].to_numpy(dtype=np.long), dtype=torch.long)
        else:
            target = torch.tensor(data[target_names].to_numpy(dtype=np.float), dtype=torch.float)
        continuous = torch.tensor(data[self.reals].to_numpy(dtype=np.float), dtype=torch.float)

        tensors = dict(reals=continuous, categoricals=categorical, groups=index, target=target, time=time)

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
                "Min encoder length and/or min_prediction_idx and/or min prediction length is too large for "
                f"{len(missing_groups)} series/groups which therefore are not present in the dataset index. "
                "This means no predictions can be made for those series"
                f"First 10 removed groups: {list(missing_groups.iloc[:10].to_dict(orient='index').values())}",
                UserWarning,
            )
        assert len(df_index) > 0, "filters should not remove entries"

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

        if variable == self.target:
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
        target = self.data["target"][index.index_start : index.index_end + 1].clone()
        groups = self.data["groups"][index.index_start].clone()
        if isinstance(self.target_normalizer, NaNLabelEncoder):
            target_scale = np.zeros(2, dtype=float)
        else:
            target_scale = self.target_normalizer.get_parameters(groups, self.group_ids)

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
            target = target[indices]

            # reset index
            if self.time_idx in self.reals:
                time_idx = self.reals.index(self.time_idx)
                data_cont[:, time_idx] = torch.linspace(
                    data_cont[0, time_idx], data_cont[-1, time_idx], len(target), dtype=data_cont.dtype
                )

            # make replacements to fill in categories
            for name, value in self.encoded_constant_fill_strategy.items():
                if name in self.reals:
                    data_cont[repetition_indices, self.reals.index(name)] = value
                elif name == "__target__":
                    if target.ndim == 2:
                        target[repetition_indices, 0] = value
                    else:
                        target[repetition_indices] = value
                elif name in self.flat_categoricals:
                    data_cat[repetition_indices, self.flat_categoricals.index(name)] = value
                elif name == self.target:  # target is just not an input value
                    pass
                else:
                    raise KeyError(f"Variable {name} is not known and thus cannot be filled in")

            sequence_length = len(target)

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
            if new_encoder_length + new_decoder_length < len(target):
                data_cat = data_cat[encoder_length - new_encoder_length : encoder_length + new_decoder_length]
                data_cont = data_cont[encoder_length - new_encoder_length : encoder_length + new_decoder_length]
                target = target[encoder_length - new_encoder_length : encoder_length + new_decoder_length]
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

        # rescale covariates
        for name, scaler in self.scalers.items():
            if name != self.target and name in self.reals and isinstance(scaler, EncoderNormalizer):
                # fit and transform
                pos = self.reals.index(name)
                scaler.fit(data_cont[:encoder_length, pos])
                data_cont[:, pos] = scaler.transform(data_cont[:, pos])

        # rescale target
        if self.target_normalizer is not None and isinstance(self.target_normalizer, EncoderNormalizer):
            # fit and transform
            self.target_normalizer.fit(target[:encoder_length])
            # get new scale
            target_scale = self.target_normalizer.get_parameters()
            # modify input data
            if self.target in self.reals:
                data_cont[:, self.reals.index(self.target)] = self.target_normalizer.transform(target)
            if self.add_target_scales:
                data_cont[:, self.reals.index(f"{self.target}_center")] = self.transform_values(
                    f"{self.target}_center", target_scale[0]
                )[0]
                data_cont[:, self.reals.index(f"{self.target}_scale")] = self.transform_values(
                    f"{self.target}_scale", target_scale[1]
                )[0]
            target_scale = target_scale.numpy()  # scale needs to be numpy to be consistent with GroupNormalizer

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

        return (
            dict(
                x_cat=data_cat,
                x_cont=data_cont,
                encoder_length=encoder_length,
                encoder_target=target[:encoder_length],
                encoder_time_idx_start=time[0],
                groups=groups,
                target_scale=target_scale,
            ),
            target[encoder_length:],
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
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: minibatch
        """
        # collate function for dataloader
        # lengths
        encoder_lengths = torch.tensor([batch[0]["encoder_length"] for batch in batches], dtype=torch.long)
        decoder_lengths = torch.tensor([len(batch[1]) for batch in batches], dtype=torch.long)

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
        encoder_target = rnn.pad_sequence([batch[0]["encoder_target"] for batch in batches], batch_first=True)

        decoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][length:] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )
        decoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][length:] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )

        # target
        target = rnn.pad_sequence([batch[1] for batch in batches], batch_first=True)
        if isinstance(batches[0][0]["target_scale"], torch.Tensor):  # stack tensor
            target_scale = torch.stack([batch[0]["target_scale"] for batch in batches])
        else:  # convert to tensor
            target_scale = torch.tensor([batch[0]["target_scale"] for batch in batches], dtype=torch.float)

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
            target,
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


        Examples:

            To samples for training:

            .. code-block:: python

                from torch.utils.data import WeightedRandomSampler

                # length of probabilties for sampler have to be equal to the length of the index
                probabilities = np.sqrt(1 + data.loc[dataset.index, "target"])
                sampler = WeightedRandomSampler(probabilities, len(probabilities))
                dataset.to_dataloader(train=True, sampler=sampler, shuffle=False)

        Returns:
            DataLoader: dataloader that returns Tuple.
                First entry is a dictionary with the entries

                    * encoder_cat
                    * encoder_cont
                    * encoder_target
                    * encoder_lengths
                    * decoder_cat
                    * decoder_cont
                    * decoder_target
                    * decoder_lengths

                Second entry is target
        )
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
