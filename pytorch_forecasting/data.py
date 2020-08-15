"""
Timeseries data is special and has to be processed and fed to algorithms in a special way. This module
defines a class that is able to handle a wide variety of timeseries data problems.
"""

import pickle
import warnings
from copy import deepcopy
import inspect
from typing import Callable, Union, Dict, List, Tuple, Any

import matplotlib.pyplot as plt
from numpy.lib.arraysetops import isin
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
from torch import ones
from torch.distributions import Binomial, Beta
from torch.nn.utils import rnn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class NaNLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Labelencoder that can optionally always encode nan as class 0
    """

    def __init__(self, add_nan: bool = False, warn: bool = True):
        """
        init NaNLabelEncoder

        Args:
            add_nan: if to force encoding of nan at 0
            warn: if to warn if additional nans are added because items are unknown
        """
        self.add_nan = add_nan
        self.warn = warn
        super().__init__()

    def fit_transform(self, y):
        if self.add_nan:
            self.fit(y)
            return self.transform(y)
        return super().transform(y)

    def is_numeric(self, y):
        return y.dtype.kind in "bcif" or (isinstance(y, pd.CategoricalDtype) and y.cat.categories.dtype.kind in "bcif")

    def fit(self, y):
        if self.add_nan:
            if self.is_numeric(y):
                nan = np.nan
            else:
                nan = "nan"
            self.classes_ = {nan: 0}
            for idx, val in enumerate(np.unique(y)):
                self.classes_[val] = idx + 1
        else:
            self.classes_ = {val: idx for idx, val in enumerate(np.unique(y))}
        self.classes_vector_ = np.array(list(self.classes_.keys()))
        return self

    def transform(self, y):
        if self.add_nan:
            if self.warn:
                cond = ~np.isin(y, self.classes_)
                if cond.any():
                    warnings.warn(f"Found {y[cond].nunique()} unknown classes which were set to NaN", UserWarning)

            encoded = [self.classes_.get(v, 0) for v in y]

        else:
            encoded = [self.classes_[v] for v in y]

        if isinstance(y, torch.Tensor):
            encoded = torch.tensor(encoded, dtype=torch.long, device=y.device)
        else:
            encoded = np.array(encoded)
        return encoded

    def inverse_transform(self, y):
        if y.max() >= len(self.classes_vector_):
            raise KeyError("New unknown values detected")

        # decode
        decoded = self.classes_vector_[y]
        return decoded


class TorchNormalizer(BaseEstimator, TransformerMixin):
    """
    Basic target transformer that can be fit also on torch tensors.

    Args:
        BaseEstimator ([type]): [description]
        TransformerMixin ([type]): [description]
    """

    def __init__(
        self,
        method: str = "standard",
        center: bool = True,
        log_scale: Union[bool, float] = False,
        log_zero_value: float = 0.0,
        coerce_positive: Union[float, bool] = None,
        eps: float = 1e-8,
    ):
        """
        Initialize

        Args:
            method (str, optional): method to rescale series. Either "standard" (standard scaling) or "robust"
                (scale using quantiles 0.25-0.75). Defaults to "standard".
            center (bool, optional): If to center the output to zero. Defaults to True.
            log_scale (bool, optional): If to take log of values. Defaults to False. Defaults to False.
            log_zero_value (float, optional): Value to map 0 to for ``log_scale=True`` or in softplus. Defaults to 0.0
            coerce_positive (Union[bool, float, str], optional): If to coerce output to positive. Valid values:
                * None, i.e. is automatically determined and might change to True if all values are >= 0 (Default).
                * True, i.e. output is clamped at 0.
                * False, i.e. values are not coerced
                * float, i.e. softmax is applied with beta = coerce_positive.
            eps (float, optional): Number for numerical stability of calcualtions. Defaults to 1e-8.
        """
        self.method = method
        assert method in ["standard", "robust"], f"method has invalid value {method}"
        self.center = center
        self.eps = eps

        # set log scale
        self.log_zero_value = np.exp(log_zero_value)
        self.log_scale = log_scale

        # check if coerce positive should be determined automatically
        if coerce_positive is None:
            if log_scale:
                coerce_positive = False
        else:
            assert not (self.log_scale and coerce_positive), (
                "log scale means that output is transformed to a positive number by default while coercing positive"
                " will apply softmax function - decide for either one or the other"
            )
        self.coerce_positive = coerce_positive

    def get_parameters(self, *args, **kwargs):
        return torch.tensor([self.center_, self.scale_])

    def _preprocess_y(self, y):
        if self.coerce_positive is None and not self.log_scale:
            self.coerce_positive = (y >= 0).all()

        if self.log_scale:
            if isinstance(y, torch.Tensor):
                y = torch.log(y + self.log_zero_value)
            else:
                y = np.log(y + self.log_zero_value)
        return y

    def fit(self, y):
        y = self._preprocess_y(y)

        if self.method == "standard":
            if isinstance(y, torch.Tensor):
                self.center_ = torch.mean(y)
                self.scale_ = torch.std(y) / (self.center_ + self.eps)
            else:
                self.center_ = np.mean(y)
                self.scale_ = np.std(y) / (self.center_ + self.eps)

        elif self.method == "robust":
            if isinstance(y, torch.Tensor):
                self.center_ = torch.median(y)
                q_75 = y.kthvalue(int(len(y) * 0.75)).values
                q_25 = y.kthvalue(int(len(y) * 0.25)).values
            else:
                self.center_ = np.median(y)
                q_75 = np.percentiley(y, 75)
                q_25 = np.percentiley(y, 25)
            self.scale_ = (q_75 - q_25) / (self.center_ + self.eps) / 2.0
        return self

    def transform(self, y, return_norm: bool = False):
        if self.log_scale:
            if isinstance(y, torch.Tensor):
                y = (y + self.log_zero_value + self.eps).log()
            else:
                y = np.log(y + self.log_zero_value + self.eps)
        if self.center:
            y = (y / (self.center_ + self.eps) - 1) / (self.scale_ + self.eps)
        else:
            y = y / (self.center_ + self.eps)
        if return_norm:
            return y, self.get_parameters().numpy()[None, :]
        else:
            return y

    def inverse_transform(self, y):
        return self(dict(prediction=y, target_scale=self.get_parameters().unsqueeze(0)))

    def __call__(self, data: Dict[str, torch.Tensor]):
        # inverse transformation with tensors
        norm = data["target_scale"]

        # use correct shape for norm
        if data["prediction"].ndim > norm.ndim:
            norm = norm.unsqueeze(-1)

        # transform
        if self.center:
            y_normed = (data["prediction"] * norm[:, 1, None] + 1) * norm[:, 0, None]
        else:
            y_normed = data["prediction"] * norm[:, 0, None]
        if self.log_scale:
            y_normed = (y_normed.exp() - self.log_zero_value).clamp_min(0.0)
        elif isinstance(self.coerce_positive, bool) and self.coerce_positive:
            y_normed = y_normed.clamp_min(0.0)
        elif isinstance(self.coerce_positive, float):
            y_normed = F.softplus(y_normed, beta=float(self.coerce_positive))

        # return correct shape
        if data["prediction"].ndim == 1 and y_normed.ndim > 1:
            y_normed = y_normed.squeeze(0)
        return y_normed


class EncoderNormalizer(TorchNormalizer):
    """
    Special Normalizer that is fit on each encoding sequence
    """

    pass


class GroupNormalizer(TorchNormalizer):
    # todo: allow window (exp weighted), different methods such as quantile for robust scaling
    def __init__(
        self,
        method: str = "standard",
        groups: List[str] = [],
        center: bool = True,
        scale_by_group: bool = False,
        log_scale: Union[bool, float] = False,
        log_zero_value: float = 0.0,
        coerce_positive: Union[float, bool] = None,
        eps: float = 1e-8,
    ):
        """
        Group normalizer to normalize a given entry by groups. Can be used as target normalizer.

        Args:
            method (str, optional): method to rescale series. Either "standard" (standard scaling) or "robust"
                (scale using quantiles 0.25-0.75). Defaults to "standard".
            groups (List[str], optional): Group names to normalize by. Defaults to [].
            center (bool, optional): If to center the output to zero. Defaults to True.
            scale_by_group (bool, optional): If to scale the output by group, i.e. norm is calculated as
                ``(group1_norm * group2_norm * ...) ^ (1 / n_groups)``. Defaults to False.
            log_scale (bool, optional): If to take log of values. Defaults to False. Defaults to False.
            log_zero_value (float, optional): Value to map 0 to for ``log_scale=True`` or in softplus. Defaults to 0.0
            coerce_positive (Union[bool, float, str], optional): If to coerce output to positive. Valid values:
                * None, i.e. is automatically determined and might change to True if all values are >= 0 (Default).
                * True, i.e. output is clamped at 0.
                * False, i.e. values are not coerced
                * float, i.e. softmax is applied with beta = coerce_positive.
            eps (float, optional): Number for numerical stability of calcualtions. Defaults to 1e-8.
        """
        self.groups = groups
        self.scale_by_group = scale_by_group
        super().__init__(
            method=method,
            center=center,
            log_scale=log_scale,
            log_zero_value=log_zero_value,
            coerce_positive=coerce_positive,
            eps=eps,
        )

    def fit(self, y, X):
        y = self._preprocess_y(y)
        if len(self.groups) == 0:
            assert not self.scale_by_group, "No groups are defined, i.e. `scale_by_group=[]`"
            if self.method == "standard":
                mean = np.mean(y)
                self.norm_ = mean, np.std(y) / (mean + self.eps)
            else:
                quantiles = np.quantile(y, [0.25, 0.5, 0.75])
                self.norm_ = quantiles[1], (quantiles[2] - quantiles[0]) / (quantiles[1] + self.eps)

        elif self.scale_by_group:
            if self.method == "standard":
                self.norm_ = {
                    g: X[[g]]
                    .assign(y=y)
                    .groupby(g, observed=True)
                    .agg(mean=("y", "mean"), scale=("y", "std"))
                    .assign(scale=lambda x: x.scale / (x["mean"] + self.eps))
                    for g in self.groups
                }
            else:
                self.norm_ = {
                    g: X[[g]]
                    .assign(y=y)
                    .groupby(g, observed=True)
                    .y.quantile([0.25, 0.5, 0.75])
                    .unstack(-1)
                    .assign(
                        median=lambda x: x[0.5] + self.eps,
                        scale=lambda x: (x[0.75] - x[0.25] + self.eps) / (x[0.5] + self.eps),
                    )[["median", "scale"]]
                    for g in self.groups
                }
            # calculate missings
            self.missing_ = {group: scales.median().to_dict() for group, scales in self.norm_.items()}

        else:
            if self.method == "standard":
                self.norm_ = (
                    X[self.groups]
                    .assign(y=y)
                    .groupby(self.groups, observed=True)
                    .agg(mean=("y", "mean"), scale=("y", "std"))
                    .assign(scale=lambda x: x.scale / (x["mean"] + self.eps))
                )
            else:
                self.norm_ = (
                    X[self.groups]
                    .assign(y=y)
                    .groupby(self.groups, observed=True)
                    .y.quantile([0.25, 0.5, 0.75])
                    .unstack(-1)
                    .assign(
                        median=lambda x: x[0.5] + self.eps,
                        scale=lambda x: (x[0.75] - x[0.25] + self.eps) / (x[0.5] + self.eps) / 2.0,
                    )[["median", "scale"]]
                )
            self.missing_ = self.norm_.median().to_dict()
        return self

    @property
    def names(self) -> List[str]:
        if self.method == "standard":
            return ["mean", "scale"]
        else:
            return ["median", "scale"]

    def fit_transform(self, y, X, return_norm: bool = False) -> pd.DataFrame:
        return self.fit(y, X).transform(y, X, return_norm=return_norm)

    def inverse_transform(self, y, X):
        raise NotImplementedError()

    def transform(self, y: pd.Series, X: pd.DataFrame, return_norm: bool = False) -> pd.DataFrame:
        norm = self.get_norm(X)
        y = self._preprocess_y(y)
        if self.center:
            y_normed = (y / (norm[:, 0] + self.eps) - 1) / (norm[:, 1] + self.eps)
        else:
            y_normed = y / (norm[:, 0] + self.eps)
        if return_norm:
            return y_normed, norm
        else:
            return y_normed

    def get_parameters(self, groups, group_names: List[str] = None):
        if isinstance(groups, torch.Tensor):
            groups = groups.tolist()
        if isinstance(groups, list):
            groups = tuple(groups)
        if group_names is None:
            group_names = self.groups
        else:
            # filter group names
            group_names = [name for name in group_names if name in self.groups]
        assert len(group_names) == len(self.groups), "Passed groups and fitted do not match"

        if len(self.groups) == 0:
            params = np.asarray(self.norm_).squeeze()
        elif self.scale_by_group:
            norm = np.array([1.0, 1.0])
            for group, group_name in zip(groups, group_names):
                try:
                    norm = norm * self.norm_[group_name].loc[group].to_numpy()
                except KeyError:
                    norm = norm * np.asarray([self.missing_[group_name][name] for name in self.names])
            norm = np.power(norm, 1.0 / len(self.groups))
            params = norm
        else:
            try:
                params = self.norm_.loc[groups].to_numpy()
            except (KeyError, TypeError):
                params = np.asarray([self.missing_[name] for name in self.names])
        return params

    def get_norm(self, X):
        if len(self.groups) == 0:
            norm = np.asarray(self.norm_).reshape(1, -1)
        elif self.scale_by_group:
            norm = [
                np.prod(
                    [
                        X[group_name]
                        .map(self.norm_[group_name][name])
                        .fillna(self.missing_[group_name][name])
                        .to_numpy()
                        for group_name in self.groups
                    ],
                    axis=0,
                )
                for name in self.names
            ]
            norm = np.power(np.stack(norm, axis=1), 1.0 / len(self.groups))
        else:
            norm = X[self.groups].set_index(self.groups).join(self.norm_).fillna(self.missing_).to_numpy()
        return norm


class TimeSeriesDataSet(Dataset):
    """Dataset Basic Structure for Temporal Fusion Transformer"""

    def __init__(
        self,
        data: pd.DataFrame,
        time_idx: str,
        target: str,
        group_ids: List[str],
        weight: Union[str, None] = None,
        max_encoder_length: int = 30,
        min_encoder_length: int = 0,
        min_prediction_idx: int = None,
        min_prediction_length: int = 1,
        max_prediction_length: int = 1,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_known_categoricals: List[str] = [],
        time_varying_known_reals: List[str] = [],
        time_varying_unknown_categoricals: List[str] = [],
        time_varying_unknown_reals: List[str] = [],
        variable_groups: Dict[str, List[int]] = {},
        dropout_categoricals: List[str] = [],
        add_relative_time_idx: bool = True,
        constant_fill_strategy={},
        allow_missings: bool = False,
        categorical_encoders={},
        scalers={},
        randomize_length: Union[None, Tuple[float, float]] = (0.2, 0.05),
        predict_mode: bool = False,
        target_normalizer: Union[TorchNormalizer, str] = "auto",
        add_target_scales: bool = True,
        add_decoder_length: Union[bool, str] = "auto",
    ):
        """
        Timeseries dataset

        Args:
            data: dataframe with sequence data - each row can be identified with ``time_idx`` and the ``group_ids``
            time_idx: integer column denoting the time index
            target: float column denoting the target
            group_ids: list of column names identifying a timeseries
            weight: column name for weights
            max_encoder_length: maximum length to encode
            min_encoder_length: minimum allowed length to encode
            min_prediction_idx: minimum time index from where to start predictions
            min_prediction_length: minimum prediction length
            max_prediction_length: maximum prediction length (choose this not too short as it can help convergence)
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
            dropout_categoricals: list of categorical variables that are unknown when making a forecast without
                observed history
            add_relative_time_idx: if to add a relative time index as feature
            constant_fill_strategy: dictionary of column names with constants to fill in missing values if there are
                gaps in the sequence
                (otherwise forward fill strategy is used)
            allow_missings: if to allow missing timesteps that are automatically filled up
            categorical_encoders: dictionary of scikit learn label transformers or None
            scalers: dictionary of scikit learn scalers or None
            randomize_length: None if not to randomize lengths. Tuple of beta distribution concentrations from which
                probabilities are sampled that are used to sample new sequence lengths with a binomial distribution
            predict_mode: if to only iterate over each timeseries once (only the last provided samples)
            target_normalizer: transformer that takes group_ids, target and time_idx to return normalized target
            add_target_scales: if to add scales for target to static real features
            add_decoder_length: if to add decoder length to list of static real variables. Defaults to "auto",
                i.e. yes if ``min_encoder_length != max_encoder_length``.
        """
        super().__init__()
        self.min_encoder_length = min_encoder_length
        self.max_encoder_length = max_encoder_length
        assert (
            self.min_encoder_length <= self.max_encoder_length
        ), "max encoder length has to be larger equals min encoder length"
        self.max_prediction_length = max_prediction_length
        self.min_prediction_length = min_prediction_length
        assert (
            self.min_prediction_length <= self.max_prediction_length
        ), "max prediction length has to be larger equals min prediction length"
        assert self.min_prediction_length > 0, "prediction length must be larger than 0"
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
        self.randomize_length = randomize_length
        self.min_prediction_idx = min_prediction_idx or data[self.time_idx].min()
        self.constant_fill_strategy = {} if len(constant_fill_strategy) == 0 else constant_fill_strategy
        self.predict_mode = predict_mode
        self.allow_missings = allow_missings
        self.target_normalizer = target_normalizer
        self.categorical_encoders = {} if len(categorical_encoders) == 0 else categorical_encoders
        self.scalers = {} if len(scalers) == 0 else scalers
        self.add_target_scales = add_target_scales
        self.variable_groups = {} if len(variable_groups) == 0 else variable_groups

        # add_decoder_length
        if isinstance(add_decoder_length, str):
            assert (
                add_decoder_length == "auto"
            ), f"Only 'auto' allowed for add_decoder_length but found {add_decoder_length}"
            add_decoder_length = self.min_encoder_length != self.max_encoder_length
        assert isinstance(
            add_decoder_length, bool
        ), f"add_decoder_length should be boolean or 'auto' but found {add_decoder_length}"
        self.add_decoder_length = add_decoder_length

        # target normalizer
        if isinstance(self.target_normalizer, str) and self.target_normalizer == "auto":
            coerce_positive = (data[self.target] >= 0).all()
            if coerce_positive:
                log_scale = data[self.target].skew() > 2.5
                coerce_positive = False
            if self.max_encoder_length > 20 and self.min_encoder_length > 1:
                self.target_normalizer = EncoderNormalizer(coerce_positive=coerce_positive, log_scale=log_scale)
            else:
                self.target_normalizer = GroupNormalizer(coerce_positive=coerce_positive, log_scale=log_scale)
        assert self.min_encoder_length > 1 or not isinstance(
            self.target_normalizer, EncoderNormalizer
        ), "EncoderNormalizer is only allowed if min_encoder_length > 1"
        assert isinstance(
            self.target_normalizer, TorchNormalizer
        ), f"target_normalizer has to be either None or of class TorchNormalizer but found {self.target_normalizer}"

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
        if self.add_decoder_length:
            assert (
                "decoder_length" not in data.columns
            ), "decoder_length is a protected column and must not be present in data"
            if "decoder_length" not in self.time_varying_known_reals and "decoder_length" not in self.reals:
                self.static_reals.append("decoder_length")
            data["decoder_length"] = 0.0  # dummy - real value will be set dynamiclly in __getitem__()

        # validate
        self._validate_data(data)

        # preprocess data
        data = self._preprocess_data(data)

        # create index
        self.index = self._construct_index(data, predict_mode=predict_mode)

        # convert to torch tensor for high performance data loading later
        self.data = self._data_to_tensors(data)

    def _validate_data(self, data: pd.DataFrame):
        # check for numeric categoricals which can cause hick-ups in logging
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

    def save(self, fname: str) -> None:
        """
        Save dataset to disk

        Args:
            fname (str): filename to save to
        """
        with open(fname, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, fname: str):
        """
        Load dataset from disk

        Args:
            fname (str): filename to load from

        Returns:
            TimeSeriesDataSet
        """
        with open(fname, "rb") as file:
            obj = pickle.load(file)
        assert isinstance(obj, cls), f"Loaded file is not of class {cls}"
        return obj

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:

        # encode categoricals
        for name in set(self.categoricals + self.group_ids):
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
        for name in set(self.flat_categoricals + self.group_ids):
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
                    for name in inspect.signature(GroupNormalizer).parameters.keys()
                    if name in inspect.signature(EncoderNormalizer).parameters.keys()
                ]
                copy_kwargs = {name: getattr(self.target_normalizer, name) for name in common_init_args}
                normalizer = GroupNormalizer(groups=self.group_ids, **copy_kwargs)
                data[self.target], scales = normalizer.fit_transform(data[self.target], data, return_norm=True)
            elif isinstance(self.target_normalizer, GroupNormalizer):
                data[self.target], scales = self.target_normalizer.transform(data[self.target], data, return_norm=True)
            else:
                data[self.target], scales = self.target_normalizer.transform(data[self.target], return_norm=True)

            # add target sclaes
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

    def transform_values(self, name, values, data: pd.DataFrame = None, inverse=False):
        # remaining categories
        if name in set(self.flat_categoricals + self.group_ids):
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
            elif isinstance(scaler, EncoderNormalizer):
                return transform(values)
            else:
                if isinstance(values, pd.Series):
                    values = values.to_frame()
                else:
                    values = values.reshape(-1, 1)
                return transform(values).reshape(-1)

        return values  # fallback

    def _data_to_tensors(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:

        index = torch.tensor(data[self.group_ids].to_numpy(np.long))
        time = torch.tensor(data["__time_idx__"].to_numpy(np.long))

        categorical = torch.tensor(data[self.flat_categoricals].to_numpy(np.long))

        if self.weight is None:
            target_names = "__target__"
        else:
            target_names = ["__target__", "__weight__"]
        target = torch.tensor(data[target_names].to_numpy(dtype=np.float32))
        continuous = torch.tensor(data[self.reals].to_numpy(dtype=np.float32))

        tensors = dict(reals=continuous, categoricals=categorical, groups=index, target=target, time=time)

        return tensors

    @property
    def categoricals(self) -> List[str]:
        return self.static_categoricals + self.time_varying_known_categoricals + self.time_varying_unknown_categoricals

    @property
    def flat_categoricals(self) -> List[str]:
        categories = []
        for name in self.categoricals:
            if name in self.variable_groups:
                categories.extend(self.variable_groups[name])
            else:
                categories.append(name)
        return categories

    @property
    def variable_to_group_mapping(self) -> Dict[str, str]:
        groups = {}
        for group_name, sublist in self.variable_groups.items():
            groups.update({name: group_name for name in sublist})
        return groups

    @property
    def reals(self) -> List[str]:
        return self.static_reals + self.time_varying_known_reals + self.time_varying_unknown_reals

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters that can be used with :py:meth:`~from_parameters` to create a new dataset with the same scalers.

        Returns:
            Dict[str, Any]: dictionary of parameters
        """
        kwargs = {
            name: getattr(self, name) for name in inspect.signature(self.__class__).parameters.keys() if name != "data"
        }
        kwargs["categorical_encoders"] = self.categorical_encoders
        kwargs["scalers"] = self.scalers
        return kwargs

    @classmethod
    def from_dataset(
        cls, dataset, data: pd.DataFrame, stop_randomization: bool = False, predict: bool = False, **update_kwargs
    ):
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

        g = data.groupby(self.group_ids, observed=True)

        df_index_first = g["__time_idx__"].transform("nth", 0).to_frame("time_first")
        df_index_last = g["__time_idx__"].transform("nth", -1).to_frame("time_last")
        df_index_diff_to_next = -g["__time_idx__"].diff(-1).fillna(-1).astype(int).to_frame("time_diff_to_next")
        df_index = pd.concat([df_index_first, df_index_last, df_index_diff_to_next], axis=1)
        df_index["index_start"] = np.arange(len(df_index))
        df_index["time"] = data["__time_idx__"]
        df_index["count"] = (df_index["time_last"] - df_index["time_first"]).astype(int) + 1
        df_index["group_id"] = g.ngroup()

        # calculate maxium index to include from current index_start
        max_time = (df_index["time"] + self.max_encoder_length + self.max_prediction_length).clip(
            upper=df_index["count"] + df_index.time_first
        )

        # if there are missing timesteps, we cannot say directly what is the last timestep to include
        # therefore we iterate until it is found
        if (df_index["time_diff_to_next"] != 1).any():
            assert (
                self.allow_missings
            ), "Time difference between steps has been idenfied as larger than 1 - set allow_missings=True"
            df_index["index_end"] = df_index["index_start"]
            for _ in range(df_index["count"].max()):
                new_end_time = (
                    df_index[["time", "time_diff_to_next"]].iloc[df_index["index_end"]].sum(axis=1).to_numpy()
                )
                df_index["index_end"] = df_index["index_end"].where(
                    new_end_time + 1 > max_time, df_index["index_end"] + 1
                )
        else:
            # direct calculation of end index if there are no missing timesteps in the data
            df_index["index_end"] = df_index["index_start"] + (max_time - df_index["time"] - 1)

        # filter out where encode and decode length are not satisfied
        df_index["sequence_length"] = df_index["time"].iloc[df_index["index_end"]].to_numpy() - df_index["time"] + 1

        # filter too short sequences
        df_index = df_index[
            # sequence must be at least of minimal prediction length
            lambda x: (x.sequence_length >= self.min_prediction_length + self.min_encoder_length)
            &
            # prediction must be for after minimal prediction index + length of prediction
            (x["sequence_length"] + x["time"] - 1 >= self.min_prediction_idx - 1 + self.min_prediction_length)
        ]

        if predict_mode:  # keep longest element per series (i.e. the first element that spans to the end of the series)
            # filter all elements that are longer than the allowed maximum sequence length
            df_index = df_index[
                lambda x: (x["time_last"] - x["time"] + 1 <= self.max_prediction_length + self.max_encoder_length)
                & (x["sequence_length"] >= self.min_prediction_length + self.min_encoder_length)
            ]
            # choose longest sequence
            df_index = df_index.loc[df_index.groupby("group_id").sequence_length.idxmax()]
        assert len(df_index) > 0, "filters should not remove entries"

        return df_index

    @staticmethod
    def plot_randomization(betas=(0.2, 0.1), length=24, min_length=0) -> Tuple[plt.Figure, torch.Tensor]:
        probabilities = Beta(betas[0], betas[1]).sample((1000,))

        lengths = ((length - min_length) * probabilities).round() + min_length

        fig, ax = plt.subplots()
        ax.hist(lengths)
        return fig, lengths

    def __len__(self) -> int:
        return self.index.shape[0]

    def set_overwrite_values(self, values: Union[float, torch.Tensor], variable: str, target: str = "decoder"):
        """
        Convenience method to quickly overwrite values in decoder or encoder (or both) for a specific variable.

        Args:
            values (Union[float, torch.Tensor]): values to use for overwrite.
            variable (str): variable whose values should be overwritten.
            target (str, optional): positions to overwrite. One of "decoder", "encoder" or "all". Defaults to "decoder".
        """
        values = torch.tensor(self.transform_values(variable, np.asarray(values).reshape(-1), inverse=False)).squeeze()
        assert target in [
            "all",
            "decoder",
            "encoder",
        ], f"target has be one of 'all', 'decoder' or 'encoder' but target={target} instead"

        if variable == self.target:
            raise NotImplementedError("Target variable is not supported")
        if self.weight is not None and self.weight == variable:
            raise NotImplementedError("Weight variable is not supported")
        if isinstance(self.scalers.get(variable, self.categorical_encoders.get(variable)), TorchNormalizer):
            raise NotImplementedError("TorchNormalizer (e.g. GroupNormalizer) is not supported")

        if self._overwrite_values is None:
            self._overwrite_values = {}
        self._overwrite_values.update(dict(values=values, variable=variable, target=target))

    def reset_overwrite_values(self):
        self._overwrite_values = None

    def __getitem__(self, idx):
        index = self.index.iloc[idx]
        # get index data
        data_cont = self.data["reals"][index.index_start : index.index_end + 1].clone()
        data_cat = self.data["categoricals"][index.index_start : index.index_end + 1].clone()
        time = self.data["time"][index.index_start : index.index_end + 1].clone()
        target = self.data["target"][index.index_start : index.index_end + 1].clone()
        groups = self.data["groups"][index.index_start].clone()
        target_scale = self.target_normalizer.get_parameters(groups, self.group_ids)

        # fill in missing values (if not all time indices are specified
        sequence_length = len(time)
        if sequence_length < index.sequence_length:
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

        if self.add_decoder_length:
            data_cont[:, self.reals.index("decoder_length")] = (
                decoder_length - 0.5 * self.max_encoder_length
            ) / self.max_encoder_length

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
            if self._overwrite_values["target"] == "all":
                positions = slice(None)
            elif self._overwrite_values["target"] == "encoder":
                positions = slice(None, encoder_length)
            else:  # decoder
                positions = slice(encoder_length, None)

            if self._overwrite_values["variable"] in self.reals:
                idx = self.reals.index(self._overwrite_values["variable"])
                data_cont = data_cont  # not to overwrite original data
                data_cont[positions, idx] = self._overwrite_values["values"]
            else:
                assert (
                    self._overwrite_values["variable"] in self.flat_categoricals
                ), "overwrite values variable has to be either in real or categorical variables"
                idx = self.flat_categoricals.index(self._overwrite_values["variable"])
                data_cat = data_cat  # not to overwrite original data
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

    def _collate_fn(self, batches):
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
        target_scale = torch.tensor([batch[0]["target_scale"] for batch in batches], dtype=torch.float32)

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

    def to_dataloader(self, train: bool = True, batch_size: int = 64, **kwargs) -> DataLoader:
        """
        Get dataloader from dataset.

        Args:
            train (bool, optional): if dataloader is used for training or prediction
                Will shuffle and drop last batch if True. Defaults to True.

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
        return DataLoader(
            self,
            shuffle=train,
            drop_last=train and len(self) > batch_size,
            collate_fn=self._collate_fn,
            batch_size=batch_size,
            **kwargs,
        )

    def get_index(self) -> pd.DataFrame:
        """
        Data index / order in which items are returned in train=False mode by dataloader.

        Returns:
            dataframe with time index column for first prediction and group ids
        """
        decoder_length = pd.DataFrame(
            dict(
                prediction_idx=self.data["time"][self.index.index_end.to_numpy()] - (self.min_prediction_idx - 1),
                sequence_length=self.index.sequence_length,
                max_prediction_length=self.max_prediction_length,
            )
        ).min(axis=1)
        encoder_lengths = self.index.sequence_length - decoder_length
        index_data = {self.time_idx: self.index.time + encoder_lengths}
        for id in self.group_ids:
            index_data[id] = self.data["groups"][:, self.group_ids.index(id)][self.index.index_start.to_numpy()]
            # decode if possible
            index_data[id] = self.transform_values(id, index_data[id], inverse=True)
        index = pd.DataFrame(index_data, index=self.index.index)
        return index
