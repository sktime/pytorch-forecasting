"""
Encoders for encoding categorical variables and scaling continuous data.
"""

from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from torch.distributions import constraints
from torch.distributions.transforms import (
    ExpTransform,
    PowerTransform,
    SigmoidTransform,
    Transform,
    _clipped_sigmoid,
)
import torch.nn.functional as F
from torch.nn.utils import rnn

from pytorch_forecasting.utils import InitialParameterRepresenterMixIn


def _plus_one(x):
    return x + 1


def _minus_one(x):
    return x - 1


def _identity(x):
    return x


def _clipped_logit(x):
    finfo = torch.finfo(x.dtype)
    x = x.clamp(min=finfo.eps, max=1.0 - finfo.eps)
    return x.log() - (-x).log1p()


def softplus_inv(y):
    finfo = torch.finfo(y.dtype)
    return y.where(y > 20.0, y + (y + finfo.eps).neg().expm1().neg().log())


def _square(y):
    return torch.pow(y, 2.0)


def _clipped_log(y):
    finfo = torch.finfo(y.dtype)
    return y.log().clamp(min=-1 / finfo.eps)


class SoftplusTransform(Transform):
    r"""
    Transform via the mapping :math:`\text{Softplus}(x) = \log(1 + \exp(x))`.
    The implementation reverts to the linear function when :math:`x > 20`.
    """

    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, SoftplusTransform)

    def _call(self, x):
        return F.softplus(x)

    def _inverse(self, y):
        return softplus_inv(y)

    def log_abs_det_jacobian(self, x, y):
        return -F.softplus(-x)


class Expm1Transform(ExpTransform):
    codomain = constraints.greater_than_eq(-1.0)

    def _call(self, x):
        return super()._call(x) - 1.0

    def _inverse(self, y):
        return super()._inverse(y + 1.0)


class MinusOneTransform(Transform):
    r"""
    Transform x -> x - 1.
    """

    domain = constraints.real
    codomain = constraints.real
    sign: int = 1
    bijective: bool = True

    def _call(self, x):
        return x - 1.0

    def _inverse(self, y):
        return y + 1.0

    def log_abs_det_jacobian(self, x, y):
        return 0.0


class ReLuTransform(Transform):
    r"""
    Transform x -> max(0, x).
    """

    domain = constraints.real
    codomain = constraints.nonnegative
    sign: int = 1
    bijective: bool = False

    def _call(self, x):
        return F.relu(x)

    def _inverse(self, y):
        return y

    def log_abs_det_jacobian(self, x, y):
        return 0.0


class TransformMixIn:
    """Mixin for providing pre- and post-processing capabilities to encoders.

    Class should have a ``transformation`` attribute to indicate how to preprocess data.
    """

    # dict of PyTorch functions that transforms and inversely transforms values.
    # inverse entry required if "reverse" is not the "inverse" of "forward".
    TRANSFORMATIONS = {
        "log": dict(
            forward=_clipped_log, reverse=torch.exp, inverse_torch=ExpTransform()
        ),
        "log1p": dict(
            forward=torch.log1p,
            reverse=torch.exp,
            inverse=torch.expm1,
            inverse_torch=Expm1Transform(),
        ),
        "logit": dict(
            forward=_clipped_logit,
            reverse=_clipped_sigmoid,
            inverse_torch=SigmoidTransform(),
        ),
        "count": dict(
            forward=_plus_one,
            reverse=F.softplus,
            inverse=_minus_one,
            inverse_torch=MinusOneTransform(),
        ),
        "softplus": dict(
            forward=softplus_inv, reverse=F.softplus, inverse_torch=SoftplusTransform()
        ),
        "relu": dict(
            forward=_identity,
            reverse=F.relu,
            inverse=_identity,
            inverse_torch=ReLuTransform(),
        ),
        "sqrt": dict(
            forward=torch.sqrt,
            reverse=_square,
            inverse_torch=PowerTransform(exponent=2.0),
        ),
    }

    @classmethod
    def get_transform(
        cls, transformation: Union[str, Dict[str, Callable]]
    ) -> Dict[str, Callable]:
        """Return transformation functions.

        Parameters
        ----------
        transformation: Union[str, Dict[str, Callable]]
            name of transformation or dictionary with transformation information.

        Returns
        -------
        Dict[str, Callable]
            dictionary with transformation functions
            (forward, reverse, inverse and inverse_torch)
        """
        if isinstance(transformation, str):
            transform = cls.TRANSFORMATIONS[transformation]
        else:
            transform = transformation
        transform.setdefault("forward", _identity)
        transform.setdefault("reverse", _identity)
        return transform

    def preprocess(
        self, y: Union[pd.Series, pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Preprocess input data (e.g. take log).

        Uses ``transform`` attribute to determine how to apply transform.

        Parameters
        ----------
        y: Union[pd.Series, pd.DataFrame, np.ndarray, torch.Tensor]
            input data

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            return rescaled series with type depending on input type
        """
        if self.transformation is None:
            return y

        if isinstance(y, torch.Tensor):
            y = self.get_transform(self.transformation)["forward"](y)
        else:
            # convert first to tensor, then transform and then convert to numpy array
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y = y.to_numpy()
            y = torch.as_tensor(y)
            y = self.get_transform(self.transformation)["forward"](y)
            y = np.asarray(y)
        return y

    def inverse_preprocess(
        self, y: Union[pd.Series, np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Inverse preprocess re-scaled data (e.g. take exp).

        Uses ``transform`` attribute to determine how to apply inverse transform.

        Parameters
        ----------
        y: Union[pd.Series, np.ndarray, torch.Tensor]
            input data

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            return rescaled series with type depending on input type
        """
        if self.transformation is None:
            pass
        elif isinstance(y, torch.Tensor):
            y = self.get_transform(self.transformation)["reverse"](y)
        else:
            # convert first to tensor, then transform and then convert to numpy array
            y = torch.as_tensor(y)
            y = self.get_transform(self.transformation)["reverse"](y)
            y = np.asarray(y)
        return y


class NaNLabelEncoder(
    InitialParameterRepresenterMixIn, BaseEstimator, TransformerMixin, TransformMixIn
):
    """
    Labelencoder that can optionally always encode nan and unknown classes (in transform) as class ``0``
    """  # noqa: E501

    def __init__(self, add_nan: bool = False, warn: bool = True):
        """
        init NaNLabelEncoder

        Returns
        -------
        add_nan
            if to force encoding of nan at 0
        warn
            if to warn if additional nans are added because items are unknown
        """
        self.add_nan = add_nan
        self.warn = warn
        super().__init__()

    def fit_transform(self, y: pd.Series, overwrite: bool = False) -> np.ndarray:
        """
        Fit and transform data.

        Parameters
        ----------
        y: pd.Series
            input data
        overwrite: bool
            if to overwrite current mappings or if to add to it.

        Returns
        -------
        np.ndarray
            encoded data
        """
        self.fit(y, overwrite=overwrite)
        return self.transform(y)

    @staticmethod
    def is_numeric(y: pd.Series) -> bool:
        """
        Determine if series is numeric or not. Will also return True
        if series is a categorical type with underlying integers.

        Parameters
        ----------
        y: pd.Series
            series for which to carry out assessment

        Returns
        -------
        bool
            True if series is numeric
        """
        return y.dtype.kind in "bcif" or (
            isinstance(y.dtype, pd.CategoricalDtype)
            and y.cat.categories.dtype.kind in "bcif"
        )

    def fit(self, y: pd.Series, overwrite: bool = False):
        """
        Fit transformer

        Parameters
        ----------
        y: pd.Series
            input data to fit on
        overwrite: bool
            whether to overwrite current mappings or if to add to it.

        Returns
        -------
        NaNLabelEncoder: self
        """
        if not overwrite and hasattr(self, "classes_"):
            offset = len(self.classes_)
        else:
            offset = 0
            self.classes_ = {}

        # determine new classes
        if self.add_nan:
            if self.is_numeric(y):
                nan = np.nan
            else:
                nan = "nan"
            self.classes_[nan] = 0
            idx = 1
        else:
            idx = 0

        idx += offset
        for val in np.unique(y):
            if val not in self.classes_:
                self.classes_[val] = idx
                idx += 1

        self.classes_vector_ = np.array(list(self.classes_.keys()))
        return self

    def transform(
        self,
        y: Iterable,
        return_norm: bool = False,
        target_scale=None,
        ignore_na: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode iterable with integers.

        Parameters
        ----------
        y: Iterable
            iterable to encode
        return_norm
            only exists for compatability with other encoders - returns a tuple if true.
        target_scale
            only exists for compatability with other encoders - has no effect.
        ignore_na: bool
            if to ignore na values and map them to zeros
            (this is different to `add_nan=True` option which maps ONLY NAs to zeros
            while this options maps the first class and NAs to zeros)

        Returns
        -------
        Union[torch.Tensor, np.ndarray]
            returns encoded data as torch tensor or numpy array depending on input type
        """
        if self.add_nan:
            if self.warn:
                cond = np.array([item not in self.classes_ for item in y])
                if cond.any():
                    warnings.warn(
                        (
                            f"Found {np.unique(np.asarray(y)[cond]).size} "
                            "unknown classes which were set to NaN"
                        ),
                        UserWarning,
                    )

            encoded = [self.classes_.get(v, 0) for v in y]

        else:
            if ignore_na:
                na_fill_value = next(iter(self.classes_.values()))
                encoded = [self.classes_.get(v, na_fill_value) for v in y]
            else:
                try:
                    encoded = [self.classes_[v] for v in y]
                except KeyError as e:
                    raise KeyError(
                        f"Unknown category '{e.args[0]}' encountered. "
                        "Set `add_nan=True` to allow unknown categories"
                    )

        if isinstance(y, torch.Tensor):
            encoded = torch.tensor(encoded, dtype=torch.long, device=y.device)
        else:
            encoded = np.array(encoded)

        if return_norm:
            return encoded, self.get_parameters()
        else:
            return encoded

    def inverse_transform(self, y: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Decode data, i.e. transform from integers to labels.

        Parameters
        ----------
        y: Union[torch.Tensor, np.ndarray]
            encoded data

        Raises
        ------
        KeyError
            if unknown elements should be decoded

        Returns
        -------
        np.ndarray
            decoded data
        """
        if y.max() >= len(self.classes_vector_):
            raise KeyError("New unknown values detected")

        # decode
        decoded = self.classes_vector_[y]
        return decoded

    def __call__(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract prediction from network output. Does not map back to input
        categories as this would require a numpy tensor without grad-abilities.

        Parameters
        ----------
        data: dict[str, torch.Tensor]
            Dictionary with entries

            * prediction: data to de-scale
            * target_scale: center and scale of data

        Returns
        -------
        torch.Tensor
            prediction
        """
        return data["prediction"]

    def get_parameters(self, groups=None, group_names=None) -> np.ndarray:
        """
        Get fitted scaling parameters for a given group.

        All parameters are unused - exists for compatability.

        Returns
        -------
        np.ndarray
            zero array.
        """
        return np.zeros(2, dtype=np.float64)


class TorchNormalizer(
    InitialParameterRepresenterMixIn, BaseEstimator, TransformerMixin, TransformMixIn
):
    """
    Basic target transformer that can be fit also on torch tensors.
    """

    def __init__(
        self,
        method: str = "standard",
        center: bool = True,
        transformation: Union[str, Tuple[Callable, Callable]] = None,
        method_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters
        ----------
        method: str, optional, default="standard"
            method to rescale series. Either "identity", "standard" (standard scaling)
            or "robust" (scale using quantiles 0.25-0.75). Defaults to "standard".
        method_kwargs: Dict[str, Any], optional, default=None
            Dictionary of method specific arguments as listed below

            - "robust" method: "upper", "lower", "center" quantiles defaulting to 0.75, 0.25 and 0.5

        center: bool, optional, default=True
            If to center the output to zero. Defaults to True.
        transformation: Union[str, Dict[str, Callable]] optional, default=None
            Transform values before applying normalizer. Available options are

            - None (default): No transformation of values
            - log: Estimate in log-space leading to a multiplicative model
            - log1p: Estimate in log-space but add 1 to values before transforming for stability

                (e.g. if many small values <<1 are present).
                Note, that inverse transform is still only `torch.exp()` and
                not `torch.expm1()`.

            - logit: Apply logit transformation on values that are between 0 and 1
            - count: Apply softplus to output (inverse transformation) and x + 1 to input (transformation)
            - softplus: Apply softplus to output (inverse transformation) and inverse softplus to input (transformation)
            - relu: Apply max(0, x) to output
            - Dict[str, Callable] of PyTorch functions that transforms and inversely transforms values.

                ``forward`` and ``reverse`` entries are required. ``inverse``
                transformation is optional and
                should be defined if ``reverse`` is not the
                inverse of the forward transformation. ``inverse_torch``
                can be defined to provide a torch distribution transform for
                inverse transformations.
        """  # noqa E501
        self.method = method
        assert method in [
            "standard",
            "robust",
            "identity",
        ], f"method has invalid value {method}"
        self.center = center
        self.transformation = transformation
        self.method_kwargs = method_kwargs
        self._method_kwargs = (
            deepcopy(method_kwargs) if method_kwargs is not None else {}
        )

    def get_parameters(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns parameters that were used for encoding.

        Returns
        -------
        torch.Tensor
            First element is center of data and second is scale
        """
        return torch.stack(
            [torch.as_tensor(self.center_), torch.as_tensor(self.scale_)], dim=-1
        )

    def fit(self, y: Union[pd.Series, np.ndarray, torch.Tensor]):
        """
        Fit transformer, i.e. determine center and scale of data

        Parameters
        ----------
        y: Union[pd.Series, np.ndarray, torch.Tensor]
            input data

        Returns
        -------
        TorchNormalizer: self
        """
        y = self.preprocess(y)
        self._set_parameters(y_center=y, y_scale=y)
        return self

    def _set_parameters(
        self,
        y_center: Union[pd.Series, np.ndarray, torch.Tensor],
        y_scale: Union[pd.Series, np.ndarray, torch.Tensor],
    ):
        """
        Calculate parameters for scale and center based on input timeseries

        Parameters
        ----------
        y_center: Union[pd.Series, np.ndarray, torch.Tensor]
            timeseries for calculating center
        y_scale: Union[pd.Series, np.ndarray, torch.Tensor]
            timeseries for calculating scale
        """
        if isinstance(y_center, torch.Tensor):
            eps = torch.finfo(y_center.dtype).eps
        else:
            eps = np.finfo(np.float16).eps
        if self.method == "identity":
            if isinstance(y_center, torch.Tensor):
                self.center_ = torch.zeros(y_center.size()[:-1])
                self.scale_ = torch.ones(y_scale.size()[:-1])
            elif isinstance(y_center, (np.ndarray, pd.Series, pd.DataFrame)):
                # numpy default type is numpy.float64 while torch default
                # type is torch.float32 (if not changed)
                # therefore, we first generate torch tensors
                # (with torch default type) and then
                # convert them to numpy arrays
                self.center_ = torch.zeros(y_center.shape[:-1]).numpy()
                self.scale_ = torch.ones(y_scale.shape[:-1]).numpy()
            else:
                self.center_ = 0.0
                self.scale_ = 1.0

        elif self.method == "standard":
            if isinstance(y_center, torch.Tensor):
                self.center_ = torch.mean(y_center, dim=-1)
                self.scale_ = torch.std(y_scale, dim=-1) + eps
            elif isinstance(y_center, np.ndarray):
                self.center_ = np.mean(y_center, axis=-1)
                self.scale_ = np.std(y_scale, axis=-1) + eps
            else:
                self.center_ = np.mean(y_center)
                self.scale_ = np.std(y_scale) + eps
            # correct numpy scalar dtype promotion, e.g. fix type from
            # `np.float32(0.0) + 1e-8` gives `np.float64(1e-8)`
            if isinstance(self.scale_, np.ndarray):
                self.scale_ = self.scale_.astype(y_scale.dtype)

        elif self.method == "robust":
            if isinstance(y_center, torch.Tensor):
                self.center_ = y_center.quantile(
                    self._method_kwargs.get("center", 0.5), dim=-1
                )
                q_75 = y_scale.quantile(self._method_kwargs.get("upper", 0.75), dim=-1)
                q_25 = y_scale.quantile(self._method_kwargs.get("lower", 0.25), dim=-1)
            elif isinstance(y_center, np.ndarray):
                self.center_ = np.percentile(
                    y_center, self._method_kwargs.get("center", 0.5) * 100, axis=-1
                )
                q_75 = np.percentile(
                    y_scale, self._method_kwargs.get("upper", 0.75) * 100, axis=-1
                )
                q_25 = np.percentile(
                    y_scale, self._method_kwargs.get("lower", 0.25) * 100, axis=-1
                )
            else:
                self.center_ = np.percentile(
                    y_center, self._method_kwargs.get("center", 0.5) * 100, axis=-1
                )
                q_75 = np.percentile(
                    y_scale, self._method_kwargs.get("upper", 0.75) * 100
                )
                q_25 = np.percentile(
                    y_scale, self._method_kwargs.get("lower", 0.25) * 100
                )
            self.scale_ = (q_75 - q_25) / 2.0 + eps
        if not self.center and self.method != "identity":
            self.scale_ = self.center_
            if isinstance(y_center, torch.Tensor):
                self.center_ = torch.zeros_like(self.center_)
            else:
                self.center_ = np.zeros_like(self.center_)

        if (np.asarray(self.scale_) < 1e-7).any():
            warnings.warn(
                "scale is below 1e-7 - consider not centering "
                "the data or using data with higher variance for numerical stability",
                UserWarning,
            )

    def transform(
        self,
        y: Union[pd.Series, np.ndarray, torch.Tensor],
        return_norm: bool = False,
        target_scale: torch.Tensor = None,
    ) -> Union[
        Tuple[Union[np.ndarray, torch.Tensor], np.ndarray],
        Union[np.ndarray, torch.Tensor],
    ]:
        """
        Rescale data.

        Parameters
        ----------
        y: Union[pd.Series, np.ndarray, torch.Tensor]
            input data
        return_norm: bool, optional, default=False
            [description]. Defaults to False.
        target_scale: torch.Tensor, optional, default=None
            target scale to use instead of fitted center and scale

        Returns
        -------
        Union[Tuple[Union[np.ndarray, torch.Tensor],np.ndarray],Union[np.ndarray, torch.Tensor]]
            rescaled data with type depending on input type. returns second element if ``return_norm=True``
        """  # noqa: E501
        y = self.preprocess(y)
        # get center and scale
        if target_scale is None:
            target_scale = self.get_parameters().numpy()[None, :]
        center = target_scale[..., 0]
        scale = target_scale[..., 1]
        if y.ndim > center.ndim:  # multiple batches -> expand size
            center = center.view(*center.size(), *(1,) * (y.ndim - center.ndim))
            scale = scale.view(*scale.size(), *(1,) * (y.ndim - scale.ndim))

        # transform
        dtype = y.dtype
        y = (y - center) / scale
        try:
            y = y.astype(dtype)
        except AttributeError:  # torch.Tensor has `.type()` instead of `.astype()`
            y = y.type(dtype)

        # return with center and scale or without
        if return_norm:
            return y, target_scale
        else:
            return y

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        """
        Inverse scale.

        Parameters
        ----------
        y: torch.Tensor
            scaled data

        Returns
        -------
        torch.Tensor
            de-scaled data
        """
        return self(dict(prediction=y, target_scale=self.get_parameters().unsqueeze(0)))

    def __call__(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inverse transformation but with network output as input.

        Parameters
        ----------
        data: Dict[str, torch.Tensor]
            Dictionary with entries

            * prediction: data to de-scale
            * target_scale: center and scale of data

        Returns
        -------
        torch.Tensor
            de-scaled data
        """
        # ensure output dtype matches input dtype
        dtype = data["prediction"].dtype

        # inverse transformation with tensors
        norm = data["target_scale"]

        # use correct shape for norm
        if data["prediction"].ndim > norm.ndim:
            norm = norm.unsqueeze(-1)

        # transform
        y = data["prediction"] * norm[:, 1, None] + norm[:, 0, None]

        y = self.inverse_preprocess(y)

        # return correct shape
        if data["prediction"].ndim == 1 and y.ndim > 1:
            y = y.squeeze(0)
        return y.type(dtype)


class EncoderNormalizer(TorchNormalizer):
    """
    Special Normalizer that is fit on each encoding sequence.

    If used, this transformer will be fitted on each encoder sequence separately.
    This normalizer can be particularly useful as target normalizer.
    """

    def __init__(
        self,
        method: str = "standard",
        center: bool = True,
        max_length: Union[int, List[int]] = None,
        transformation: Union[str, Tuple[Callable, Callable]] = None,
        method_kwargs: Dict[str, Any] = None,
    ):
        """
        Initialize

        Parameters
        ----------
        method: str, optional, default="standard"
            method to rescale series. Either "identity", "standard" (standard scaling)
            or "robust" (scale using quantiles 0.25-0.75). Defaults to "standard".
        method_kwargs: Dict[str, Any], optional, default=None
            Dictionary of method specific arguments as listed below

                * "robust" method: "upper", "lower", "center" quantiles defaulting to 0.75, 0.25 and 0.5

        center: bool, optional, default=True
            If to center the output to zero. Defaults to True.
        max_length: Union[int, List[int]], optional
            Maximum length to take into account for calculating parameters.
            If tuple, first length is maximum length for calculating center and second is maximum
            length for calculating scale. Defaults to entire length of time series.
        transformation: Union[str, Tuple[Callable, Callable]] optional:
            Transform values before applying normalizer. Available options are

                * None (default): No transformation of values
                * log: Estimate in log-space leading to a multiplicative model
                * log1p: Estimate in log-space but add 1 to values before transforming for stability

                    (e.g. if many small values <<1 are present).
                    Note, that inverse transform is still only `torch.exp()` and not `torch.expm1()`.

                * logit: Apply logit transformation on values that are between 0 and 1
                * count: Apply softplus to output (inverse transformation) and x + 1 to input (transformation)
                * softplus: Apply softplus to output (inverse transformation) and inverse softplus to input (transformation)
                * relu: Apply max(0, x) to output
                * Dict[str, Callable] of PyTorch functions that transforms and inversely transforms values.

                  ``forward`` and ``reverse`` entries are required. ``inverse`` transformation is optional and
                  should be defined if ``reverse`` is not the inverse of the forward transformation. ``inverse_torch``
                  can be defined to provide a torch distribution transform for inverse transformations.
        """  # noqa: E501
        method_kwargs = deepcopy(method_kwargs) if method_kwargs is not None else {}
        super().__init__(
            method=method,
            center=center,
            transformation=transformation,
            method_kwargs=method_kwargs,
        )
        self.max_length = max_length

    def fit(self, y: Union[pd.Series, np.ndarray, torch.Tensor]):
        """
        Fit transformer, i.e. determine center and scale of data

        Parameters
        ----------
        y: Union[pd.Series, np.ndarray, torch.Tensor]
            input data

        Returns
        -------
        TorchNormalizer: self
        """
        # reduce size of time series - take only max length
        if self.max_length is None:
            y_center = y_scale = self.preprocess(y)
        elif isinstance(self.max_length, int):
            y_center = y_scale = self.preprocess(
                self._slice(y, slice(-self.max_length, None))
            )
        else:
            y = self.preprocess(self._slice(y, slice(-max(self.max_length), None)))
            if np.argmax(self.max_length) == 0:
                y_center = y
                y_scale = self._slice(y, slice(-self.max_length[1], None))
            else:
                y_center = self._slice(y, slice(-self.max_length[0], None))
                y_scale = y
        # set parameters for normalization
        self._set_parameters(y_center=y_center, y_scale=y_scale)
        return self

    @property
    def min_length(self):
        if self.method == "identity":
            return 0  # no timeseries properties used
        elif self.center:
            return 1  # only calculation of mean required
        else:
            return 2  # requires std, i.e. at least 2 entries

    @staticmethod
    def _slice(
        x: Union[pd.DataFrame, pd.Series, np.ndarray, torch.Tensor], s: slice
    ) -> Union[pd.DataFrame, pd.Series, np.ndarray, torch.Tensor]:
        """
        Slice pandas data frames, numpy arrays and tensors.

        Parameters
        ----------
        x: Union[pd.Series, np.ndarray, torch.Tensor]
            object to slice
        s: slice
            slice, e.g. ``slice(None, -5)```

        Returns
        -------
        Union[pd.Series, np.ndarray, torch.Tensor]
            sliced object
        """

        if isinstance(x, (pd.DataFrame, pd.Series)):
            return x[s]
        else:
            return x[..., s]


class GroupNormalizer(TorchNormalizer):
    """
    Normalizer that scales by groups.

    For each group a scaler is fitted and applied. This scaler can be used
    as target normalizer or also to normalize any other variable.
    """

    def __init__(
        self,
        method: str = "standard",
        groups: Optional[List[str]] = None,
        center: bool = True,
        scale_by_group: bool = False,
        transformation: Optional[Union[str, Tuple[Callable, Callable]]] = None,
        method_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Group normalizer to normalize a given entry by groups. Can be used as target normalizer.

        Parameters
        ----------
        method: str, optional, default="standard"
            method to rescale series. Either "standard" (standard scaling) or "robust"
            (scale using quantiles 0.25-0.75). Defaults to "standard".
        method_kwargs: Dict[str, Any], optional, default=None
            Dictionary of method specific arguments as listed below

                * "robust" method: "upper", "lower", "center" quantiles defaulting to 0.75, 0.25 and 0.5

        groups: List[str], optional, default=[]
            Group names to normalize by. Defaults to [].
        center: bool, optional, default=True
            If to center the output to zero. Defaults to True.
        scale_by_group: bool, optional
            If to scale the output by group, i.e. norm is calculated as
            ``(group1_norm * group2_norm * ...) ^ (1 / n_groups)``. Defaults to False.
        transformation: Union[str, Tuple[Callable, Callable]] optional, default=None):
            Transform values before applying normalizer. Available options are

                * None (default): No transformation of values
                * log: Estimate in log-space leading to a multiplicative model
                * log1p: Estimate in log-space but add 1 to values before transforming for stability

                    (e.g. if many small values <<1 are present).
                    Note, that inverse transform is still only `torch.exp()` and not `torch.expm1()`.

                * logit: Apply logit transformation on values that are between 0 and 1
                * count: Apply softplus to output (inverse transformation) and x + 1 to input
                    (transformation)
                * softplus: Apply softplus to output (inverse transformation) and inverse softplus to input
                    (transformation)
                * relu: Apply max(0, x) to output
                * Dict[str, Callable] of PyTorch functions that transforms and inversely transforms values.
                  ``forward`` and ``reverse`` entries are required. ``inverse`` transformation is optional and
                  should be defined if ``reverse`` is not the inverse of the forward transformation. ``inverse_torch``
                  can be defined to provide a torch distribution transform for inverse transformations.

        """  # noqa: E501
        self.groups = groups
        self._groups = list(groups) if groups is not None else []
        self.scale_by_group = scale_by_group
        method_kwargs = deepcopy(method_kwargs) if method_kwargs is not None else {}
        super().__init__(
            method=method,
            center=center,
            transformation=transformation,
            method_kwargs=method_kwargs,
        )

    def fit(self, y: pd.Series, X: pd.DataFrame):
        """
        Determine scales for each group

        Parameters
        ----------
        y: pd.Series
            input data
        X: pd.DataFrame
            dataframe with columns for each group defined in ``groups`` parameter.

        Returns
        -------
        self
        """
        y = self.preprocess(y)
        eps = np.finfo(np.float16).eps
        if len(self._groups) == 0:
            assert (
                not self.scale_by_group
            ), "No groups are defined, i.e. `scale_by_group=[]`"
            if self.method == "standard":
                self.norm_ = {
                    "center": np.mean(y),
                    "scale": np.std(y) + eps,
                }  # center and scale
            else:
                quantiles = np.quantile(
                    y,
                    [
                        self._method_kwargs.get("lower", 0.25),
                        self._method_kwargs.get("center", 0.5),
                        self._method_kwargs.get("upper", 0.75),
                    ],
                )
                self.norm_ = {
                    "center": quantiles[1],
                    "scale": (quantiles[2] - quantiles[0]) / 2.0 + eps,
                }  # center and scale
            if not self.center:
                self.norm_["scale"] = self.norm_["center"] + eps
                self.norm_["center"] = 0.0

        elif self.scale_by_group:
            if self.method == "standard":
                self.norm_ = {
                    g: X[[g]]
                    .assign(y=y)
                    .groupby(g, observed=True)
                    .agg(center=("y", "mean"), scale=("y", "std"))
                    .assign(center=lambda x: x["center"], scale=lambda x: x.scale + eps)
                    for g in self._groups
                }
            else:
                self.norm_ = {
                    g: X[[g]]
                    .assign(y=y)
                    .groupby(g, observed=True)
                    .y.quantile(
                        [
                            self._method_kwargs.get("lower", 0.25),
                            self._method_kwargs.get("center", 0.5),
                            self._method_kwargs.get("upper", 0.75),
                        ]
                    )
                    .unstack(-1)
                    .assign(
                        center=lambda x: x[self._method_kwargs.get("center", 0.5)],
                        scale=lambda x: (
                            x[self._method_kwargs.get("upper", 0.75)]
                            - x[self._method_kwargs.get("lower", 0.25)]
                        )
                        / 2.0
                        + eps,
                    )[["center", "scale"]]
                    for g in self._groups
                }
            # calculate missings
            if not self.center:  # swap center and scale

                def swap_parameters(norm):
                    norm["scale"] = norm["center"] + eps
                    norm["center"] = 0.0
                    return norm

                self.norm_ = {
                    g: swap_parameters(norm) for g, norm in self.norm_.items()
                }
            self.missing_ = {
                group: scales.median().to_dict() for group, scales in self.norm_.items()
            }

        else:
            if self.method == "standard":
                self.norm_ = (
                    X[self._groups]
                    .assign(y=y)
                    .groupby(self._groups, observed=True)
                    .agg(center=("y", "mean"), scale=("y", "std"))
                    .assign(center=lambda x: x["center"], scale=lambda x: x.scale + eps)
                )
            else:
                self.norm_ = (
                    X[self._groups]
                    .assign(y=y)
                    .groupby(self._groups, observed=True)
                    .y.quantile(
                        [
                            self._method_kwargs.get("lower", 0.25),
                            self._method_kwargs.get("center", 0.5),
                            self._method_kwargs.get("upper", 0.75),
                        ]
                    )
                    .unstack(-1)
                    .assign(
                        center=lambda x: x[self._method_kwargs.get("center", 0.5)],
                        scale=lambda x: (
                            x[self._method_kwargs.get("upper", 0.75)]
                            - x[self._method_kwargs.get("lower", 0.25)]
                        )
                        / 2.0
                        + eps,
                    )[["center", "scale"]]
                )
            if not self.center:  # swap center and scale
                self.norm_["scale"] = self.norm_["center"] + eps
                self.norm_["center"] = 0.0
            self.missing_ = self.norm_.median().to_dict()

        if (
            (
                self.scale_by_group
                and any(
                    (self.norm_[group]["scale"] < 1e-7).any() for group in self._groups
                )
            )
            or (
                not self.scale_by_group
                and isinstance(self.norm_["scale"], float)
                and self.norm_["scale"] < 1e-7
            )
            or (
                not self.scale_by_group
                and not isinstance(self.norm_["scale"], float)
                and (self.norm_["scale"] < 1e-7).any()
            )
        ):
            warnings.warn(
                "scale is below 1e-7 - consider not centering "
                "the data or using data with higher variance for numerical stability",
                UserWarning,
            )

        return self

    @property
    def names(self) -> List[str]:
        """
        Names of determined scales.

        Returns
        -------
        List[str]
            list of names
        """
        return ["center", "scale"]

    def fit_transform(
        self, y: pd.Series, X: pd.DataFrame, return_norm: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Fit normalizer and scale input data.

        Parameters
        ----------
        y: pd.Series
            data to scale
        X: pd.DataFrame
            dataframe with ``groups`` columns
        return_norm: bool, optional, default=False
            If to return . Defaults to False.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Scaled data, if ``return_norm=True``, returns also scales as second element
        """
        return self.fit(y, X).transform(y, X, return_norm=return_norm)

    def inverse_transform(self, y: pd.Series, X: pd.DataFrame):
        """
        Rescaling data to original scale - not implemented - call class with target scale instead.
        """  # noqa: E501
        raise NotImplementedError()

    def transform(
        self,
        y: pd.Series,
        X: pd.DataFrame = None,
        return_norm: bool = False,
        target_scale: torch.Tensor = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Scale input data.

        Parameters
        ----------
        y: pd.Series
            data to scale
        X: pd.DataFrame
            dataframe with ``groups`` columns
        return_norm: bool, optional, default=False
            If to return . Defaults to False.
        target_scale: torch.Tensor
            target scale to use instead of fitted center and scale

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Scaled data, if ``return_norm=True``, returns also scales as second element
        """
        # # check if arguments are wrong way round
        if isinstance(y, pd.DataFrame) and not isinstance(X, pd.DataFrame):
            raise ValueError("X and y is in wrong positions")
        if target_scale is None:
            assert X is not None, "either target_scale or X has to be passed"
            target_scale = self.get_norm(X)
        return super().transform(y, return_norm=return_norm, target_scale=target_scale)

    def get_parameters(
        self, groups: Union[torch.Tensor, list, tuple], group_names: List[str] = None
    ) -> np.ndarray:
        """
        Get fitted scaling parameters for a given group.

        Parameters
        ----------
        groups: Union[torch.Tensor, list, tuple]
            group ids for which to get parameters
        group_names: List[str], optional, default=None
            Names of groups corresponding to positions in ``groups``.
            Defaults to None, i.e. the instance attribute ``groups``.

        Returns
        -------
        np.ndarray
            parameters used for scaling
        """
        if isinstance(groups, torch.Tensor):
            groups = groups.tolist()
        if isinstance(groups, list):
            groups = tuple(groups)
        if group_names is None:
            group_names = self._groups
        else:
            # filter group names
            group_names = [name for name in group_names if name in self._groups]
        assert len(group_names) == len(
            self._groups
        ), "Passed groups and fitted do not match"

        if len(self._groups) == 0:
            params = np.array([self.norm_["center"], self.norm_["scale"]])
        elif self.scale_by_group:
            norm = np.array([1.0, 1.0])
            for group, group_name in zip(groups, group_names):
                try:
                    norm = norm * self.norm_[group_name].loc[group].to_numpy()
                except KeyError:
                    norm = norm * np.asarray(
                        [self.missing_[group_name][name] for name in self.names]
                    )
            norm = np.power(norm, 1.0 / len(self._groups))
            params = norm
        else:
            try:
                params = self.norm_.loc[groups].to_numpy()
            except (KeyError, TypeError):
                params = np.asarray([self.missing_[name] for name in self.names])
        return params

    def get_norm(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get scaling parameters for multiple groups.

        Parameters
        ----------
        X: pd.DataFrame
            dataframe with ``groups`` columns

        Returns
        -------
        pd.DataFrame
            dataframe with scaling parameterswhere each row corresponds
            to the input dataframe
        """
        if len(self._groups) == 0:
            norm = np.asarray([self.norm_["center"], self.norm_["scale"]]).reshape(
                1, -1
            )
        elif self.scale_by_group:
            norm = [
                np.prod(
                    [
                        X[group_name]
                        .map(self.norm_[group_name][name])
                        .fillna(self.missing_[group_name][name])
                        .to_numpy()
                        for group_name in self._groups
                    ],
                    axis=0,
                )
                for name in self.names
            ]
            norm = np.power(np.stack(norm, axis=1), 1.0 / len(self._groups))
        else:
            norm = (
                X[self._groups]
                .set_index(self._groups)
                .join(self.norm_)
                .fillna(self.missing_)
                .to_numpy()
            )
        return norm


class MultiNormalizer(TorchNormalizer):
    """
    Normalizer for multiple targets.

    This normalizers wraps multiple other normalizers.
    """

    def __init__(self, normalizers: List[TorchNormalizer]):
        """
        Parameters
        ----------
        normalizers: List[TorchNormalizer]
            list of normalizers to apply to targets
        """
        self.normalizers = normalizers

    def fit(
        self, y: Union[pd.DataFrame, np.ndarray, torch.Tensor], X: pd.DataFrame = None
    ):
        """
        Fit transformer, i.e. determine center and scale of data

        Parameters
        ----------
        y: Union[pd.Series, np.ndarray, torch.Tensor]
            input data

        Returns
        -------
        MultiNormalizer: self
        """
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        for idx, normalizer in enumerate(self.normalizers):
            if isinstance(normalizer, GroupNormalizer):
                normalizer.fit(y[:, idx], X)
            else:
                normalizer.fit(y[:, idx])

        self.fitted_ = True
        return self

    def __getitem__(self, idx: int):
        """
        Return normalizer.

        Parameters
        ----------
        idx: int
            metric index
        """
        return self.normalizers[idx]

    def __iter__(self):
        """
        Iter over normalizers.
        """
        return iter(self.normalizers)

    def __len__(self) -> int:
        """
        Number of normalizers.
        """
        return len(self.normalizers)

    def transform(
        self,
        y: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        X: pd.DataFrame = None,
        return_norm: bool = False,
        target_scale: List[torch.Tensor] = None,
    ) -> Union[
        List[Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]],
        List[Union[np.ndarray, torch.Tensor]],
    ]:
        """
        Scale input data.

        Parameters
        ----------
        y: Union[pd.DataFrame, np.ndarray, torch.Tensor]
            data to scale
        X: pd.DataFrame
            dataframe with ``groups`` columns. Only necessary
            if :py:class:`~GroupNormalizer` is among normalizers
        return_norm: bool, optional
            If to return . Defaults to False.
        target_scale: List[torch.Tensor]
            target scale to use instead of fitted center and scale

        Returns
        -------
        Union[List[Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]], List[Union[np.ndarray, torch.Tensor]]]
                List of scaled data, if ``return_norm=True``, returns also scales as second element
        """  # noqa: E501
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy().transpose()

        res = []
        for idx, normalizer in enumerate(self.normalizers):
            if target_scale is not None:
                scale = target_scale[idx]
            else:
                scale = None
            if isinstance(normalizer, GroupNormalizer):
                r = normalizer.transform(
                    y[idx], X, return_norm=return_norm, target_scale=scale
                )
            else:
                r = normalizer.transform(
                    y[idx], return_norm=return_norm, target_scale=scale
                )
            res.append(r)

        if return_norm:
            return [r[0] for r in res], [r[1] for r in res]
        else:
            return res

    def __call__(
        self, data: Dict[str, Union[List[torch.Tensor], torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Inverse transformation but with network output as input.

        Parameters
        ----------
        data: Dict[str, Union[List[torch.Tensor], torch.Tensor]])
            Dictionary with entries

            * prediction: list of data to de-scale
            * target_scale: list of center and scale of data

        Returns
        -------
        List[torch.Tensor]
            list of de-scaled data
        """
        denormalized = [
            normalizer(
                dict(
                    prediction=data["prediction"][idx],
                    target_scale=data["target_scale"][idx],
                )
            )
            for idx, normalizer in enumerate(self.normalizers)
        ]
        return denormalized

    def get_parameters(self, *args, **kwargs) -> List[torch.Tensor]:
        """
        Returns parameters that were used for encoding.

        Returns
        -------
        List[torch.Tensor]
            First element is center of data and second is scale
        """
        return [
            normalizer.get_parameters(*args, **kwargs)
            for normalizer in self.normalizers
        ]

    def __getattr__(self, name: str):
        """
        Return dynamically attributes.

        Return attributes if defined in this class. If not,
        create dynamically attributes based on attributes of underlying normalizers
        that are lists. Create functions if necessary. Arguments to functions are
        distributed to the functions if they are lists and their length matches the
        number of normalizers. Otherwise, they are directly passed to each callable of
        the normalizers.

        Parameters
        name: str
            name of attribute

        Returns
        -------
        ret
            attributes of this class or list of attributes of underlying class
        """
        try:
            return super().__getattr__(name)
        except AttributeError as e:
            attribute_exists = all(hasattr(norm, name) for norm in self.normalizers)
            if attribute_exists:
                # check if to return callable or not and return function if yes
                if callable(getattr(self.normalizers[0], name)):
                    n = len(self.normalizers)

                    def func(*args, **kwargs):
                        # if arg/kwarg is list and of length normalizers,
                        # then apply each part to a normalizer.
                        #  otherwise pass it directly to all normalizers
                        results = []
                        for idx, norm in enumerate(self.normalizers):
                            new_args = [
                                (
                                    arg[idx]
                                    if isinstance(arg, (list, tuple))
                                    and not isinstance(arg, rnn.PackedSequence)
                                    and len(arg) == n
                                    else arg
                                )
                                for arg in args
                            ]
                            new_kwargs = {
                                key: (
                                    val[idx]
                                    if isinstance(val, list)
                                    and not isinstance(val, rnn.PackedSequence)
                                    and len(val) == n
                                    else val
                                )
                                for key, val in kwargs.items()
                            }
                            results.append(getattr(norm, name)(*new_args, **new_kwargs))
                        return results

                    return func
                else:
                    # else return list of attributes
                    return [getattr(norm, name) for norm in self.normalizers]
            else:  # attribute does not exist for all normalizers
                raise e
