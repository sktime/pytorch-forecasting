from abc import abstractmethod

import pandas as pd
import torch

from pytorch_forecasting import EncoderNormalizer, GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.adapters.utils import (
    ArrayLike,
    _is_sklearn_transformer,
    _series_from,
    _to_numpy,
    _to_tensor,
    _was_2d_singleton,
)
from pytorch_forecasting.base._base_object import _BaseObject


class ScalerStrategy(_BaseObject):
    """Default behavior for scalers."""

    _tags = {
        "object_type": "scaler_strategy",
        "fit_per_sequence": False,
        "is_label_encoder": False,
    }

    @staticmethod
    @abstractmethod
    def _is_applicable(scaler) -> bool:
        """Whether the scaler follows the given strategy."""

    def prepare_input(self, data: ArrayLike) -> ArrayLike:
        t = _to_tensor(data)
        return t.squeeze(-1) if (t.ndim == 2 and t.shape[1] == 1) else t

    def fit(self, scaler, prepared: ArrayLike, X: pd.DataFrame = None) -> None:
        scaler.fit(prepared)

    def transform(
        self, scaler, prepared: ArrayLike, data: ArrayLike, X: pd.DataFrame = None
    ) -> torch.Tensor:
        result = _to_tensor(scaler.transform(prepared))
        return result.unsqueeze(-1) if _was_2d_singleton(data) else result


class EncoderNormalizerStrategy(ScalerStrategy):
    """EncoderNormalizer must be re-fit per encoder window."""

    _tags = {
        "fit_per_sequence": True,
    }

    @staticmethod
    def _is_applicable(scaler) -> bool:
        return isinstance(scaler, EncoderNormalizer)


class SklearnStrategy(ScalerStrategy):
    """sklearn scalers expect/return 2D numpy arrays."""

    @staticmethod
    def _is_applicable(scaler) -> bool:
        return _is_sklearn_transformer(scaler)

    def prepare_input(self, data: ArrayLike) -> ArrayLike:
        return _to_numpy(data).reshape(-1, 1)

    def fit(self, scaler, prepared: ArrayLike, X: pd.DataFrame = None) -> None:
        scaler.fit(prepared)

    def transform(
        self, scaler, prepared: ArrayLike, data: ArrayLike, X: pd.DataFrame = None
    ) -> torch.Tensor:
        original_shape = _to_numpy(data).shape
        result = scaler.transform(prepared).reshape(original_shape)
        return torch.tensor(result, dtype=torch.float32)


class LabelEncoderStrategy(ScalerStrategy):
    _tags = {
        "is_label_encoder": True,
    }

    @staticmethod
    def _is_applicable(scaler) -> bool:
        return isinstance(scaler, NaNLabelEncoder)

    def prepare_input(self, data: ArrayLike) -> ArrayLike:
        return _series_from(data)

    def transform(
        self, scaler, prepared: ArrayLike, data: ArrayLike, X: pd.DataFrame = None
    ) -> torch.Tensor:
        result = _to_tensor(scaler.transform(prepared))
        return result.unsqueeze(-1) if _was_2d_singleton(data) else result


class GroupNormalizerStrategy(ScalerStrategy):
    @staticmethod
    def _is_applicable(scaler) -> bool:
        return isinstance(scaler, GroupNormalizer)

    def prepare_input(self, data: ArrayLike) -> ArrayLike:
        return _series_from(data)

    def fit(self, scaler, prepared: ArrayLike, X: pd.DataFrame = None) -> None:
        assert X is not None, (
            "GroupNormalizer requires X (DataFrame with group columns) "
            "to be passed to fit()."
        )
        scaler.fit(prepared, X)

    def transform(
        self, scaler, prepared: ArrayLike, data: ArrayLike, X: pd.DataFrame = None
    ) -> torch.Tensor:
        assert X is not None, (
            "GroupNormalizer requires X (DataFrame with group columns) "
            "to be passed to transform()."
        )
        input_was_2d = isinstance(data, torch.Tensor) and data.ndim == 2
        result = _to_tensor(scaler.transform(prepared, X))
        return result.unsqueeze(-1) if input_was_2d else result
