import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
import torch

from pytorch_forecasting.data.encoders import TransformMixIn

ArrayLike = torch.Tensor | np.ndarray | pd.Series


def _to_numpy(data: ArrayLike) -> np.ndarray:
    """Convert any array-like to numpy."""
    if isinstance(data, torch.Tensor):
        return data.detach().numpy()
    elif isinstance(data, pd.Series):
        return data.to_numpy()
    return np.asarray(data)


def _to_tensor(data: ArrayLike, dtype=torch.float32) -> torch.Tensor:
    """Convert any array-like to a float32 tensor."""
    if isinstance(data, torch.Tensor):
        return data.to(dtype)
    elif isinstance(data, pd.Series):
        return torch.tensor(data.to_numpy(), dtype=dtype)
    return torch.tensor(np.asarray(data), dtype=dtype)


def _is_sklearn_transformer(scaler):
    is_sklearn_transform = isinstance(scaler, TransformerMixin)
    is_ptf_transform = isinstance(scaler, TransformMixIn)

    return is_sklearn_transform and not is_ptf_transform


def _series_from(data: ArrayLike) -> pd.Series:
    """Prep for scalers that want a pd.Series (label encoder, group normalizer)."""
    if isinstance(data, pd.Series):
        return data
    np_data = _to_numpy(data)
    return pd.Series(np_data.squeeze() if np_data.ndim == 2 else np_data)


def _was_2d_singleton(data: ArrayLike) -> bool:
    """True if `data` is a torch.Tensor of shape (n, 1)."""
    return isinstance(data, torch.Tensor) and data.ndim == 2 and data.shape[1] == 1
