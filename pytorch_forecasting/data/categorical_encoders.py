"""
Categorical encoders tailored for the ptf-v2 Datamodules.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import torch


class CategoricalEncoderMixin:
    """Mixin to provide consistent types and tensor outputs for ptf-v2."""

    def _to_numpy(self, y: pd.Series | np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(y, torch.Tensor):
            return y.detach().cpu().numpy()
        elif isinstance(y, pd.Series):
            return y.to_numpy()
        return np.asarray(y)


class PTFOrdinalEncoder(BaseEstimator, TransformerMixin, CategoricalEncoderMixin):
    """
    Ordinal Encoder that assigns a unique integer to each category.
    Maps unknown categories during transformation to an unseen class representation
    (0 or max_class).
    """

    def __init__(self, add_unknown: bool = True):
        self.add_unknown = add_unknown
        self.mapping_ = {}
        self.inverse_mapping_ = {}
        self.unknown_idx = 0

    def fit(self, y: pd.Series | np.ndarray | torch.Tensor) -> "PTFOrdinalEncoder":
        y_np = self._to_numpy(y)
        unique_vals = np.unique(y_np)

        start_idx = 1 if self.add_unknown else 0

        self.mapping_ = {val: idx + start_idx for idx, val in enumerate(unique_vals)}
        self.inverse_mapping_ = {idx: val for val, idx in self.mapping_.items()}

        return self

    def transform(self, y: pd.Series | np.ndarray | torch.Tensor) -> torch.Tensor:
        y_np = self._to_numpy(y)

        # apply mapping
        encoded = np.array([self.mapping_.get(val, self.unknown_idx) for val in y_np])
        return torch.tensor(encoded, dtype=torch.long)


class PTFOneHotEncoder(BaseEstimator, TransformerMixin, CategoricalEncoderMixin):
    """
    One-Hot Encoder that converts categorical variables to binary matrices.
    Returns float32 tensors to seamlessly append to continuous features.
    """

    def __init__(self, add_unknown: bool = True):
        self.add_unknown = add_unknown
        self.mapping_ = {}

    def fit(self, y: pd.Series | np.ndarray | torch.Tensor) -> "PTFOneHotEncoder":
        y_np = self._to_numpy(y)
        self.unique_vals_ = np.unique(y_np)
        self.mapping_ = {val: idx for idx, val in enumerate(self.unique_vals_)}
        self.num_classes_ = len(self.unique_vals_)
        return self

    def transform(self, y: pd.Series | np.ndarray | torch.Tensor) -> torch.Tensor:
        y_np = self._to_numpy(y)
        encoded = np.zeros((len(y_np), self.num_classes_), dtype=np.float32)

        for i, val in enumerate(y_np):
            if val in self.mapping_:
                encoded[i, self.mapping_[val]] = 1.0
            # If unknown, all zeros

        return torch.tensor(encoded, dtype=torch.float32)
