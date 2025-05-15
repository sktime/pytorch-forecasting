"""
Experimmental data module for integrating `tslib` time series deep learning library.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

from lightning.pytorch import LightningDataModule
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)
from pytorch_forecasting.data.timeseries._timeseries_v2 import TimeSeries
from pytorch_forecasting.utils._coerce import _coerce_to_dict

NORMALIZER = Union[TorchNormalizer, EncoderNormalizer, NaNLabelEncoder]


class TslibDataModule(LightningDataModule):
    """
    Experimental data module for integrating `tslib` time series into
    PyTorch Forecasting.

    This module serves as the D2 layer for `tslib` models including transformer-based
    architectures like Informer, AutoFormer, TimeXer and other model deep learning model
    architectures.

    Parameters
    ----------
    time_series_dataset: TimeSeries
        The time series dataset to be used for training and validation. This is the
        newly implemented D1 layer.
    model_family: str = "transformer"
        The model family to be used. Currently, only "transformer" is supported. Ensures
        modularity and extensibility for future model families, while considering the
        diversity of model architectures currently available in the `tslib` library.
    context_length: int = 96
        The length of the context window for the model. This is the number of time steps
        used as input to the model.
    prediction_length: int = 24
        The length of the prediction window for the model. This is the number of time
        steps to be predicted by the model.
    freq: str = "h"
        The frequency of the time series data. This is used to determine the time steps
        for the model.
    add_time_features: bool = True
        Whether to add frequency-based time features to the model.
    features: str = "MS"
        features : str, default="MS"
        Feature combination mode:
          - "S": Single variable forecasting (target only)
          - "M": Multivariate forecasting, using all variables
          - "MS": Multivariate to single, using all variables to predict target
    min_prediction_idx: Optional[int] = None
        The minimum index for the prediction window. This is used to ensure that the
        prediction window does not exceed the length of the time series.
    allow_missing_timesteps: bool = False
        Whether to allow missing timesteps in the time series. If True, the model will
        be able to handle missing timesteps in the input data.
    add_relative_time_idx: bool =  False
        Whether to allow the relative time index to be used with the model.
    add_target_scales: bool = False
        Whether to add target scaling info.
    target_normalizer :
        Union[NORMALIZER, str, List[NORMALIZER], Tuple[NORMALIZER], None],
         default="auto"
        Normalizer for the target variable. If "auto", uses `RobustScaler`.
    scalers : Optional[Dict[str, Union[StandardScaler, RobustScaler, TorchNormalizer]]], default=None #noqa: E501
        Dictionary of feature scalers.
    batch_size : int, default=32
        Batch size for dataloader.
    num_workers : int, default=0
        Number of workers for dataloader.
    train_val_test_split : tuple, default=(0.7, 0.15, 0.15)
        Proportions for train, validation, and test dataset splits.
    """  # noqa: E501

    def __init__(
        self,
        time_series_dataset: TimeSeries,
        model_family: str = "transformer",
        context_length: int = 96,
        prediction_length: int = 24,
        freq: str = "h",
        add_time_features: bool = True,
        features: str = "MS",
        min_prediction_idx: Optional[int] = None,
        allow_missing_timesteps: bool = False,
        add_relative_time_idx: bool = False,
        add_target_scales: bool = False,
        target_normalizer: Union[
            NORMALIZER, str, List[NORMALIZER], Tuple[NORMALIZER], None
        ] = "auto",  # noqa: E501
        scalers: Optional[
            Dict[str, Union[StandardScaler, RobustScaler, TorchNormalizer]]
        ] = None,  # noqa: E501
        batch_size: int = 32,
        num_workers: int = 0,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> None:
        super().__init__()

        self.time_series_dataset = time_series_dataset
        self.model_family = model_family.lower()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.min_prediction_idx = min_prediction_idx
        self.freq = freq
        self.add_time_features = add_time_features
        self.features = features
        self.allow_missing_timesteps = allow_missing_timesteps
        self.add_relative_time_idx = add_relative_time_idx
        self.add_target_scales = add_target_scales
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split

        warnings.warn(
            "TslibDataModule is experimental and subject to change. "
            "The API is not stable and may change without prior warning.",
            UserWarning,
        )

        if isinstance(target_normalizer, str) and target_normalizer.lower() == "auto":
            self._target_normalizer = RobustScaler()
        else:
            self._target_normalizer = target_normalizer

        self.scalers = scalers or {}
        self._train_indices = None
        self._val_indices = None
        self._test_indices = None
        self._metadata = None
        self.time_series_metadata = time_series_dataset.get_metadata()
