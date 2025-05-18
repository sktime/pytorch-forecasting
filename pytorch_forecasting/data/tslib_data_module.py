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
    context_length: int
        The length of the context window for the model. This is the number of time steps
        used as input to the model.
    prediction_length: int
        The length of the prediction window for the model. This is the number of time
        steps to be predicted by the model.
    freq: str, default = "h"
        The frequency of the time series data. This is used to determine the time steps
        for the model.
    features: str = "MS"
        Feature combination mode:
          - "S": Single variable forecasting (target only)
          - "M": Multivariate forecasting, using all variables
          - "MS": Multivariate to single, using all variables to predict target
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
    shuffle : bool, default=True
        Whether to shuffle the data at every epoch.
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
        context_length: int,
        prediction_length: int,
        freq: str = "h",
        features: str = "MS",
        add_relative_time_idx: bool = False,
        add_target_scales: bool = False,
        target_normalizer: Union[
            NORMALIZER, str, List[NORMALIZER], Tuple[NORMALIZER], None
        ] = "auto",  # noqa: E501
        scalers: Optional[
            Dict[str, Union[StandardScaler, RobustScaler, TorchNormalizer]]
        ] = None,  # noqa: E501
        shuffle: bool = True,
        batch_size: int = 32,
        num_workers: int = 0,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        collate_fn: Optional[callable] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.time_series_dataset = time_series_dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.freq = freq
        self.features = features
        self.add_relative_time_idx = add_relative_time_idx
        self.add_target_scales = add_target_scales
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.collate_fn = collate_fn
        self.kwargs = kwargs

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
        self.shuffle = shuffle

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.time_series_metadata = time_series_dataset.get_metadata()

    def _prepare_metadata(self) -> None:
        """
        Prepare metadata for `tslib` time series data module.

        Returns
        -------
        dict containing the following as keys:
            - feature_names: Dict[str, List[str]]
                Dictionary of feature names for each feature type.
            - feature_indices: Dict[str, List[int]]
                Dictionary of feature indices for each feature type.
            - n_features: Dict[str, int]
                Dictionary of number of features for each feature type.
            - context_length: int
                Length of the context window for the model, as set in the data module.
            - prediction_length: int
                Length of the prediction window for the model, as set in the data
                module.
            - freq: str or None
            - features: str
                Feature combination mode.
        """
        # TODO: include handling for datasets without get_metadata()
        ds_metadata = self.time_series_metadata.get_metadata()

        feature_names = {
            "categorical": [],
            "continuous": [],
            "static": [],
            "known": [],
            "unknown": [],
            "target": [],
            "all": [],
        }

        feature_indices = {
            "categorical": [],
            "continuous": [],
            "static": [],
            "known": [],
            "unknown": [],
            "target": [],
            "all": [],
        }

        cols = ds_metadata.get("cols", {})
        col_type = ds_metadata.get("col_type", {})
        col_known = ds_metadata.get("col_known", {})

        all_features = cols.get("x", [])
        static_features = cols.get("st", [])
        target_features = cols.get("y", [])
        feature_names["all"] = list(all_features)
        feature_names["static"] = list(static_features)
        feature_names["target"] = list(target_features)

        for idx, col in enumerate(all_features):
            if col_type.get(col, "F") == "C":
                feature_names["categorical"].append(col)
                feature_indices["categorical"].append(idx)
            else:
                feature_names["continuous"].append(col)
                feature_indices["continuous"].append(idx)

            if col_known.get(col, "U") == "K":
                feature_names["known"].append(col)
                feature_indices["known"].append(idx)
            else:
                feature_names["unknown"].append(col)
                feature_indices["unknown"].append(idx)

        static_cat_names, static_cont_names = [], []
        for col in static_features:
            if col_type.get(col, "F") == "C":
                static_cat_names.append(col)
            else:
                static_cont_names.append(col)

        feature_names["static_categorical"] = static_cat_names
        feature_names["static_continuous"] = static_cont_names

        for idx, col in enumerate(target_features):
            feature_indices["target"].append(idx)

        n_features = {k: len(v) for k, v in feature_names.items()}

        metadata = dict(
            feature_names=feature_names,
            feature_indices=feature_indices,
            n_features=n_features,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            freq=self.freq,
            features=self.features,
        )

        return metadata
