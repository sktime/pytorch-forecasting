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

# class _TsLibDataset(Dataset):


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
    window_stride : int, default=1
        The stride for the sliding window. This is used to create overlapping windows
        for the data.
    batch_size : int, default=32
        Batch size for dataloader.
    num_workers : int, default=0
        Number of workers for dataloader.
    train_val_test_split : tuple, default=(0.7, 0.15, 0.15)
        Proportions for train, validation, and test dataset splits.
    collate_fn : Optional[callable], default=None
        Custom collate function for the dataloader.
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
            Dict[
                str,
                Union[StandardScaler, RobustScaler, TorchNormalizer, EncoderNormalizer],
            ]
        ] = None,  # noqa: E501
        shuffle: bool = True,
        window_stride: int = 1,
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

        self._metadata = None

        self.categorical_indices = []
        self.continuous_indices = []

        for idx, col in enumerate(self.time_series_dataset["cols"]["x"]):
            if self.time_series_metadata["col_type"].get(col) == "C":
                self.categorical_indices.append(idx)
            else:
                self.continuous_indices.append(idx)

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

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Compute the metadata via the `_prepare_metadata` method.
        This method is called when the `metadata` property is accessed for the first.
        Returns
        -------
        dict
            Metadata for the data module. Refer to the `_prepare_metadata` method for
            the keys and values in the metadata dictionary.
        """
        if not hasattr(self, "_metadata"):
            self._metadata = self._prepare_metadata()
        return self._metadata

    def _process_data(self, idx: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Process the the time series data at the given index, before feeding it
        to the `_TsLibDataset` class.

        Parameters
        ----------
        idx : torch.Tensor
            The index of the time series data to be processed.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the processed data.

        Notes
        -----
        - The target data `y` and features `x` are converted to torch.float32 tensors.
        - The timepoints before the cutoff time are masked off.
        - Splits data into categorical and continous features, which are grouped based on the indices.
        """  # noqa: E501

        series = self.time_series_dataset[idx]
        if series is None:
            raise ValueError(f"series at index {idx} is None. Check the dataset.")
        target = series["y"]
        features = series["x"]
        timestep = series["t"]
        cutoff_time = series["cutoff_time"]

        mask_timestep = torch.tensor(timestep <= cutoff_time, dtype=torch.bool)

        if isinstance(torch, torch.Tensor):
            target = target.float()
        else:
            target = torch.tensor(target, dtype=torch.float32)

        if isinstance(features, torch.Tensor):
            features = features.float()
        else:
            features = torch.tensor(features, dtype=torch.float32)

        # scaling and normlization
        target_scale = {}

        if self._target_normalizer is not None:
            if self.add_target_scales:
                if hasattr(self._target_normalizer, "scale_"):
                    target_scale["scale"] = torch.tensor(self._target_normalizer.scale_)
                if hasattr(self._target_normalizer, "center_"):
                    target_scale["center"] = torch.tensor(
                        self._target_normalizer.center_
                    )  # noqa: E501

            if isinstance(self._target_normalizer, TorchNormalizer):
                target = self._target_normalizer.transform(target)
            else:
                # extra case for handling non-native normalizers
                # apart from those in NORMALIZER.
                target_np = target.reshape(-1, 1).numpy()
                target = torch.tensor(
                    self._target_normalizer.transform(target_np),
                    dtype=torch.float32,
                ).reshape(target.shape)

        if self.scalers:
            feature_indices = self.metadat["feature_indices"]
            feature_names = self.metadata["feature_names"]

            for i, idx in enumerate(feature_indices.get("continuous", [])):
                feature_name = feature_names["continuous"][i]
                if feature_name in self.scalers:
                    scaler = self.scalers[feature_name]
                    if isinstance(scaler, TorchNormalizer):
                        features[..., idx] = scaler.transform(features[..., idx])
                    else:
                        feature_np = features[..., idx].reshape(-1, 1).numpy()
                        features[..., idx] = torch.tensor(
                            scaler.transform(feature_np),
                            dtype=torch.float32,
                        ).reshape(features[..., idx].shape)

        categorical_features = (
            features[:, self.categorical_indices]
            if self.categorical_indices
            else torch.zeros((features.shape[0], 0))
        )

        continuous_features = (
            features[:, self.continuous_indices]
            if self.continuous_indices
            else torch.zeros((features.shape[0], 0))
        )

        res = {
            "features": {
                "categorical": categorical_features,
                "continuous": continuous_features,
            },
            "target": target,
            "static": series["st"],
            "group": series.get("group", torch.tensor([0])),
            "length": len(series),
            "time_mask": mask_timestep,
            "cutoff_time": cutoff_time,
            "timestep": timestep,
        }

        if target_scale:
            res["target_scale"] = target_scale

        return res

    def _create_windows(self, indices: torch.Tensor) -> List[Tuple[int, int, int, int]]:
        """
        Create windows for the data in the given indices, for training, testing
        and validation.

        Parameters
        ----------
        indices : torch.Tensor
            The indices of the time series data to be processed.

        Returns
        -------
        List[Tuple[int, int, int, int]]
            A list of tuples where each tuple contains:
            - series_idx: Index of time series in the dataset
            - start_idx: Start index of the window
            - context_length: Length of the context/encoder window
            - prediction_length: Length of the prediction/decoder window
        """

        windows = []

        min_seq_length = self.context_length + self.prediction_length

        for idx in indices:
            series_idx = idx.item() if isinstance(idx, torch.Tensor) else idx
            sample = self.time_series_dataset[series_idx]
            sequence_length = len(sample["t"])

            if sequence_length < min_seq_length:
                continue

            cutoff_time = sample.get("cutoff_time", None)

            max_start = sequence_length - min_seq_length + 1

            stride = self.window_stride

            for start_idx in range(0, max_start, stride):
                window_end = start_idx + min_seq_length - 1  # 0-indexed

                if cutoff_time is not None:  # skip window if exceed cutoff time.
                    end_time = sample["t"][window_end].item()
                    if end_time > cutoff_time:
                        continue

                windows.append(
                    (
                        series_idx,
                        start_idx,
                        self.context_length,
                        self.prediction_length,
                    )
                )

        return windows
