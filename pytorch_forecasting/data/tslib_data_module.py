"""
Experimmental data module for integrating `tslib` time series deep learning library.
"""

from typing import Any, Optional, Union
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


class _TslibDataset(Dataset):
    """
    Dataset class for `tslib` time series dataset.

    Parameters
    ----------
    dataset : TimeSeries
        The time series dataset to be used for training and validation.
    data_module : TslibDataModule
        The data module that contains the metadata and other configurations for the
        dataset.
    windows: list[tuple[int, int, int, int]]
        A list of tuples where each tuple contains:
            - series_idx: Index of time series in the dataset
            - start_idx: Start index of the window
            - context_length: Length of the context/encoder window
            - prediction_length: Length of the prediction/decoder window
    add_relative_time_idx: bool
        Whether to add relative time index to the dataset.
    """

    def __init__(
        self,
        dataset: TimeSeries,
        data_module: "TslibDataModule",
        windows: list[tuple[int, int, int, int]],
        add_relative_time_idx: bool = False,
    ):
        self.dataset = dataset
        self.data_module = data_module
        self.windows = windows
        self.add_relative_time_idx = add_relative_time_idx

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get the processed dataset item at the given index.

        Parameters
        ----------
        idx : int
            The index of the dataset item to be retrieved.

        Returns
        -------
        x: dict[str, torch.Tensor]
            A dictionary containing the processed data.
        y: torch.Tensor
            The target variable.
        """

        series_idx, start_idx, context_length, prediction_length = self.windows[idx]

        processed_data = self.data_module._preprocess_data(series_idx)

        continous_features = processed_data["features"]["continuous"]
        categorical_features = processed_data["features"]["categorical"]

        end_idx = start_idx + context_length + prediction_length
        history_indices = slice(start_idx, start_idx + context_length)
        future_indices = slice(start_idx + context_length, end_idx)

        metadata = self.data_module.metadata

        history_cont = continous_features[history_indices]
        history_cat = categorical_features[history_indices]

        future_cont = continous_features[future_indices]
        future_cat = categorical_features[future_indices]

        known_features = set(metadata["feature_names"]["known"])
        continuous_feature_names = metadata["feature_names"]["continuous"]
        categorical_feature_names = metadata["feature_names"]["categorical"]

        # use masking to filter out known and unknow features.
        cont_known_mask = torch.tensor(
            [feat in known_features for feat in continuous_feature_names],
            dtype=torch.bool,
        )

        cat_known_mask = torch.tensor(
            [feat in known_features for feat in categorical_feature_names],
            dtype=torch.bool,
        )

        future_cont = (
            future_cont[:, cont_known_mask]
            if len(cont_known_mask) > 0
            else torch.zeros((future_cont.shape[0], 0))
        )  # noqa: E501
        future_cat = (
            future_cat[:, cat_known_mask]
            if len(cat_known_mask) > 0
            else torch.zeros((future_cat.shape[0], 0))
        )  # noqa: E501

        history_mask = (
            processed_data["time_mask"][history_indices]
            if "time_mask" in processed_data
            else torch.ones(context_length, dtype=torch.bool)
        )

        future_mask = (
            processed_data["time_mask"][future_indices]
            if "time_mask" in processed_data
            else torch.ones(prediction_length, dtype=torch.bool)
        )

        history_target = processed_data["target"][history_indices]
        future_target = processed_data["target"][future_indices]

        # history_time_idx = processed_data["timestep"][history_indices]
        # future_time_idx = processed_data["timestep"][future_indices]

        x = {
            "history_cont": history_cont,
            "history_cat": history_cat,
            "future_cont": future_cont,
            "future_cat": future_cat,
            "history_length": torch.tensor(context_length),
            "future_length": torch.tensor(prediction_length),
            "history_mask": history_mask,
            "future_mask": future_mask,
            "groups": processed_data["group"],
            "history_time_idx": torch.arange(context_length),
            "future_time_idx": torch.arange(
                context_length, context_length + prediction_length
            ),
            "history_target": history_target,
            "future_target": future_target,
            "future_target_len": torch.tensor(prediction_length),
        }

        if self.add_relative_time_idx:
            x["history_relative_time_idx"] = torch.arange(-context_length, 0)
            x["future_relative_time_idx"] = torch.arange(0, prediction_length)

        if processed_data["static"] is not None:
            x["static_categorical_features"] = processed_data["static"].unsqueeze(0)
            x["static_continuous_features"] = processed_data["static"].unsqueeze(0)

        if "target_scale" in processed_data:
            x["target_scale"] = processed_data["target_scale"]

        y = processed_data["target"][future_indices]

        return x, y


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
        Union[NORMALIZER, str, list[NORMALIZER], tuple[NORMALIZER], None],
         default="auto"
        Normalizer for the target variable. If "auto", uses `RobustScaler`.
    scalers : Optional[dict[str, Union[StandardScaler, RobustScaler, TorchNormalizer]]], default=None #noqa: E501
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
            NORMALIZER, str, list[NORMALIZER], tuple[NORMALIZER], None
        ] = "auto",  # noqa: E501
        scalers: Optional[
            dict[
                str,
                Union[StandardScaler, RobustScaler, TorchNormalizer, EncoderNormalizer],
            ]
        ] = None,  # noqa: E501
        shuffle: bool = True,
        window_stride: int = 1,
        batch_size: int = 32,
        num_workers: int = 0,
        train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
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
        self.collate_fn = (
            collate_fn if collate_fn is not None else self.__class__.collate_fn
        )  # noqa: E501
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

        self.scalers = scalers or {}
        self.shuffle = shuffle

        self.continuous_indices = []
        self.categorical_indices = []

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.window_stride = window_stride

        self.time_series_metadata = time_series_dataset.get_metadata()

        for idx, col in enumerate(self.time_series_metadata["cols"]["x"]):
            if self.time_series_metadata["col_type"].get(col) == "C":
                self.categorical_indices.append(idx)
            else:
                self.continuous_indices.append(idx)

    def _prepare_metadata(self) -> dict[str, Any]:
        """
        Prepare metadata for `tslib` time series data module.

        Returns
        -------
        dict containing the following as keys:
            - feature_names: dict[str, list[str]]
                Dictionary of feature names for each feature type.
            - feature_indices: dict[str, list[int]]
                Dictionary of feature indices for each feature type.
            - n_features: dict[str, int]
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
        ds_metadata = self.time_series_metadata

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

        feature_indices["target"] = list(range(len(target_features)))

        feature_names["static_categorical"] = static_cat_names
        feature_names["static_continuous"] = static_cont_names

        n_features = {k: len(v) for k, v in feature_names.items()}

        metadata = {
            "feature_names": feature_names,
            "feature_indices": feature_indices,
            "n_features": n_features,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "freq": self.freq,
            "features": self.features,
        }

        return metadata

    @property
    def metadata(self) -> dict[str, Any]:
        """ "
        Compute the metadata via the `_prepare_metadata` method.
        This method is called when the `metadata` property is accessed for the first.
        Returns
        -------
        dict
            Metadata for the data module. Refer to the `_prepare_metadata` method for
            the keys and values in the metadata dictionary.
        """
        if self._metadata is None:
            self._metadata = self._prepare_metadata()
        return self._metadata

    def _preprocess_data(self, idx: torch.Tensor) -> list[dict[str, Any]]:
        """
        Process the the time series data at the given index, before feeding it
        to the `_TslibDataset` class.

        Parameters
        ----------
        idx : torch.Tensor
            The index of the time series data to be processed.

        Returns
        -------
        dict[str, torch.Tensor]
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

        if isinstance(target, torch.Tensor):
            target = target.detach().clone().float()
        else:
            target = torch.tensor(target, dtype=torch.float32)

        if isinstance(features, torch.Tensor):
            features = features.detach().clone().float()
        else:
            features = torch.tensor(features, dtype=torch.float32)

        # scaling and normlization
        target_scale = {}

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

    def _create_windows(self, indices: torch.Tensor) -> list[tuple[int, int, int, int]]:
        """
        Create windows for the data in the given indices, for training, testing
        and validation.

        Parameters
        ----------
        indices : torch.Tensor
            The indices of the time series data to be processed.

        Returns
        -------
        list[tuple[int, int, int, int]]
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
                    end_time = sample["t"][window_end]

                    if isinstance(end_time, torch.Tensor):
                        end_time = end_time.item()

                    # Convert both to pandas Timestamp for consistent comparison
                    try:
                        if not isinstance(end_time, pd.Timestamp):
                            end_time = pd.Timestamp(end_time)
                        if not isinstance(cutoff_time, pd.Timestamp):
                            cutoff_time = pd.Timestamp(cutoff_time)

                        if end_time > cutoff_time:
                            continue
                    except (ValueError, TypeError) as e:
                        # If conversion fails, skip this window
                        warnings.warn(
                            f"Could not convert time values for comparison: {e}"
                        )
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

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup the data module by preparing the datasets for training,
        testing and validation.

        Parameters
        ----------
        stage: Optional[str]
            The stage of the data module. This can be "fit", "test" or "predict".
            If None, the data module will be setup for training.
        """

        # TODO: Add support for temporal/random/group splits.
        # Currently, it only supports random splits.
        # Handle the case where the dataset is empty.

        total_series = len(self.time_series_dataset)

        if total_series == 0:
            raise ValueError(
                "The time series dataset is empty. "
                "Please provide a non-empty dataset."
            )

        # this is a very rudimentary way to handle the splits when
        # the dataset is of size equal to 1 or 2.
        self._indices = torch.randperm(total_series)
        if total_series == 1:
            self._train_indices = self._indices
            self._val_indices = self._indices
            self._test_indices = self._indices
        elif total_series == 2:
            self._train_indices = self._indices[0:1]
            self._val_indices = self._indices[1:2]
            self._test_indices = self._indices[1:2]
        else:
            self._train_size = max(1, int(self.train_val_test_split[0] * total_series))
            self._val_size = max(1, int(self.train_val_test_split[1] * total_series))

            self._train_indices = self._indices[: self._train_size]
            self._val_indices = self._indices[
                self._train_size : self._train_size + self._val_size
            ]

            self._test_indices = self._indices[
                self._train_size + self._val_size : total_series
            ]

        assert (
            len(self._train_indices) > 0
        ), "Training dataset must contain at least one time series"
        assert (
            len(self._val_indices) > 0
        ), "Validation dataset must contain at least one time series"
        assert (
            len(self._test_indices) > 0
        ), "Test dataset must contain at least one time series"

        if stage == "fit" or stage is None:
            if not hasattr(self, "_train_dataset") or not hasattr(self, "_val_dataset"):
                self._train_windows = self._create_windows(self._train_indices)
                self._val_windows = self._create_windows(self._val_indices)

                self.train_dataset = _TslibDataset(
                    dataset=self.time_series_dataset,
                    data_module=self,
                    windows=self._train_windows,
                    add_relative_time_idx=self.add_relative_time_idx,
                )

                self.val_dataset = _TslibDataset(
                    dataset=self.time_series_dataset,
                    data_module=self,
                    windows=self._val_windows,
                    add_relative_time_idx=self.add_relative_time_idx,
                )
        elif stage == "test":
            if not hasattr(self, "_test_dataset"):
                self._test_windows = self._create_windows(self._test_indices)

                self.test_dataset = _TslibDataset(
                    dataset=self.time_series_dataset,
                    data_module=self,
                    windows=self._test_windows,
                    add_relative_time_idx=self.add_relative_time_idx,
                )

        elif stage == "predict":
            predict_indices = torch.arange(len(self.time_series_dataset))
            self._predict_windows = self._create_windows(predict_indices)

            self.predict_dataset = _TslibDataset(
                dataset=self.time_series_dataset,
                data_module=self,
                windows=self._predict_windows,
                add_relative_time_idx=self.add_relative_time_idx,
            )

    def train_dataloader(self) -> DataLoader:
        """
        Create the train dataloader.

        Returns
        -------
        DataLoader
            The train dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create the validation dataloader.
        Returns
        -------
        DataLoader
            The validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create the test dataloader.

        Returns
        -------
        DataLoader
            The test dataloader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Create the prediction dataloader.

        Returns
        -------
        DataLoader
            The prediction dataloader.
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for the dataloader.

        Parameters
        ----------
        batch: list[tuple[dict[str, Any]]]
            The batch of data to be collated.

        Returns
        -------
        tuple[dict[str, torch.Tensor], torch.Tensor]
            A tuple containing the collated data and the target variable.
        """

        x_batch = {
            "history_cont": torch.stack([x["history_cont"] for x, _ in batch]),
            "history_cat": torch.stack([x["history_cat"] for x, _ in batch]),
            "future_cont": torch.stack([x["future_cont"] for x, _ in batch]),
            "future_cat": torch.stack([x["future_cat"] for x, _ in batch]),
            "history_length": torch.stack([x["history_length"] for x, _ in batch]),
            "future_length": torch.stack([x["future_length"] for x, _ in batch]),
            "history_mask": torch.stack([x["history_mask"] for x, _ in batch]),
            "future_mask": torch.stack([x["future_mask"] for x, _ in batch]),
            "groups": torch.stack([x["groups"] for x, _ in batch]),
            "history_time_idx": torch.stack([x["history_time_idx"] for x, _ in batch]),
            "future_time_idx": torch.stack([x["future_time_idx"] for x, _ in batch]),
            "history_target": torch.stack([x["history_target"] for x, _ in batch]),
            "future_target": torch.stack([x["future_target"] for x, _ in batch]),
            "future_target_len": torch.stack(
                [x["future_target_len"] for x, _ in batch]
            ),
        }

        if "target_scale" in batch[0][0]:
            x_batch["target_scale"] = torch.stack([x["target_scale"] for x, _ in batch])

        if "history_relative_time_idx" in batch[0][0]:
            x_batch["history_relative_time_idx"] = torch.stack(
                [x["history_relative_time_idx"] for x, _ in batch]
            )
            x_batch["future_relative_time_idx"] = torch.stack(
                [x["future_relative_time_idx"] for x, _ in batch]
            )

        if "static_categorical_features" in batch[0][0]:
            x_batch["static_categorical_features"] = torch.stack(
                [x["static_categorical_features"] for x, _ in batch]
            )
            x_batch["static_continuous_features"] = torch.stack(
                [x["static_continuous_features"] for x, _ in batch]
            )

        y_batch = torch.stack([y for _, y in batch])
        return x_batch, y_batch
