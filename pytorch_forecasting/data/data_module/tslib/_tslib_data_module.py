"""
Experimental data module for integrating `tslib` time series deep learning library.
"""

from collections.abc import Callable
from typing import Any
import warnings

from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
from torch.utils.data import Dataset

from pytorch_forecasting.data.data_module.base._base_data_module import (
    NORMALIZER,
    BaseTimeSeriesDataModule,
)
from pytorch_forecasting.data.data_module.tslib._tslib_dataset import _TslibDataset
from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)
from pytorch_forecasting.data.timeseries._timeseries_v2 import TimeSeries


class TslibDataModule(BaseTimeSeriesDataModule):
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
        add_relative_time_idx: bool = False,
        add_target_scales: bool = False,
        target_normalizer: NORMALIZER
        | str
        | list[NORMALIZER]
        | tuple[NORMALIZER]
        | None = "auto",  # noqa: E501
        scalers: dict[
            str, StandardScaler | RobustScaler | TorchNormalizer | EncoderNormalizer
        ]
        | None = None,  # noqa: E501
        shuffle: bool = True,
        window_stride: int = 1,
        batch_size: int = 32,
        num_workers: int = 0,
        train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
        collate_fn: Callable | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            time_series_dataset=time_series_dataset,
            target_normalizer=target_normalizer,
            batch_size=batch_size,
            num_workers=num_workers,
            train_val_test_split=train_val_test_split,
            add_relative_time_idx=add_relative_time_idx,
        )

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.freq = freq
        self.add_target_scales = add_target_scales
        self.scalers = scalers or {}
        self.shuffle = shuffle
        self.window_stride = window_stride
        self.kwargs = kwargs
        self.collate_fn = (
            collate_fn if collate_fn is not None else self.__class__.collate_fn
        )

        self._validate_indices()

    def _context_length(self) -> int:
        return self.context_length

    def _prediction_length(self) -> int:
        return self.prediction_length

    @property
    def train_shuffle(self) -> bool:
        """Return whether to shuffle at the training dataloader."""
        return self.shuffle

    def _build_dataset(self, windows: list[tuple[int, int, int, int]]) -> Dataset:
        """Return a ``_TslibDataset`` over *windows*.

        Parameters
        ----------
        windows : list of tuple[int, int, int, int]
        A list of tuples where each tuple contains the following:
            - series_idx: Index of time series in the dataset
            - start_idx: Start index of the window
            - context_length: Length of the context/encoder window
            - prediction_length: Length of the prediction/decoder window
            - add_relative_time_idx: bool Whether to add relative time index to dataset.

        """
        return _TslibDataset(
            dataset=self.time_series_dataset,
            data_module=self,
            windows=windows,
            add_relative_time_idx=self.add_relative_time_idx,
        )

    def _validate_indices(self):
        """
        Validate that we have meaningful features for training.
        Raises warnings for missing features or indices.
        """

        has_continuous = self.continuous_indices and len(self.continuous_indices) > 0
        has_categorical = self.categorical_indices and len(self.categorical_indices) > 0
        has_targets = len(self.time_series_metadata.get("cols", {}).get("y", [])) > 0
        if not has_targets:
            raise ValueError(
                "No target variables found in the dataset. "
                "Cannot proceed with model training."
            )

        if not has_continuous and not has_categorical and has_targets:
            warnings.warn(
                "No continuous or categorical features found. "
                "Proceeding with pure univariate forecasting "
                "using target history only.",
                UserWarning,
            )
            return

        if not has_continuous:
            warnings.warn(
                "No continuous features found in the dataset. "
                "Some models (TimeXer) requires continuous features. "
                "Consider adding continuous featuresinto the dataset.",
                UserWarning,
            )

        if not has_categorical:
            warnings.warn(
                "No categorical features found in the dataset. "
                "This may limit the model capabilities and and restrict "
                "the usage to continuous features only.",
                UserWarning,
            )

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

        if len(target_features) == 0:
            raise ValueError(
                "The time series dataset must have at least one target variable. "
                "Please provide a dataset with a target variable."
            )

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

        # detect the feature mode - S/MS/M

        n_targets = n_features["target"]
        n_cont = n_features["continuous"]
        n_cat = n_features["categorical"]

        if n_targets == 1 and (n_cont + n_cat) == 0:
            self.features = "S"
        elif n_targets == 1 and (n_cont + n_cat) >= 1:
            self.features = "MS"
        elif n_targets > 1 and (n_cont + n_cat) > 0:
            self.features = "M"
        else:
            self.features = "M"

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

            effective_min_prediction_idx = self.context_length

            max_prediction_idx = sequence_length - self.prediction_length + 1

            if max_prediction_idx <= effective_min_prediction_idx:
                continue

            stride = self.window_stride

            for start_idx in range(
                0, max_prediction_idx - effective_min_prediction_idx, stride
            ):  # noqa: E501
                if start_idx + self.context_length + self.prediction_length <= (
                    sequence_length
                ):
                    windows.append(
                        (
                            series_idx,
                            start_idx,
                            self.context_length,
                            self.prediction_length,
                        )
                    )

        return windows

    def _split_data_indices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split data indices into train, val, and test sets based on the
        train_val_test_split ratio.
        """
        total_series = len(self.time_series_dataset)
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
            self._train_size = int(self.train_val_test_split[0] * total_series)
            self._val_size = int(self.train_val_test_split[1] * total_series)

            self._train_indices = self._indices[: self._train_size]
            self._val_indices = self._indices[
                self._train_size : self._train_size + self._val_size
            ]

            self._test_indices = self._indices[
                self._train_size + self._val_size : total_series
            ]

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
        tuple[dict[str, torch.Tensor], torch.Tensor or list of torch.Tensor]
            A tuple containing the collated data and the target variable.
            If the dataset has multiple targets, a list of tensors each of shape
            (batch_size, prediction_length,). Otherwise, a single tensor of shape
            (batch_size, prediction_length).
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

        if isinstance(batch[0][1], list | tuple):
            num_targets = len(batch[0][1])
            y_batch = []
            for i in range(num_targets):
                target_tensors = [sample_y[i] for _, sample_y in batch]
                stacked_target = torch.stack(target_tensors)
                y_batch.append(stacked_target)
        else:
            y_batch = torch.stack([y for _, y in batch])
        return x_batch, y_batch
