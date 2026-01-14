

from typing import Any, Optional, Union
from warnings import warn

from lightning.pytorch import LightningDataModule
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)
from pytorch_forecasting.data.timeseries import TimeSeries
from pytorch_forecasting.utils._coerce import _coerce_to_dict

NORMALIZER = TorchNormalizer | EncoderNormalizer | NaNLabelEncoder


class EncoderDecoderTimeSeriesDataModule(LightningDataModule):
    """
    Lightning DataModule for processing time series data in an encoder-decoder format.

    This module handles preprocessing, splitting, and batching of time series data
    for use in deep learning models. It supports categorical and continuous features,
    various scalers, and automatic target normalization.

    Parameters
    ----------
    time_series_dataset : TimeSeries
        The dataset containing time series data.
    max_encoder_length : int, default=30
        Maximum length of the encoder input sequence.
    min_encoder_length : Optional[int], default=None
        Minimum length of the encoder input sequence.
        Defaults to `max_encoder_length` if not specified.
    max_prediction_length : int, default=1
        Maximum length of the decoder output sequence.
    min_prediction_length : Optional[int], default=None
        Minimum length of the decoder output sequence.
        Defaults to `max_prediction_length` if not specified.
    min_prediction_idx : Optional[int], default=None
        Minimum index from which predictions start.
    allow_missing_timesteps : bool, default=False
        Whether to allow missing timesteps in the dataset.
    add_relative_time_idx : bool, default=False
        Whether to add a relative time index feature.
    add_target_scales : bool, default=False
        Whether to add target scaling information.
    add_encoder_length : Union[bool, str], default="auto"
        Whether to include encoder length information.
    target_normalizer :
        Union[NORMALIZER, str, List[NORMALIZER], Tuple[NORMALIZER], None],
         default="auto"
        Normalizer for the target variable. If "auto", uses `RobustScaler`.

    categorical_encoders : Optional[Dict[str, NaNLabelEncoder]], default=None
        Dictionary of categorical encoders.

    scalers :
    Optional[Dict[str, Union[StandardScaler, RobustScaler,
                        TorchNormalizer, EncoderNormalizer]]], default=None
        Dictionary of feature scalers.

    randomize_length : Union[None, Tuple[float, float], bool], default=False
        Whether to randomize input sequence length.
    batch_size : int, default=32
        Batch size for DataLoader.
    num_workers : int, default=0
        Number of workers for DataLoader.
    train_val_test_split : tuple, default=(0.7, 0.15, 0.15)
        Proportions for train, validation, and test dataset splits.
    """

    def __init__(
        self,
        time_series_dataset: TimeSeries,
        max_encoder_length: int = 30,
        min_encoder_length: int | None = None,
        max_prediction_length: int = 1,
        min_prediction_length: int | None = None,
        min_prediction_idx: int | None = None,
        allow_missing_timesteps: bool = False,
        add_relative_time_idx: bool = False,
        add_target_scales: bool = False,
        add_encoder_length: bool | str = "auto",
        target_normalizer: NORMALIZER
        | str
        | list[NORMALIZER]
        | tuple[NORMALIZER]
        | None = "auto",
        categorical_encoders: dict[str, NaNLabelEncoder] | None = None,
        scalers: dict[
            str, StandardScaler | RobustScaler | TorchNormalizer | EncoderNormalizer
        ]
        | None = None,
        randomize_length: None | tuple[float, float] | bool = False,
        batch_size: int = 32,
        num_workers: int = 0,
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
    ):
        self.time_series_dataset = time_series_dataset
        self.max_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        self.max_prediction_length = max_prediction_length
        self.min_prediction_length = min_prediction_length
        self.min_prediction_idx = min_prediction_idx
        self.allow_missing_timesteps = allow_missing_timesteps
        self.add_relative_time_idx = add_relative_time_idx
        self.add_target_scales = add_target_scales
        self.add_encoder_length = add_encoder_length
        self.randomize_length = randomize_length
        self.target_normalizer = target_normalizer
        self.categorical_encoders = categorical_encoders
        self.scalers = scalers
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split

        warn(
            "EncoderDecoderTimeSeriesDataModule is part of an experimental "
            "rework of the "
            "pytorch-forecasting data layer, "
            "scheduled for release with v2.0.0. "
            "The API is not stable and may change without prior warning. "
            "For beta testing, but not for stable production use. "
            "Feedback and suggestions are very welcome in "
            "pytorch-forecasting issue 1736, "
            "https://github.com/sktime/pytorch-forecasting/issues/1736",
            UserWarning,
        )

        super().__init__()

        # handle defaults and derived attributes
        if isinstance(target_normalizer, str) and target_normalizer.lower() == "auto":
            self._target_normalizer = RobustScaler()
        else:
            self._target_normalizer = target_normalizer

        self.time_series_metadata = time_series_dataset.get_metadata()
        self._min_prediction_length = min_prediction_length or max_prediction_length
        self._min_encoder_length = min_encoder_length or max_encoder_length
        self._categorical_encoders = _coerce_to_dict(categorical_encoders)
        self._scalers = _coerce_to_dict(scalers)
        self.n_targets = len(self.time_series_metadata["cols"]["y"])

        self.categorical_indices = []
        self.continuous_indices = []
        self._metadata = None

        for idx, col in enumerate(self.time_series_metadata["cols"]["x"]):
            if self.time_series_metadata["col_type"].get(col) == "C":
                self.categorical_indices.append(idx)
            else:
                self.continuous_indices.append(idx)

    def _prepare_metadata(self):
        encoder_cat_count = len(self.categorical_indices)
        encoder_cont_count = len(self.continuous_indices)

        decoder_cat_count = len(
            [
                col
                for col in self.time_series_metadata["cols"]["x"]
                if self.time_series_metadata["col_type"].get(col) == "C"
                and self.time_series_metadata["col_known"].get(col) == "K"
            ]
        )
        decoder_cont_count = len(
            [
                col
                for col in self.time_series_metadata["cols"]["x"]
                if self.time_series_metadata["col_type"].get(col) == "F"
                and self.time_series_metadata["col_known"].get(col) == "K"
            ]
        )

        target_count = len(self.time_series_metadata["cols"]["y"])
        metadata = {
            "encoder_cat": encoder_cat_count,
            "encoder_cont": encoder_cont_count,
            "decoder_cat": decoder_cat_count,
            "decoder_cont": decoder_cont_count,
            "target": target_count,
        }
        if self.time_series_metadata["cols"]["st"]:
            static_cat_count = len(
                [
                    col
                    for col in self.time_series_metadata["cols"]["st"]
                    if self.time_series_metadata["col_type"].get(col) == "C"
                ]
            )
            static_cont_count = (
                len(self.time_series_metadata["cols"]["st"]) - static_cat_count
            )

            metadata["static_categorical_features"] = static_cat_count
            metadata["static_continuous_features"] = static_cont_count
        else:
            metadata["static_categorical_features"] = 0
            metadata["static_continuous_features"] = 0

        metadata.update(
            {
                "max_encoder_length": self.max_encoder_length,
                "max_prediction_length": self.max_prediction_length,
                "min_encoder_length": self._min_encoder_length,
                "min_prediction_length": self._min_prediction_length,
            }
        )

        return metadata

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self._prepare_metadata()
        return self._metadata

    def _preprocess_data(self, series_idx: torch.Tensor) -> list[dict[str, Any]]:
        """Preprocess the data before feeding it into _ProcessedEncoderDecoderDataset.

        Preprocessing steps
        
        * Converts target (`y`) and features (`x`) to `torch.float32`.
        * Applies target normalization if `_target_normalizer` is set.
        * Applies scaling to continuous features if `_scalers` are provided.
        * Masks time points that are at or before the cutoff time.
        * Splits features into categorical and continuous subsets based on indices.
        """
        sample = self.time_series_dataset[series_idx]

        target = sample["y"]
        features = sample["x"]
        times = sample["t"]
        cutoff_time = sample["cutoff_time"]

        time_mask = torch.tensor(times <= cutoff_time, dtype=torch.bool)

        # convert to tensors
        target = target.float() if isinstance(target, torch.Tensor) else torch.tensor(target, dtype=torch.float32)
        features = features.float() if isinstance(features, torch.Tensor) else torch.tensor(features, dtype=torch.float32)

        #  TARGET NORMALIZATION 
        if self._target_normalizer is not None:
            target_np = target.unsqueeze(-1).numpy()
            if hasattr(self._target_normalizer, "fit") and not hasattr(self._target_normalizer, "scale_"):
                self._target_normalizer.fit(target_np)
                target = torch.tensor(self._target_normalizer.transform(target_np), dtype=torch.float32).squeeze(-1)
            elif hasattr(self._target_normalizer, "fit_") and not getattr(self._target_normalizer, "fitted_", False):
                self._target_normalizer.fit_(target)
                target = self._target_normalizer.transform(target)

        # CONTINUOUS FEATURE SCALING 
        continuous = features[:, self.continuous_indices] if self.continuous_indices else torch.zeros((features.shape[0], 0))
        for i, idx in enumerate(self.continuous_indices):
            feature_name = self.time_series_metadata["cols"]["x"][idx]
            scaler = self._scalers.get(feature_name, None)
            if scaler:
                col = continuous[:, i].unsqueeze(-1)
                if hasattr(scaler, "fit") and not hasattr(scaler, "scale_"):
                    scaler.fit(col.numpy())
                    col_transformed = torch.tensor(scaler.transform(col.numpy()), dtype=torch.float32)
                elif hasattr(scaler, "fit_") and not getattr(scaler, "fitted_", False):
                    scaler.fit_(col)
                    col_transformed = scaler.transform(col)
                else:
                    col_transformed = torch.tensor(scaler.transform(col.numpy()), dtype=torch.float32) if hasattr(scaler, "transform") else scaler.transform(col)
                continuous[:, i] = col_transformed.squeeze(-1)

        categorical = features[:, self.categorical_indices] if self.categorical_indices else torch.zeros((features.shape[0], 0))

        return {
            "features": {"categorical": categorical, "continuous": continuous},
            "target": target,
            "static": sample.get("st", None),
            "group": sample.get("group", torch.tensor([0])),
            "length": len(target),
            "time_mask": time_mask,
            "times": times,
            "cutoff_time": cutoff_time,
        }
