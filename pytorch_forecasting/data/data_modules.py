from typing import Any, Dict, List, Optional, Tuple, Union

from lightning.pytorch import LightningDataModule
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)
from pytorch_forecasting.data.timeseries import TimeSeries, _coerce_to_dict

NORMALIZER = Union[TorchNormalizer, NaNLabelEncoder, EncoderNormalizer]


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
        min_encoder_length: Optional[int] = None,
        max_prediction_length: int = 1,
        min_prediction_length: Optional[int] = None,
        min_prediction_idx: Optional[int] = None,
        allow_missing_timesteps: bool = False,
        add_relative_time_idx: bool = False,
        add_target_scales: bool = False,
        add_encoder_length: Union[bool, str] = "auto",
        target_normalizer: Union[
            NORMALIZER, str, List[NORMALIZER], Tuple[NORMALIZER], None
        ] = "auto",
        categorical_encoders: Optional[Dict[str, NaNLabelEncoder]] = None,
        scalers: Optional[
            Dict[
                str,
                Union[StandardScaler, RobustScaler, TorchNormalizer, EncoderNormalizer],
            ]
        ] = None,
        randomize_length: Union[None, Tuple[float, float], bool] = False,
        batch_size: int = 32,
        num_workers: int = 0,
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
    ):
        super().__init__()
        self.time_series_dataset = time_series_dataset
        self.metadata = time_series_dataset.get_metadata()

        self.max_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length or max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.min_prediction_length = min_prediction_length or max_prediction_length
        self.min_prediction_idx = min_prediction_idx

        self.allow_missing_timesteps = allow_missing_timesteps
        self.add_relative_time_idx = add_relative_time_idx
        self.add_target_scales = add_target_scales
        self.add_encoder_length = add_encoder_length
        self.randomize_length = randomize_length

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split

        if isinstance(target_normalizer, str) and target_normalizer.lower() == "auto":
            self.target_normalizer = RobustScaler()
        else:
            self.target_normalizer = target_normalizer

        self.categorical_encoders = _coerce_to_dict(categorical_encoders)
        self.scalers = _coerce_to_dict(scalers)

        self.categorical_indices = []
        self.continuous_indices = []

        for idx, col in enumerate(self.metadata["cols"]["x"]):
            if self.metadata["col_type"].get(col) == "C":
                self.categorical_indices.append(idx)
            else:
                self.continuous_indices.append(idx)

    def _preprocess_data(self, indices: torch.Tensor) -> List[Dict[str, Any]]:
        processed_data = []

        for idx in indices:
            sample = self.time_series_dataset[idx.item()]

            target = sample["y"]
            features = sample["x"]
            times = sample["t"]
            cutoff_time = sample["cutoff_time"]

            time_mask = torch.tensor(times <= cutoff_time, dtype=torch.bool)

            if isinstance(target, torch.Tensor):
                target = target.float()
            else:
                target = torch.tensor(target, dtype=torch.float32)

            if isinstance(features, torch.Tensor):
                features = features.float()
            else:
                features = torch.tensor(features, dtype=torch.float32)

            features_imputed = features.clone()
            for i in range(features.shape[1]):
                if torch.isnan(features[:, i]).any():
                    valid_values = features[time_mask, i]
                    valid_values = valid_values[~torch.isnan(valid_values)]
                    if len(valid_values) > 0:
                        mean_value = valid_values.mean()
                    else:
                        mean_value = 0.0
                    features_imputed[:, i] = torch.where(
                        torch.isnan(features[:, i]), mean_value, features[:, i]
                    )

            categorical = (
                features_imputed[:, self.categorical_indices]
                if self.categorical_indices
                else torch.zeros((features_imputed.shape[0], 0))
            )
            continuous = (
                features_imputed[:, self.continuous_indices]
                if self.continuous_indices
                else torch.zeros((features_imputed.shape[0], 0))
            )

            processed_data.append(
                {
                    "features": {"categorical": categorical, "continuous": continuous},
                    "target": target,
                    "static": sample.get("st", None),
                    "group": sample.get("group", torch.tensor([0])),
                    "length": len(target),
                    "time_mask": time_mask,
                    "times": times,
                    "cutoff_time": cutoff_time,
                }
            )

        return processed_data

    class _ProcessedEncoderDecoderDataset(Dataset):
        """PyTorch Dataset for processed encoder-decoder time series data.

        Parameters
        ----------
        processed_data : List[Dict[str, Any]]
            List of preprocessed time series samples.
        windows : List[Tuple[int, int, int, int]]
            List of window tuples containing
            (series_idx, start_idx, enc_length, pred_length).
        add_relative_time_idx : bool, default=False
            Whether to include relative time indices.
        """

        def __init__(
            self,
            processed_data: List[Dict[str, Any]],
            windows: List[Tuple[int, int, int, int]],
            add_relative_time_idx: bool = False,
        ):
            self.processed_data = processed_data
            self.windows = windows
            self.add_relative_time_idx = add_relative_time_idx

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, idx):
            series_idx, start_idx, enc_length, pred_length = self.windows[idx]
            data = self.processed_data[series_idx]

            end_idx = start_idx + enc_length + pred_length
            encoder_indices = slice(start_idx, start_idx + enc_length)
            decoder_indices = slice(start_idx + enc_length, end_idx)

            target_scale = data["target"][encoder_indices]
            target_scale = target_scale[~torch.isnan(target_scale)].abs().mean()
            if torch.isnan(target_scale) or target_scale == 0:
                target_scale = torch.tensor(1.0)

            encoder_mask = (
                data["time_mask"][encoder_indices]
                if "time_mask" in data
                else torch.ones(enc_length, dtype=torch.bool)
            )
            decoder_mask = (
                data["time_mask"][decoder_indices]
                if "time_mask" in data
                else torch.zeros(pred_length, dtype=torch.bool)
            )

            x = {
                "encoder_cat": data["features"]["categorical"][encoder_indices],
                "encoder_cont": data["features"]["continuous"][encoder_indices],
                "decoder_cat": data["features"]["categorical"][decoder_indices],
                "decoder_cont": data["features"]["continuous"][decoder_indices],
                "encoder_lengths": torch.tensor(enc_length),
                "decoder_lengths": torch.tensor(pred_length),
                "decoder_target_lengths": torch.tensor(pred_length),
                "groups": data["group"],
                "encoder_time_idx": torch.arange(enc_length),
                "decoder_time_idx": torch.arange(enc_length, enc_length + pred_length),
                "target_scale": target_scale,
                "encoder_mask": encoder_mask,
                "decoder_mask": decoder_mask,
            }

            if data["static"] is not None:
                x["static_categorical_features"] = data["static"].unsqueeze(0)
                x["static_continuous_features"] = torch.zeros((1, 0))

            y = data["target"][decoder_indices]
            if y.ndim == 1:
                y = y.unsqueeze(-1)

            return x, y

    def _create_windows(
        self, processed_data: List[Dict[str, Any]]
    ) -> List[Tuple[int, int, int, int]]:
        windows = []

        for idx, data in enumerate(processed_data):
            sequence_length = data["length"]

            if sequence_length < self.max_encoder_length + self.max_prediction_length:
                continue

            effective_min_prediction_idx = (
                self.min_prediction_idx
                if self.min_prediction_idx is not None
                else self.max_encoder_length
            )

            max_prediction_idx = sequence_length - self.max_prediction_length + 1

            if max_prediction_idx <= effective_min_prediction_idx:
                continue

            for start_idx in range(
                0, max_prediction_idx - effective_min_prediction_idx
            ):
                if (
                    start_idx + self.max_encoder_length + self.max_prediction_length
                    <= sequence_length
                ):
                    windows.append(
                        (
                            idx,
                            start_idx,
                            self.max_encoder_length,
                            self.max_prediction_length,
                        )
                    )

        return windows

    def setup(self, stage: Optional[str] = None):
        total_series = len(self.time_series_dataset)
        self._split_indices = torch.randperm(total_series)

        self._train_size = int(self.train_val_test_split[0] * total_series)
        self._val_size = int(self.train_val_test_split[1] * total_series)

        self._train_indices = self._split_indices[: self._train_size]
        self._val_indices = self._split_indices[
            self._train_size : self._train_size + self._val_size
        ]
        self._test_indices = self._split_indices[self._train_size + self._val_size :]

        if stage is None or stage == "fit":
            if not hasattr(self, "train_dataset") or not hasattr(self, "val_dataset"):
                self.train_processed = self._preprocess_data(self._train_indices)
                self.val_processed = self._preprocess_data(self._val_indices)

                self.train_windows = self._create_windows(self.train_processed)
                self.val_windows = self._create_windows(self.val_processed)

                self.train_dataset = self._ProcessedEncoderDecoderDataset(
                    self.train_processed, self.train_windows, self.add_relative_time_idx
                )
                self.val_dataset = self._ProcessedEncoderDecoderDataset(
                    self.val_processed, self.val_windows, self.add_relative_time_idx
                )

        elif stage is None or stage == "test":
            if not hasattr(self, "test_dataset"):
                self.test_processed = self._preprocess_data(self._test_indices)
                self.test_windows = self._create_windows(self.test_processed)

                self.test_dataset = self._ProcessedEncoderDecoderDataset(
                    self.test_processed, self.test_windows, self.add_relative_time_idx
                )
        elif stage == "predict":
            predict_indices = torch.arange(len(self.time_series_dataset))
            self.predict_processed = self._preprocess_data(predict_indices)
            self.predict_windows = self._create_windows(self.predict_processed)
            self.predict_dataset = self._ProcessedEncoderDecoderDataset(
                self.predict_processed, self.predict_windows, self.add_relative_time_idx
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        x_batch = {
            "encoder_cat": torch.stack([x["encoder_cat"] for x, _ in batch]),
            "encoder_cont": torch.stack([x["encoder_cont"] for x, _ in batch]),
            "decoder_cat": torch.stack([x["decoder_cat"] for x, _ in batch]),
            "decoder_cont": torch.stack([x["decoder_cont"] for x, _ in batch]),
            "encoder_lengths": torch.stack([x["encoder_lengths"] for x, _ in batch]),
            "decoder_lengths": torch.stack([x["decoder_lengths"] for x, _ in batch]),
            "decoder_target_lengths": torch.stack(
                [x["decoder_target_lengths"] for x, _ in batch]
            ),
            "groups": torch.stack([x["groups"] for x, _ in batch]),
            "encoder_time_idx": torch.stack([x["encoder_time_idx"] for x, _ in batch]),
            "decoder_time_idx": torch.stack([x["decoder_time_idx"] for x, _ in batch]),
            "target_scale": torch.stack([x["target_scale"] for x, _ in batch]),
            "encoder_mask": torch.stack([x["encoder_mask"] for x, _ in batch]),
            "decoder_mask": torch.stack([x["decoder_mask"] for x, _ in batch]),
        }

        if "static_categorical_features" in batch[0][0]:
            x_batch["static_categorical_features"] = torch.stack(
                [x["static_categorical_features"] for x, _ in batch]
            )
            x_batch["static_continuous_features"] = torch.stack(
                [x["static_continuous_features"] for x, _ in batch]
            )

        y_batch = torch.stack([y for _, y in batch])
        return x_batch, y_batch
