#######################################################################################
# Disclaimer: This data-module is still work in progress and experimental, please
# use with care. This data-module is a basic skeleton of how the data-handling pipeline
# may look like in the future.
# This is D2 layer that will handle the preprocessing and data loaders.
# For now, this pipeline handles the simplest situation: The whole data can be loaded
# into the memory.
#######################################################################################

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
            NORMALIZER, str, list[NORMALIZER], tuple[NORMALIZER], None
        ] = "auto",
        categorical_encoders: Optional[dict[str, NaNLabelEncoder]] = None,
        scalers: Optional[
            dict[
                str,
                Union[StandardScaler, RobustScaler, TorchNormalizer, EncoderNormalizer],
            ]
        ] = None,
        randomize_length: Union[None, tuple[float, float], bool] = False,
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
            "TimeSeries is part of an experimental rework of the "
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

        self.categorical_indices = []
        self.continuous_indices = []
        self._metadata = None

        for idx, col in enumerate(self.time_series_metadata["cols"]["x"]):
            if self.time_series_metadata["col_type"].get(col) == "C":
                self.categorical_indices.append(idx)
            else:
                self.continuous_indices.append(idx)

    def _prepare_metadata(self):
        """Prepare metadata for model initialisation.

        Returns
        -------
        dict
            dictionary containing the following keys:

            * ``encoder_cat``: Number of categorical variables in the encoder.
                Computed as ``len(self.categorical_indices)``, which counts the
                categorical feature indices.
            * ``encoder_cont``: Number of continuous variables in the encoder.
                Computed as ``len(self.continuous_indices)``, which counts the
                continuous feature indices.
            * ``decoder_cat``: Number of categorical variables in the decoder that
                are known in advance.
                Computed by filtering ``self.time_series_metadata["cols"]["x"]``
                where col_type == "C"(categorical) and col_known == "K" (known)
            * ``decoder_cont``:  Number of continuous variables in the decoder that
                are known in advance.
                Computed by filtering ``self.time_series_metadata["cols"]["x"]``
                where col_type == "F"(continuous) and col_known == "K"(known)
            * ``target``: Number of target variables.
                Computed as ``len(self.time_series_metadata["cols"]["y"])``, which
                gives the number of output target columns..
            * ``static_categorical_features``: Number of static categorical features
                Computed by filtering ``self.time_series_metadata["cols"]["st"]``
                (static features) where col_type == "C" (categorical).
            * ``static_continuous_features``: Number of static continuous features
                Computed as difference of
                ``len(self.time_series_metadata["cols"]["st"])`` (static features)
                and static_categorical_features that gives static continuous feature
            * ``max_encoder_length``: maximum encoder length
                Taken directly from `self.max_encoder_length`.
            * ``max_prediction_length``: maximum prediction length
                Taken directly from `self.max_prediction_length`.
            * ``min_encoder_length``: minimum encoder length
                Taken directly from `self.min_encoder_length`.
            * ``min_prediction_length``: minimum prediction length
                Taken directly from `self.min_prediction_length`.
        """
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
        """Compute metadata for model initialization.

        This property returns a dictionary containing the shapes and key information
        related to the time series model. The metadata includes:

        * ``encoder_cat``: Number of categorical variables in the encoder.
        * ``encoder_cont``: Number of continuous variables in the encoder.
        * ``decoder_cat``: Number of categorical variables in the decoder that are
                            known in advance.
        * ``decoder_cont``:  Number of continuous variables in the decoder that are
                            known in advance.
        * ``target``: Number of target variables.

        If static features are present, the following keys are added:

        * ``static_categorical_features``: Number of static categorical features
        * ``static_continuous_features``: Number of static continuous features

        It also contains the following information:

        * ``max_encoder_length``: maximum encoder length
        * ``max_prediction_length``: maximum prediction length
        * ``min_encoder_length``: minimum encoder length
        * ``min_prediction_length``: minimum prediction length
        """
        if self._metadata is None:
            self._metadata = self._prepare_metadata()
        return self._metadata

    def _preprocess_data(self, series_idx: torch.Tensor) -> list[dict[str, Any]]:
        """Preprocess the data before feeding it into _ProcessedEncoderDecoderDataset.

        Preprocessing steps
        --------------------

        * Converts target (`y`) and features (`x`) to `torch.float32`.
        * Masks time points that are at or before the cutoff time.
        * Splits features into categorical and continuous subsets based on
            predefined indices.


        TODO: add scalers, target normalizers etc.
        """
        sample = self.time_series_dataset[series_idx]

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

        # TODO: add scalers, target normalizers etc.

        categorical = (
            features[:, self.categorical_indices]
            if self.categorical_indices
            else torch.zeros((features.shape[0], 0))
        )
        continuous = (
            features[:, self.continuous_indices]
            if self.continuous_indices
            else torch.zeros((features.shape[0], 0))
        )

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

    class _ProcessedEncoderDecoderDataset(Dataset):
        """PyTorch Dataset for processed encoder-decoder time series data.

        Parameters
        ----------
        dataset : TimeSeries
            The base time series dataset that provides access to raw data and metadata.
        data_module : EncoderDecoderTimeSeriesDataModule
            The data module handling preprocessing and metadata configuration.
        windows : List[Tuple[int, int, int, int]]
            List of window tuples containing
            (series_idx, start_idx, enc_length, pred_length).
        add_relative_time_idx : bool, default=False
            Whether to include relative time indices.
        """

        def __init__(
            self,
            dataset: TimeSeries,
            data_module: "EncoderDecoderTimeSeriesDataModule",
            windows: list[tuple[int, int, int, int]],
            add_relative_time_idx: bool = False,
        ):
            self.dataset = dataset
            self.data_module = data_module
            self.windows = windows
            self.add_relative_time_idx = add_relative_time_idx

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, idx):
            """Retrieve a processed time series window for dataloader input.

            x : dict
                Dictionary containing model inputs:

                * ``encoder_cat`` : tensor of shape (enc_length, n_cat_features)
                  Categorical features for the encoder.
                * ``encoder_cont`` : tensor of shape (enc_length, n_cont_features)
                  Continuous features for the encoder.
                * ``decoder_cat`` : tensor of shape (pred_length, n_cat_features)
                  Categorical features for the decoder.
                * ``decoder_cont`` : tensor of shape (pred_length, n_cont_features)
                  Continuous features for the decoder.
                * ``encoder_lengths`` : tensor of shape (1,)
                  Length of the encoder sequence.
                * ``decoder_lengths`` : tensor of shape (1,)
                  Length of the decoder sequence.
                * ``decoder_target_lengths`` : tensor of shape (1,)
                  Length of the decoder target sequence.
                * ``groups`` : tensor of shape (1,)
                  Group identifier for the time series instance.
                * ``encoder_time_idx`` : tensor of shape (enc_length,)
                  Time indices for the encoder sequence.
                * ``decoder_time_idx`` : tensor of shape (pred_length,)
                  Time indices for the decoder sequence.
                * ``target_scale`` : tensor of shape (1,)
                  Scaling factor for the target values.
                * ``encoder_mask`` : tensor of shape (enc_length,)
                  Boolean mask indicating valid encoder time points.
                * ``decoder_mask`` : tensor of shape (pred_length,)
                  Boolean mask indicating valid decoder time points.

                  If static features are present, the following keys are added:

                * ``static_categorical_features`` : tensor of shape
                                                    (1, n_static_cat_features), optional
                  Static categorical features, if available.
                * ``static_continuous_features`` : tensor of shape (1, 0), optional
                  Placeholder for static continuous features (currently empty).

            y : tensor of shape ``(pred_length, n_targets)``
                Target values for the decoder sequence.
            """
            series_idx, start_idx, enc_length, pred_length = self.windows[idx]
            data = self.data_module._preprocess_data(series_idx)

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

            encoder_cat = data["features"]["categorical"][encoder_indices]
            encoder_cont = data["features"]["continuous"][encoder_indices]

            features = data["features"]
            metadata = self.data_module.time_series_metadata

            known_cat_indices = [
                i
                for i, col in enumerate(metadata["cols"]["x"])
                if metadata["col_type"].get(col) == "C"
                and metadata["col_known"].get(col) == "K"
            ]

            known_cont_indices = [
                i
                for i, col in enumerate(metadata["cols"]["x"])
                if metadata["col_type"].get(col) == "F"
                and metadata["col_known"].get(col) == "K"
            ]

            cat_map = {
                orig_idx: i
                for i, orig_idx in enumerate(self.data_module.categorical_indices)
            }
            cont_map = {
                orig_idx: i
                for i, orig_idx in enumerate(self.data_module.continuous_indices)
            }

            mapped_known_cat_indices = [
                cat_map[idx] for idx in known_cat_indices if idx in cat_map
            ]
            mapped_known_cont_indices = [
                cont_map[idx] for idx in known_cont_indices if idx in cont_map
            ]

            decoder_cat = (
                features["categorical"][decoder_indices][:, mapped_known_cat_indices]
                if mapped_known_cat_indices
                else torch.zeros((pred_length, 0))
            )

            decoder_cont = (
                features["continuous"][decoder_indices][:, mapped_known_cont_indices]
                if mapped_known_cont_indices
                else torch.zeros((pred_length, 0))
            )

            x = {
                "encoder_cat": encoder_cat,
                "encoder_cont": encoder_cont,
                "decoder_cat": decoder_cat,
                "decoder_cont": decoder_cont,
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
                raw_st_tensor = data.get("static")
                static_col_names = self.data_module.time_series_metadata["cols"]["st"]

                is_categorical_mask = torch.tensor(
                    [
                        self.data_module.time_series_metadata["col_type"].get(col_name)
                        == "C"
                        for col_name in static_col_names
                    ],
                    dtype=torch.bool,
                )

                is_continuous_mask = ~is_categorical_mask

                st_cat_values_for_item = raw_st_tensor[is_categorical_mask]
                st_cont_values_for_item = raw_st_tensor[is_continuous_mask]

                if st_cat_values_for_item.shape[0] > 0:
                    x["static_categorical_features"] = st_cat_values_for_item.unsqueeze(
                        0
                    )
                else:
                    x["static_categorical_features"] = torch.zeros(
                        (1, 0), dtype=torch.float32
                    )

                if st_cont_values_for_item.shape[0] > 0:
                    x["static_continuous_features"] = st_cont_values_for_item.unsqueeze(
                        0
                    )
                else:
                    x["static_continuous_features"] = torch.zeros(
                        (1, 0), dtype=torch.float32
                    )

            y = data["target"][decoder_indices]
            if y.ndim == 1:
                y = y.unsqueeze(-1)

            return x, y

    def _create_windows(self, indices: torch.Tensor) -> list[tuple[int, int, int, int]]:
        """Generate sliding windows for training, validation, and testing.

        Returns
        -------
        List[Tuple[int, int, int, int]]
            A list of tuples, where each tuple consists of:
            - ``series_idx`` : int
              Index of the time series in `time_series_dataset`.
            - ``start_idx`` : int
              Start index of the encoder window.
            - ``enc_length`` : int
              Length of the encoder input sequence.
            - ``pred_length`` : int
              Length of the decoder output sequence.
        """
        windows = []

        for idx in indices:
            series_idx = idx.item()
            sample = self.time_series_dataset[series_idx]
            sequence_length = len(sample["y"])

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
                            series_idx,
                            start_idx,
                            self.max_encoder_length,
                            self.max_prediction_length,
                        )
                    )

        return windows

    def setup(self, stage: Optional[str] = None):
        """Prepare the datasets for training, validation, testing, or prediction.

        Parameters
        ----------
        stage : Optional[str], default=None
            Specifies the stage of setup. Can be one of:
            - ``"fit"`` : Prepares training and validation datasets.
            - ``"test"`` : Prepares the test dataset.
            - ``"predict"`` : Prepares the dataset for inference.
            - ``None`` : Prepares ``fit`` datasets.
        """
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
                self.train_windows = self._create_windows(self._train_indices)
                self.val_windows = self._create_windows(self._val_indices)

                self.train_dataset = self._ProcessedEncoderDecoderDataset(
                    self.time_series_dataset,
                    self,
                    self.train_windows,
                    self.add_relative_time_idx,
                )
                self.val_dataset = self._ProcessedEncoderDecoderDataset(
                    self.time_series_dataset,
                    self,
                    self.val_windows,
                    self.add_relative_time_idx,
                )

        elif stage == "test":
            if not hasattr(self, "test_dataset"):
                self.test_windows = self._create_windows(self._test_indices)
                self.test_dataset = self._ProcessedEncoderDecoderDataset(
                    self.time_series_dataset,
                    self,
                    self.test_windows,
                    self.add_relative_time_idx,
                )
        elif stage == "predict":
            predict_indices = torch.arange(len(self.time_series_dataset))
            self.predict_windows = self._create_windows(predict_indices)
            self.predict_dataset = self._ProcessedEncoderDecoderDataset(
                self.time_series_dataset,
                self,
                self.predict_windows,
                self.add_relative_time_idx,
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
