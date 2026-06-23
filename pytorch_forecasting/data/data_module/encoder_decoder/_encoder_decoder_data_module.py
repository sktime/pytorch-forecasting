#######################################################################################
# Disclaimer: This data-module is still work in progress and experimental, please
# use with care. This data-module is a basic skeleton of how the data-handling pipeline
# may look like in the future.
# This is D2 layer that will handle the preprocessing and data loaders.
# For now, this pipeline handles the simplest situation: The whole data can be loaded
# into the memory.
#######################################################################################

from typing import Any

from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
from torch.utils.data import Dataset

from pytorch_forecasting.data.data_module.base._base_data_module import (
    NORMALIZER,
    BaseTimeSeriesDataModule,
)
from pytorch_forecasting.data.data_module.encoder_decoder._encoder_decoder_dataset import (  # noqa: E501
    _ProcessedEncoderDecoderDataset,
)
from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)
from pytorch_forecasting.data.timeseries import TimeSeries
from pytorch_forecasting.utils._coerce import _coerce_to_dict


class EncoderDecoderTimeSeriesDataModule(BaseTimeSeriesDataModule):
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
        super().__init__(
            time_series_dataset=time_series_dataset,
            target_normalizer=target_normalizer,
            batch_size=batch_size,
            num_workers=num_workers,
            train_val_test_split=train_val_test_split,
            add_relative_time_idx=add_relative_time_idx,
        )

        self.max_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        self.max_prediction_length = max_prediction_length
        self.min_prediction_length = min_prediction_length
        self.min_prediction_idx = min_prediction_idx
        self.allow_missing_timesteps = allow_missing_timesteps
        self.add_target_scales = add_target_scales
        self.add_encoder_length = add_encoder_length
        self.randomize_length = randomize_length
        self.categorical_encoders = categorical_encoders
        self.scalers = scalers

        self._min_prediction_length = min_prediction_length or max_prediction_length
        self._min_encoder_length = min_encoder_length or max_encoder_length
        self._categorical_encoders = _coerce_to_dict(categorical_encoders)
        self._scalers = _coerce_to_dict(scalers)

    def _context_length(self) -> int:
        return self.max_encoder_length

    def _prediction_length(self) -> int:
        return self.max_prediction_length

    def _build_dataset(self, windows: list[tuple[int, int, int, int]]) -> Dataset:
        """Return a ``_ProcessedEncoderDecoderDataset`` over *windows*.

        Parameters
        ----------
        windows : list of tuple[int, int, int, int]
        List of window tuples having (series_idx, start_idx, enc_length, pred_length).
        """
        return _ProcessedEncoderDecoderDataset(
            self.time_series_dataset,
            self,
            windows,
            self.add_relative_time_idx,
        )

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

    def _split_data_indices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split data indices into train, val, and test sets based on the
        train_val_test_split ratio.
        """
        # this is a very rudimentary way to handle the splits when
        # the dataset is of size equal to 1 or 2.
        total_series = len(self.time_series_dataset)
        self._split_indices = torch.randperm(total_series)

        self._train_size = int(self.train_val_test_split[0] * total_series)
        self._val_size = int(self.train_val_test_split[1] * total_series)

        self._train_indices = self._split_indices[: self._train_size]
        self._val_indices = self._split_indices[
            self._train_size : self._train_size + self._val_size
        ]
        self._test_indices = self._split_indices[self._train_size + self._val_size :]

    @staticmethod
    def collate_fn(batch):
        """Stack encoder-decoder samples into batched ``x`` and ``y``.

        Parameters
        ----------
        batch : list of tuple[dict, target]
            Samples from ``_ProcessedEncoderDecoderDataset`` dataset.

        Returns
        -------
        tuple[dict, torch.Tensor or list[torch.Tensor]]
            Collated inputs and targets. Multivariate targets become a list of tensors.
        """
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
            "target_past": torch.stack([x["target_past"] for x, _ in batch]),
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
