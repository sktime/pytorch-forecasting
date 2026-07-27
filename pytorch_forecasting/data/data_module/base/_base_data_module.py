"""Base D2 data module for v2 time series pipelines."""
#######################################################################################
# Disclaimer: This data-module is still work in progress and experimental, please
# use with care. This data-module is a basic skeleton of how the data-handling pipeline
# may look like in the future.
# This is D2 layer that will handle the preprocessing and data loaders.
# For now, this pipeline handles the simplest situation: The whole data can be loaded
# into the memory.
#######################################################################################

from abc import abstractmethod
from typing import Any
from warnings import warn

from lightning.pytorch import LightningDataModule
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)
from pytorch_forecasting.data.timeseries import TimeSeries

NORMALIZER = TorchNormalizer | EncoderNormalizer | NaNLabelEncoder

_EXPERIMENTAL_WARNING = (
    "{module_name} is part of an experimental rework of the "
    "pytorch-forecasting data layer, scheduled for release with v2.0.0. "
    "The API is not stable and may change without prior warning. "
    "For beta testing, but not for stable production use. "
    "Feedback and suggestions are very welcome in pytorch-forecasting issue 1736, "
    "https://github.com/sktime/pytorch-forecasting/issues/1736"
)


class BaseTimeSeriesDataModule(LightningDataModule):
    """
    Base Lightning datamodule for v2 time series pipelines (D2 layer).

    Parameters
    ----------
    time_series_dataset : TimeSeries
        D1 dataset passed to windowing and preprocessing. Must be non-empty.
    target_normalizer : normalizer, str, list, tuple, or None, default="auto"
        Target scaling. ``"auto"`` resolves to ``RobustScaler()``.
    batch_size : int, default=32
        Batch size for all dataloaders.
    num_workers : int, default=0
        Worker count for all dataloaders.
    train_val_test_split : tuple of float, default=(0.7, 0.15, 0.15)
        Stored for future use; not applied in the current v2 base implementation.
    add_relative_time_idx : bool, default=False
        Passed through to processed datasets when supported by the subclass.
    """

    def __init__(
        self,
        time_series_dataset: TimeSeries,
        target_normalizer: NORMALIZER
        | str
        | list[NORMALIZER]
        | tuple[NORMALIZER]
        | None = "auto",
        batch_size: int = 32,
        num_workers: int = 0,
        train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
        add_relative_time_idx: bool = False,
    ):
        super().__init__()
        warn(
            _EXPERIMENTAL_WARNING.format(module_name=type(self).__name__),
            UserWarning,
            stacklevel=3,
        )

        self.time_series_dataset = time_series_dataset
        self.target_normalizer = target_normalizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.add_relative_time_idx = add_relative_time_idx

        self._target_normalizer = self._coerce_target_normalizer(target_normalizer)
        self.time_series_metadata = time_series_dataset.get_metadata()
        self.n_targets = len(self.time_series_metadata["cols"]["y"])

        self.categorical_indices, self.continuous_indices = (
            self._extract_feature_type_indices(self.time_series_metadata)
        )

        self._metadata = None
        self.train_windows: list | None = None
        self.val_windows: list | None = None
        self.test_windows: list | None = None
        self.predict_windows: list | None = None
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.predict_dataset: Dataset | None = None

    def _coerce_target_normalizer(
        self,
        target_normalizer: NORMALIZER
        | str
        | list[NORMALIZER]
        | tuple[NORMALIZER]
        | None,
    ):
        # handle defaults and derived attributes
        if isinstance(target_normalizer, str) and target_normalizer.lower() == "auto":
            return RobustScaler()
        return target_normalizer

    def _extract_feature_type_indices(self, time_series_metadata: dict):
        """Extract feature type indices from the time series metadata."""
        categorical_indices = []
        continuous_indices = []
        for idx, col in enumerate(time_series_metadata["cols"]["x"]):
            if time_series_metadata["col_type"].get(col) == "C":
                categorical_indices.append(idx)
            else:
                continuous_indices.append(idx)
        return categorical_indices, continuous_indices

    @abstractmethod
    def _preprocess_data(self, series_idx) -> dict[str, Any]:
        """Preprocess one series into a cache dict.

        Composes coercion, feature splitting, and global normalization.

        Parameters
        ----------
        series_idx : int or torch.Tensor
            The index of the time series data to be processed.

        Returns
        -------
        dict of features of series item.
            Suggested keys: ``features`` (categorical/continuous), ``target``,
            ``static``, ``group``, ``length``, ``time_mask``, ``cutoff_time``,
            ``times``, ``timestep``.
        """  # noqa: E501

    @abstractmethod
    def _prepare_metadata(self) -> dict:
        """Prepare metadata for model initialisation."""

    @abstractmethod
    def _context_length(self) -> int:
        """Return encoder/context window length for this datamodule."""

    @abstractmethod
    def _prediction_length(self) -> int:
        """Return decoder/prediction window length for this datamodule."""

    @abstractmethod
    def _create_windows(self, indices: torch.Tensor) -> list[tuple[int, int, int, int]]:
        """Generate sliding windows for training, validation, and testing.

        Parameters
        ----------
        indices : torch.Tensor
            The indices of the time series data to be processed.

        Returns
        -------
        list of tuple[int, int, int, int]
            Each tuple is ``(series_idx, start_idx, context_length, prediction_length)``
            Series shorter than context + prediction are skipped.
        """

    @abstractmethod
    def _build_dataset(self, indices: torch.Tensor) -> Dataset:
        """Preprocess series at *indices*, create windows, and return a Dataset.

        Implementations typically call ``_preprocess_data``, ``_create_windows``,
        then wrap the result in a format-specific processed ``Dataset``. The
        returned dataset must expose a ``.windows`` attribute so base ``setup()``
        can cache window lists on the data module.

        Parameters
        ----------
        indices : torch.Tensor
            Series indices for this split (train, val, test, or predict).

        Returns
        -------
        Dataset
            A dataset that contains the processed data for the split.
        """

    @staticmethod
    @abstractmethod
    def collate_fn(batch):
        """Stack samples from dataset into a model-ready batch.

        Parameters
        ----------
        batch : list of tuple[dict, target]
            Samples as returned by the processed dataset.

        Returns
        -------
        tuple[dict, target]
            Collated ``x`` dict and ``y`` (tensor or list of tensors for multivariate).
        """

    @property
    def train_shuffle(self) -> bool:
        """Return whether to shuffle at the training dataloader."""
        return True

    def _get_collate_fn(self):
        collate = getattr(self, "collate_fn", None)
        if collate is not None:
            return collate
        return self.__class__.collate_fn

    @property
    def metadata(self) -> dict:
        """Compute metadata for model initialization.

        This property returns a dictionary containing the shapes and key information
        related to the time series model.
        """
        if self._metadata is None:
            self._metadata = self._prepare_metadata()
        return self._metadata

    @abstractmethod
    def _ensure_split(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split data indices into train, val, and test sets based on the
        train_val_test_split ratio once and cache them.
        """

    def setup(self, stage: str | None = None):
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
        if len(self.time_series_dataset) == 0:
            raise ValueError(
                f"Error in {type(self).__name__} setup stage '{stage}': "
                "The time series dataset is empty. "
                "Please provide a non-empty dataset."
            )

        # Series-level train/val/test splitting is not part of the v2 base API yet;
        # `setup()` currently windows over all series via
        # `torch.arange(len(time_series_dataset))`.

        self._ensure_split()

        if stage is None or stage == "fit":
            if self.train_dataset is None:
                self.train_dataset = self._build_dataset(self._train_indices)
                self.val_dataset = self._build_dataset(self._val_indices)
                self.train_windows = self.train_dataset.windows
                self.val_windows = self.val_dataset.windows
        elif stage == "test":
            if self.test_dataset is None:
                self.test_dataset = self._build_dataset(self._test_indices)
                self.test_windows = self.test_dataset.windows
        elif stage == "predict":
            predict_indices = torch.arange(len(self.time_series_dataset))
            self.predict_dataset = self._build_dataset(predict_indices)
            self.predict_windows = self.predict_dataset.windows

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.train_shuffle,
            collate_fn=self._get_collate_fn(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
        )
