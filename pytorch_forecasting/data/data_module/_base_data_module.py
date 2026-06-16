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
    """Abstract D2 base for v2 time series datamodules."""

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
            stacklevel=3
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

        self.categorical_indices: list[int] = []
        self.continuous_indices: list[int] = []
        self._init_feature_indices()

        self._metadata = None
        self.train_windows: list | None = None
        self.val_windows: list | None = None
        self.test_windows: list | None = None
        self.predict_windows: list | None = None
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.predict_dataset: Dataset | None = None

    @staticmethod
    def _coerce_target_normalizer(
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

    def _init_feature_indices(self):
        for idx, col in enumerate(self.time_series_metadata["cols"]["x"]):
            if self.time_series_metadata["col_type"].get(col) == "C":
                self.categorical_indices.append(idx)
            else:
                self.continuous_indices.append(idx)

    @abstractmethod
    def _prepare_metadata(self) -> dict:
        """Prepare metadata for model initialisation."""

    @abstractmethod
    def _context_length(self) -> int:
        """Return encoder/context window length."""

    @abstractmethod
    def _prediction_length(self) -> int:
        """Return decoder/prediction window length."""

    @abstractmethod
    def _create_windows(
        self, indices: torch.Tensor
    ) -> list[tuple[int, int, int, int]]:
        """Create sliding windows for the given series indices."""

    @abstractmethod
    def _build_dataset(self, windows: list[tuple[int, int, int, int]]) -> Dataset:
        """Build a processed dataset from window tuples."""

    @staticmethod
    @abstractmethod
    def collate_fn(batch):
        """Collate a batch of samples."""

    def _train_shuffle(self) -> bool:
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

    def _compute_split_indices(self):
        total_series = len(self.time_series_dataset)
        if total_series == 0:
            raise ValueError(
                "The time series dataset is empty. "
                "Please provide a non-empty dataset."
            )

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
        
        self._compute_split_indices()

        if stage is None or stage == "fit":
            if self.train_dataset is None:
                self.train_windows = self._create_windows(self._train_indices)
                self.val_windows = self._create_windows(self._val_indices)
                self.train_dataset = self._build_dataset(self.train_windows)
                self.val_dataset = self._build_dataset(self.val_windows)
        elif stage == "test":
            if self.test_dataset is None:
                self.test_windows = self._create_windows(self._test_indices)
                self.test_dataset = self._build_dataset(self.test_windows)
        elif stage == "predict":
            predict_indices = torch.arange(len(self.time_series_dataset))
            self.predict_windows = self._create_windows(predict_indices)
            self.predict_dataset = self._build_dataset(self.predict_windows)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self._train_shuffle(),
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
