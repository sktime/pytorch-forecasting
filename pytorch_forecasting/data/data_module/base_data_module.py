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
import inspect
from typing import Any
from warnings import warn

from lightning.pytorch import LightningDataModule
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_forecasting.data._metadata import TimeSeriesMetadata
from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)
from pytorch_forecasting.data.timeseries import TimeSeries
from pytorch_forecasting.utils._validation import _check_fractions, _check_type

NORMALIZER = TorchNormalizer | EncoderNormalizer | NaNLabelEncoder

_WRAP_HINT = "Wrap the data frame first, e.g. TimeSeries(df, time=..., target=...)."

_EXPERIMENTAL_WARNING = (
    "{module_name} is part of an experimental rework of the "
    "pytorch-forecasting data layer, scheduled for release with v2.0.0. "
    "The API is not stable and may change without prior warning. "
    "For beta testing, but not for stable production use. "
    "Feedback and suggestions are very welcome in pytorch-forecasting issue 1736, "
    "https://github.com/sktime/pytorch-forecasting/issues/1736"
)


class BaseTimeSeriesDataModule(LightningDataModule):
    """Base Lightning datamodule for v2 time series pipelines (D2 layer).

    Parameters
    ----------
    time_series : TimeSeries, optional, default=None
        The input data. If ``None``, the module is a configuration only: it carries
        its parameters, and data is attached later with :meth:`with_data`.
    target_normalizer : normalizer, str, list, tuple, or None, default="auto"
        Target scaling. ``"auto"`` resolves to ``RobustScaler()``; subclasses may
        override :meth:`_coerce_target_normalizer` to change this.
    batch_size : int, default=32
        Batch size for all dataloaders.
    num_workers : int, default=0
        Worker count for all dataloaders.
    train_val_test_split : tuple of float, default=(0.7, 0.15, 0.15)
        Fractions of series that go to the train, validation and test split.
    add_relative_time_idx : bool, default=False
        Whether to add a relative time index feature.
    """

    def __init__(
        self,
        time_series: TimeSeries | None = None,
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

        self.time_series = time_series
        self.target_normalizer = target_normalizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.add_relative_time_idx = add_relative_time_idx

        self._validate_init_params()

        self._init_kwargs = {
            name: getattr(self, name)
            for name in inspect.signature(type(self).__init__).parameters
            if name not in ("self", "time_series", "kwargs")
        }

        warn(
            _EXPERIMENTAL_WARNING.format(module_name=type(self).__name__),
            UserWarning,
            stacklevel=3,
        )

        self._target_normalizer = self._coerce_target_normalizer(target_normalizer)
        self._metadata = None

        self.train_windows = None
        self.val_windows = None
        self.test_windows = None
        self.predict_windows = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        # without data there is no schema, so column positions and target count
        # cannot be derived yet
        self.time_series_metadata = None
        self.n_targets = None
        self.categorical_indices = []
        self.continuous_indices = []
        if time_series is not None:
            self._bind_time_series()

    def _validate_init_params(self):
        """Check the constructor arguments.

        Raises
        ------
        TypeError
            If ``time_series`` is not a :class:`TimeSeries`.
        ValueError
            If ``train_val_test_split`` is not three non-negative fractions
            summing to at most 1.
        """
        _check_type(
            self.time_series,
            TimeSeries,
            "time_series",
            allow_none=True,
            hint=_WRAP_HINT,
        )
        _check_fractions(self.train_val_test_split, "train_val_test_split")

    def _coerce_target_normalizer(
        self,
        target_normalizer: NORMALIZER
        | str
        | list[NORMALIZER]
        | tuple[NORMALIZER]
        | None,
    ):
        """Resolve the ``target_normalizer`` argument to the object actually used."""
        if isinstance(target_normalizer, str) and target_normalizer.lower() == "auto":
            return RobustScaler()
        return target_normalizer

    def _bind_time_series(self):
        """Derive the schema-dependent state from the attached data.

        Subclasses extend via ``super()`` for checks that need the schema.
        """
        self.time_series_metadata = self.time_series.get_metadata()
        self.n_targets = len(self.time_series_metadata["cols"]["y"])
        self.categorical_indices, self.continuous_indices = (
            self._extract_feature_type_indices(self.time_series_metadata)
        )

    @staticmethod
    def _extract_feature_type_indices(time_series_metadata: TimeSeriesMetadata):
        """Positions of categorical and continuous columns within ``cols["x"]``."""
        categorical_indices = []
        continuous_indices = []
        for idx, col in enumerate(time_series_metadata["cols"]["x"]):
            if time_series_metadata["col_type"].get(col) == "C":
                categorical_indices.append(idx)
            else:
                continuous_indices.append(idx)
        return categorical_indices, continuous_indices

    def _check_has_data(self, action: str):
        """Raise if an operation needs data and none is attached.

        Parameters
        ----------
        action : str
            What was attempted, named in the error message.

        Raises
        ------
        RuntimeError
            If the module was constructed without data.
        """
        if self.time_series is None:
            raise RuntimeError(
                f"{type(self).__name__} was constructed without data, so "
                f"{action} is not available. Attach data with "
                "`.with_data(time_series)`, which returns a new module, or "
                "pass `time_series` to the constructor."
            )

    def with_data(self, data: TimeSeries) -> "BaseTimeSeriesDataModule":
        """Return a copy of this module holding ``data``.

        Parameters
        ----------
        data : TimeSeries
            The data to attach.

        Returns
        -------
        BaseTimeSeriesDataModule
            A new module of the same class, configured identically.

        Raises
        ------
        TypeError
            If ``data`` is not a :class:`TimeSeries`.
        """
        _check_type(data, TimeSeries, "data", hint=_WRAP_HINT)
        return type(self)(time_series=data, **self._init_kwargs)

    @property
    def metadata(self):
        """Shapes and key information the model needs, computed once.

        See the subclass's ``_prepare_metadata`` for the keys.
        """
        self._check_has_data("`metadata`")
        if self._metadata is None:
            self._metadata = self._prepare_metadata()
        return self._metadata

    @abstractmethod
    def _prepare_metadata(self):
        """Prepare metadata for model initialisation."""

    @abstractmethod
    def _preprocess_data(self, series_idx) -> dict[str, Any]:
        """Preprocess one series into a cache dict.

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
        """

    @abstractmethod
    def _build_dataset(self, indices: torch.Tensor) -> Dataset:
        """Wrap the windows over indices in a format-specific ``Dataset``.

        Parameters
        ----------
        indices : torch.Tensor
            Series indices for this split (train, val, test, or predict).
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

    def _ensure_split(self):
        """Compute train/val/test series indices once and cache them."""
        if hasattr(self, "_split_indices"):
            return

        total_series = len(self.time_series)
        self._split_indices = torch.randperm(total_series)

        if total_series == 1:
            self._train_indices = self._split_indices
            self._val_indices = self._split_indices
            self._test_indices = self._split_indices
        elif total_series == 2:
            self._train_indices = self._split_indices[:1]
            self._val_indices = self._split_indices[1:]
            self._test_indices = self._split_indices[1:]
        else:
            train_size = int(self.train_val_test_split[0] * total_series)
            val_size = int(self.train_val_test_split[1] * total_series)
            self._train_indices = self._split_indices[:train_size]
            self._val_indices = self._split_indices[train_size : train_size + val_size]
            self._test_indices = self._split_indices[train_size + val_size :]

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
        self._check_has_data("`setup`")
        if len(self.time_series) == 0:
            raise ValueError(
                "The time series dataset is empty. "
                "Please provide a non-empty dataset."
            )

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
            predict_indices = torch.arange(len(self.time_series))
            self.predict_dataset = self._build_dataset(predict_indices)
            self.predict_windows = self.predict_dataset.windows

    @property
    def train_shuffle(self) -> bool:
        """Whether the training dataloader shuffles."""
        return True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.train_shuffle,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
