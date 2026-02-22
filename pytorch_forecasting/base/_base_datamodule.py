"""Base class for PyTorch Forecasting data modules."""

from abc import abstractmethod

from lightning.pytorch import LightningDataModule

from pytorch_forecasting.base._base_object import _BaseObject

__all__ = ["_BaseDataModule"]


class _BaseDataModule(_BaseObject, LightningDataModule):
    """Abstract base class for all PyTorch Forecasting data modules.

    Inherits from both:
    - ``_BaseObject`` (for sktime ecosystem consistency and configuration management)
    - ``LightningDataModule`` (for PyTorch Lightning integration)

    This ensures all data modules:
    - Follow sktime configuration patterns
    - Support parameter validation and cloning via skbase
    - Integrate seamlessly with PyTorch Lightning trainers
    - Have consistent APIs across the library

    Attributes
    ----------
    batch_size : int
        Batch size for data loaders.
    num_workers : int
        Number of workers for data loading.
    """

    def __init__(self, batch_size: int = 32, num_workers: int = 0, **kwargs):
        """Initialize the data module.

        Parameters
        ----------
        batch_size : int, default=32
            Batch size for data loaders.
        num_workers : int, default=0
            Number of workers for data loading.
        **kwargs
            Additional keyword arguments passed to parent classes.
        """
        # Initialize skbase._BaseObject first
        _BaseObject.__init__(self, **kwargs)
        # Initialize LightningDataModule second
        LightningDataModule.__init__(self)

        self.batch_size = batch_size
        self.num_workers = num_workers

    @abstractmethod
    def prepare_data(self):
        """Download, tokenize, etc. - called only once."""
        pass

    @abstractmethod
    def setup(self, stage: str = None):
        """Create train/val/test datasets."""
        pass

    @abstractmethod
    def train_dataloader(self):
        """Return training DataLoader."""
        pass

    @abstractmethod
    def val_dataloader(self):
        """Return validation DataLoader."""
        pass

    @abstractmethod
    def test_dataloader(self):
        """Return test DataLoader."""
        pass

    @abstractmethod
    def predict_dataloader(self):
        """Return prediction DataLoader."""
        pass
