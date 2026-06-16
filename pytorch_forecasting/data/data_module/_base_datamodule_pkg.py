"""Base package class for D2 datamodule discovery and testing."""

from pytorch_forecasting.base._base_object import _BaseObject
from pytorch_forecasting.data.timeseries import TimeSeries


class _BasePtDataModule(_BaseObject):
    """Base class for datamodule package objects discoverable by the test suite."""

    _tags = {"object_type": "datamodule_v2"}

    @classmethod
    def get_cls(cls):
        """Return the Lightning datamodule class."""
        raise NotImplementedError("Subclasses must implement `get_cls`.")

    @classmethod
    def name(cls):
        """Return the datamodule name from tags or class name."""
        name = cls.get_class_tags().get("info:name", None)
        if name is None:
            name = cls.get_cls().__name__
        return name

    @classmethod
    def get_test_timeseries(cls, **kwargs) -> TimeSeries:
        """Return a D1 TimeSeries configured for this datamodule format."""
        raise NotImplementedError("Subclasses must implement `get_test_timeseries`.")

    @classmethod
    def get_datamodule_test_params(cls):
        """Return parameter dicts for datamodule instantiation (excluding dataset)."""
        return [{}]

    @classmethod
    def get_expected_metadata_keys(cls):
        """Return metadata keys expected after setup for format-specific tests."""
        return []

    @classmethod
    def get_batch_keys(cls):
        """Return expected keys in the collated batch x-dict."""
        return []

    @classmethod
    def get_sample_item_keys(cls):
        """Return expected keys in a single dataset __getitem__ x-dict."""
        return cls.get_batch_keys()
