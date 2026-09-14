"""Base package class for D2 datamodule discovery and testing."""

from pytorch_forecasting.base._base_object import _BaseObject


class _BasePtDataModule(_BaseObject):
    """Base class for datamodule package objects discoverable by the test suite."""

    _tags = {"object_type": "datamodule_v2"}

    @classmethod
    def get_cls(cls):
        """Return the Lightning datamodule class wrapped by this package."""
        raise NotImplementedError("Subclasses must implement `get_cls`.")

    @classmethod
    def name(cls):
        """Return the datamodule name from tags or class name."""
        name = cls.get_class_tags().get("info:name", None)
        if name is None:
            name = cls.get_cls().__name__
        return name

    @classmethod
    def get_datamodule_test_params(cls):
        """Return parameter dicts for parametrized tests."""
        return [{}]

    @classmethod
    def get_expected_metadata_keys(cls):
        """
        Return metadata keys that must be present after ``setup()``
        for format tests.
        """
        return []

    @classmethod
    def get_batch_keys(cls):
        """Return required keys in the collated batch."""
        return []

    @classmethod
    def get_sample_item_keys(cls):
        """Return required keys in a single dataset item."""
        return cls.get_batch_keys()
