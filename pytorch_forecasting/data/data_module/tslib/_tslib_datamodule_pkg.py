"""Package container for TslibDataModule."""

from pytorch_forecasting.data.data_module.base._base_datamodule_pkg import (
    _BasePtDataModule,
)


class TslibDataModule_pkg(_BasePtDataModule):
    """Package container for tslib D2 datamodule."""

    _tags = {
        "object_type": "datamodule_v2",
        "batch_format": "tslib",
        "info:name": "TslibDataModule",
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.data.data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def get_datamodule_test_params(cls):
        """Constructor kwargs for parametrized tests."""
        return [
            {
                "context_length": 8,
                "prediction_length": 4,
                "batch_size": 2,
                "num_workers": 0,
            },
            {
                "context_length": 6,
                "prediction_length": 3,
                "batch_size": 2,
                "window_stride": 2,
            },
        ]

    @classmethod
    def get_expected_metadata_keys(cls):
        """Keys that must exist in ``datamodule.metadata`` after ``setup()``."""
        return [
            "feature_names",
            "feature_indices",
            "n_features",
            "context_length",
            "prediction_length",
            "freq",
            "features",
        ]

    @classmethod
    def get_batch_keys(cls):
        """Required keys in the collated batch."""
        return [
            "history_cont",
            "history_cat",
            "future_cont",
            "future_cat",
            "history_length",
            "future_length",
            "history_mask",
            "future_mask",
            "groups",
            "history_time_idx",
            "future_time_idx",
            "history_target",
            "future_target",
            "future_target_len",
        ]

    @classmethod
    def get_sample_item_keys(cls):
        """Required keys in a single dataset item."""
        return [
            "history_cont",
            "history_cat",
            "future_cont",
            "future_cat",
            "history_length",
            "future_length",
            "history_mask",
            "future_mask",
            "groups",
            "history_time_idx",
            "future_time_idx",
            "future_target",
            "future_target_len",
        ]
