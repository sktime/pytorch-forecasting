"""Extension template package container for a custom D2 datamodule."""

from pytorch_forecasting.data.data_module._base_datamodule_pkg import _BasePtDataModule


class MyDataModule_pkg(_BasePtDataModule):
    """Package container for MyDataModule — used by the unified test suite."""

    _tags = {
        "object_type": "datamodule_v2",
        "batch_format": "encoder_decoder",  # or "tslib", or a custom tag
        "info:name": "MyDataModule",
        "capability:static_features": False,
        "capability:multivariate_target": False,
    }

    @classmethod
    def get_cls(cls):
        from extension_templates.v2.data_module.data_module import MyDataModule

        return MyDataModule

    @classmethod
    def get_test_timeseries(cls, **kwargs):
        from pytorch_forecasting.tests._data_scenarios import (
            make_encoder_decoder_timeseries,
        )

        return make_encoder_decoder_timeseries(**kwargs)

    @classmethod
    def get_datamodule_test_params(cls):
        return [
            {
                "max_encoder_length": 8,
                "max_prediction_length": 4,
                "batch_size": 2,
            },
        ]

    @classmethod
    def get_expected_metadata_keys(cls):
        return ["target"]

    @classmethod
    def get_batch_keys(cls):
        return []
