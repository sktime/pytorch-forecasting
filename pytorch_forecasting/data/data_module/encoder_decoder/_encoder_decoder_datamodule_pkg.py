"""Package container for EncoderDecoderTimeSeriesDataModule."""

from pytorch_forecasting.data.data_module.base._base_datamodule_pkg import (
    _BasePtDataModule,
)


class EncoderDecoderDataModule_pkg(_BasePtDataModule):
    """Package container for encoder-decoder D2 datamodule."""

    _tags = {
        "object_type": "datamodule_v2",
        "batch_format": "encoder_decoder",
        "info:name": "EncoderDecoderTimeSeriesDataModule",
        "capability:static_features": True,
        "capability:multivariate_target": True,
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    @classmethod
    def get_datamodule_test_params(cls):
        """Constructor kwargs for parametrized tests."""
        return [
            {
                "max_encoder_length": 24,
                "max_prediction_length": 12,
                "batch_size": 4,
                "train_val_test_split": (0.7, 0.15, 0.15),
            },
            {
                "max_encoder_length": 10,
                "max_prediction_length": 5,
                "batch_size": 2,
            },
        ]

    @classmethod
    def get_expected_metadata_keys(cls):
        """Keys that must exist in ``datamodule.metadata`` after ``setup()``."""
        return [
            "encoder_cat",
            "encoder_cont",
            "decoder_cat",
            "decoder_cont",
            "target",
            "max_encoder_length",
            "max_prediction_length",
        ]

    @classmethod
    def get_batch_keys(cls):
        """Keys present in collated batches."""
        return [
            "encoder_cat",
            "encoder_cont",
            "decoder_cat",
            "decoder_cont",
            "encoder_lengths",
            "decoder_lengths",
            "decoder_target_lengths",
            "groups",
            "encoder_time_idx",
            "decoder_time_idx",
            "target_past",
            "target_scale",
            "encoder_mask",
            "decoder_mask",
        ]
