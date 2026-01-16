"""
Autoformer package container for v2 interface.
"""

from pytorch_forecasting.models.base._base_object import _BasePtForecasterV2


class Autoformer_pkg_v2(_BasePtForecasterV2):
    """
    Autoformer package container for v2 interface.
    
    Provides test configuration and metadata for the Autoformer model.
    """

    _tags = {
        "info:name": "Autoformer",
        "authors": ["Satarupa22-SD"],
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.autoformer._autoformer_v2 import Autoformer

        return Autoformer

    @classmethod
    def _get_test_datamodule_from(cls, trainer_kwargs):
        """
        Create test dataloaders from trainer_kwargs.
        
        Args:
            trainer_kwargs (dict): Training configuration parameters.
            
        Returns:
            dict: Dictionary containing train, val, test dataloaders and data module.
        """
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )
        from pytorch_forecasting.tests._data_scenarios import (
            data_with_covariates_v2,
            make_datasets_v2,
        )

        data_with_covariates = data_with_covariates_v2()

        data_loader_default_kwargs = dict(
            target="target",
            group_ids=["agency_encoded", "sku_encoded"],
            add_relative_time_idx=True,
        )

        data_loader_kwargs = trainer_kwargs.get("data_loader_kwargs", {})
        data_loader_default_kwargs.update(data_loader_kwargs)

        datasets_info = make_datasets_v2(
            data_with_covariates, **data_loader_default_kwargs
        )

        training_dataset = datasets_info["training_dataset"]
        validation_dataset = datasets_info["validation_dataset"]
        training_max_time_idx = datasets_info["training_max_time_idx"]

        max_encoder_length = data_loader_kwargs.get("max_encoder_length", 4)
        max_prediction_length = data_loader_kwargs.get("max_prediction_length", 3)
        add_relative_time_idx = data_loader_kwargs.get("add_relative_time_idx", True)
        batch_size = data_loader_kwargs.get("batch_size", 2)

        train_datamodule = EncoderDecoderTimeSeriesDataModule(
            time_series_dataset=training_dataset,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            add_relative_time_idx=add_relative_time_idx,
            batch_size=batch_size,
            train_val_test_split=(0.8, 0.2, 0.0),
        )

        val_datamodule = EncoderDecoderTimeSeriesDataModule(
            time_series_dataset=validation_dataset,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            min_prediction_idx=training_max_time_idx,
            add_relative_time_idx=add_relative_time_idx,
            batch_size=batch_size,
            train_val_test_split=(0.0, 1.0, 0.0),
        )

        test_datamodule = EncoderDecoderTimeSeriesDataModule(
            time_series_dataset=validation_dataset,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            min_prediction_idx=training_max_time_idx,
            add_relative_time_idx=add_relative_time_idx,
            batch_size=1,
            train_val_test_split=(0.0, 0.0, 1.0),
        )

        train_datamodule.setup("fit")
        val_datamodule.setup("fit")
        test_datamodule.setup("test")

        train_dataloader = train_datamodule.train_dataloader()
        val_dataloader = val_datamodule.val_dataloader()
        test_dataloader = test_datamodule.test_dataloader()

        return {
            "train": train_dataloader,
            "val": val_dataloader,
            "test": test_dataloader,
            "data_module": train_datamodule,
        }

    @classmethod
    def get_test_train_params(cls):
        """
        Return testing parameters settings for the trainer.
        
        Returns:
            list[dict]: List of parameter configurations for testing.
        """
        from pytorch_forecasting.metrics import QuantileLoss

        return [
            {
                "d_model": 32,
                "enc_layers": 1,
                "dec_layers": 1,
                "moving_avg": 25,
                "use_revin": False,
            },
            {
                "d_model": 16,
                "enc_layers": 1,
                "dec_layers": 1,
                "moving_avg": 25,
                "use_revin": True,
                "out_channels": 1,
            },
            {
                "loss": QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
                "d_model": 32,
                "enc_layers": 2,
                "dec_layers": 1,
                "moving_avg": 25,
                "use_revin": False,
            },
        ]