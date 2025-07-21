"""TFT package container."""

from pytorch_forecasting.models.base import _BasePtForecasterV2


class TFT_pkg_v2(_BasePtForecasterV2):
    """TFT package container."""

    _tags = {
        "info:name": "TFT",
        "authors": ["phoeenniixx"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 import TFT

        return TFT

    @classmethod
    def _get_test_datamodule_from(cls, trainer_kwargs):
        """Create test dataloaders from trainer_kwargs - following v1 pattern."""
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
        """Return testing parameter settings for the trainer.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return [
            {},
            dict(
                hidden_size=25,
                attention_head_size=5,
            ),
            dict(
                data_loader_kwargs=dict(max_encoder_length=5, max_prediction_length=3)
            ),
            dict(
                hidden_size=24,
                attention_head_size=8,
                data_loader_kwargs=dict(
                    max_encoder_length=5,
                    max_prediction_length=3,
                    add_relative_time_idx=False,
                ),
            ),
            dict(
                hidden_size=12,
                data_loader_kwargs=dict(max_encoder_length=7, max_prediction_length=10),
            ),
            dict(attention_head_size=2),
        ]
