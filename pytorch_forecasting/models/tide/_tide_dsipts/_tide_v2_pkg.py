"""TIDE package container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecasterV2


class TIDE_pkg_v2(_BasePtForecasterV2):
    """TIDE package container."""

    _tags = {
        "info:name": "TIDE",
        "authors": ["fbk_dsipts"],
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.tide._tide_dsipts._tide_v2 import TIDE

        return TIDE

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
        import torch.nn as nn

        return [
            dict(
                hidden_size=16,
                d_model=8,
                n_add_enc=1,
                n_add_dec=1,
                dropout_rate=0.1,
            ),
            dict(
                hidden_size=32,
                d_model=16,
                n_add_enc=2,
                n_add_dec=2,
                dropout_rate=0.2,
                data_loader_kwargs=dict(max_encoder_length=5, max_prediction_length=3),
                loss=nn.MSELoss(),
            ),
            dict(
                hidden_size=64,
                d_model=32,
                n_add_enc=3,
                n_add_dec=2,
                dropout_rate=0.1,
                data_loader_kwargs=dict(max_encoder_length=4, max_prediction_length=2),
                loss=nn.PoissonNLLLoss(),
            ),
        ]
