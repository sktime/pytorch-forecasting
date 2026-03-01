"""
Package container for the PatchTST model.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class PatchTST_pkg_v2(Base_pkg):
    """PatchTST package container."""

    _tags = {
        "info:name": "PatchTST",
        "info:compute": 3,
        "authors": ["amruth6002"],
        "capability:exogenous": False,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.patchtst._patchtst_v2 import PatchTST

        return PatchTST

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data._tslib_data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def _get_test_datamodule_from(cls, trainer_kwargs):
        """Create test dataloaders from trainer_kwargs - following v1 pattern."""
        from pytorch_forecasting.data._tslib_data_module import TslibDataModule
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

        context_length = data_loader_kwargs.get("context_length", 16)
        prediction_length = data_loader_kwargs.get("prediction_length", 2)
        batch_size = data_loader_kwargs.get("batch_size", 2)

        train_datamodule = TslibDataModule(
            time_series_dataset=training_dataset,
            context_length=context_length,
            prediction_length=prediction_length,
            add_relative_time_idx=data_loader_kwargs.get("add_relative_time_idx", True),
            batch_size=batch_size,
            train_val_test_split=(0.8, 0.2, 0.0),
        )

        val_datamodule = TslibDataModule(
            time_series_dataset=validation_dataset,
            context_length=context_length,
            prediction_length=prediction_length,
            add_relative_time_idx=data_loader_kwargs.get("add_relative_time_idx", True),
            batch_size=batch_size,
            train_val_test_split=(0.0, 1.0, 0.0),
        )

        test_datamodule = TslibDataModule(
            time_series_dataset=validation_dataset,
            context_length=context_length,
            prediction_length=prediction_length,
            add_relative_time_idx=data_loader_kwargs.get("add_relative_time_idx", True),
            batch_size=batch_size,
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
        Return testing parameter settings for the trainer.

        Parameters
        ----------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss

        params = [
            {},  # defaults (d_model=128, n_heads=16, e_layers=3)
            dict(d_model=16, n_heads=4, e_layers=1, d_ff=32, patch_len=4, stride=2),
            dict(
                d_model=32,
                n_heads=4,
                e_layers=2,
                d_ff=64,
                patch_len=8,
                stride=4,
                logging_metrics=[SMAPE()],
            ),
            dict(
                d_model=16,
                n_heads=4,
                e_layers=1,
                d_ff=32,
                patch_len=4,
                stride=2,
                loss=QuantileLoss(),
            ),
        ]

        default_dm_cfg = {"context_length": 16, "prediction_length": 2}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = default_dm_cfg

        return params
