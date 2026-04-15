"""
Metadata container for Moirai-MoE v2.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class MoiraiMoE_pkg_v2(Base_pkg):
    """Moirai-MoE metadata container."""

    _tags = {
        "info:name": "MoiraiMoE",
        "authors": ["priyanshuharshbodhi1"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "python_dependencies": ["uni2ts", "torch", "einops", "huggingface_hub"],
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.moirai_moe._moirai_moe_v2 import MoiraiMoE

        return MoiraiMoE

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data._tslib_data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def _get_test_datamodule_from(cls, trainer_kwargs):
        """Create test dataloaders from trainer_kwargs."""
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

        context_length = data_loader_kwargs.get("context_length", 32)
        prediction_length = data_loader_kwargs.get("prediction_length", 8)
        batch_size = data_loader_kwargs.get("batch_size", 2)
        add_rel = data_loader_kwargs.get("add_relative_time_idx", True)

        train_datamodule = TslibDataModule(
            time_series_dataset=training_dataset,
            context_length=context_length,
            prediction_length=prediction_length,
            add_relative_time_idx=add_rel,
            batch_size=batch_size,
            train_val_test_split=(0.8, 0.2, 0.0),
        )

        val_datamodule = TslibDataModule(
            time_series_dataset=validation_dataset,
            context_length=context_length,
            prediction_length=prediction_length,
            add_relative_time_idx=add_rel,
            batch_size=batch_size,
            train_val_test_split=(0.0, 1.0, 0.0),
        )

        test_datamodule = TslibDataModule(
            time_series_dataset=validation_dataset,
            context_length=context_length,
            prediction_length=prediction_length,
            add_relative_time_idx=add_rel,
            batch_size=batch_size,
            train_val_test_split=(0.0, 0.0, 1.0),
        )

        train_datamodule.setup("fit")
        val_datamodule.setup("fit")
        test_datamodule.setup("test")

        return {
            "train": train_datamodule.train_dataloader(),
            "val": val_datamodule.val_dataloader(),
            "test": test_datamodule.test_dataloader(),
            "data_module": train_datamodule,
        }

    @classmethod
    def get_test_train_params(cls):
        """Return testing parameter settings for the trainer.

        The default checkpoint ``Salesforce/moirai-moe-1.0-R-small`` is kept
        across configurations to avoid re-downloading larger backbones during
        CI runs. ``training=False`` freezes the backbone so gradients are not
        tracked through the pretrained weights in the smoke tests.
        """
        params = [
            dict(training=False, num_samples=20),
            dict(
                training=False,
                patch_size=16,
                num_samples=10,
                datamodule_cfg=dict(context_length=32, prediction_length=8),
            ),
        ]

        default_dm_cfg = {"context_length": 32, "prediction_length": 8}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            merged = {**default_dm_cfg, **current_dm_cfg}
            param["datamodule_cfg"] = merged

        return params
