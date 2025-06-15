"""
Metadata container for TimeXer v2.
"""

from pytorch_forecasting.models.base._base_object import _BasePtForecasterV2


class TimeXerMetadata(_BasePtForecasterV2):
    """TimeXer metadata container."""

    _tags = {
        "info:name": "TimeXer",
        "authors": ["PranavBhatP"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.timexer._timexer_v2 import TimeXer

        return TimeXer

    @classmethod
    def _get_test_datamodule_from(cls, trainer_kwargs):
        """Create test dataloaders from trainer_kwargs - following v1 pattern."""
        from pytorch_forecasting.data._tslib_data_module import TslibDataModule
        from pytorch_forecasting.tests._conftest import make_datasets_v2
        from pytorch_forecasting.tests._data_scenarios import data_with_covariates_v2

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

        context_length = data_loader_kwargs.get("context_length", 12)
        prediction_length = data_loader_kwargs.get("prediction_length", 4)
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
                hidden_size=64,
                n_heads=4,
            ),
            dict(data_loader_kwargs=dict(context_length=12, prediction_length=3)),
            dict(
                hidden_size=32,
                n_heads=2,
                data_loader_kwargs=dict(
                    context_length=12,
                    prediction_length=3,
                    add_relative_time_idx=False,
                ),
            ),
            dict(
                hidden_size=128,
                patch_length=12,
                data_loader_kwargs=dict(context_length=16, prediction_length=4),
            ),
            dict(
                n_heads=2,
                e_layers=1,
                patch_length=6,
            ),
            dict(
                hidden_size=256,
                n_heads=8,
                e_layers=3,
                d_ff=1024,
                patch_length=8,
                factor=3,
                activation="gelu",
                dropout=0.2,
            ),
            dict(
                hidden_size=32,
                n_heads=2,
                e_layers=1,
                d_ff=64,
                patch_length=4,
                factor=2,
                activation="relu",
                dropout=0.05,
                data_loader_kwargs=dict(
                    context_length=16,
                    prediction_length=4,
                ),
            ),
        ]
