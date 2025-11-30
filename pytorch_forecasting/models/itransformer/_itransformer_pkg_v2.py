"""iTransformer package container v2."""

from pytorch_forecasting.models.base._base_object import _BasePtForecasterV2


class iTransformer_pkg_v2(_BasePtForecasterV2):
    """iTransformer metadata container."""

    _tags = {
        "info:name": "iTransformer",
        "authors": ["JATAYU000"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.itransformer._itransformer_v2 import (
            iTransformer,
        )

        return iTransformer

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
        """Get test train params."""
        # todo: expand test parameters
        return [
            {},
            dict(d_model=16, n_heads=2, e_layers=2, d_ff=64),
            dict(
                d_model=32,
                n_heads=4,
                e_layers=3,
                d_ff=128,
                dropout=0.1,
                data_loader_kwargs=dict(
                    batch_size=4, context_length=8, prediction_length=4
                ),
            ),
        ]
