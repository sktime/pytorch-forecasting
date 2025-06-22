"""
Packages container for DLinear model.
"""

from pytorch_forecasting.models.base._base_object import _BasePtForecasterV2


class DLinear_pkg_v2(_BasePtForecasterV2):
    """DLinear package container."""

    _tags = {
        "info:name": "DLinear",
        "info:compute": 2,
        "authors": ["PranavBhatP"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.dlinear._dlinear_v2 import DLinear

        return DLinear

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

        context_length = data_loader_kwargs.get("context_length", 8)
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
            Parameters to create testing instances of the class
        """

        from pytorch_forecasting.metrics import MAE, MAPE, SMAPE, QuantileLoss

        return [
            {},
            dict(moving_avg=25, individual=False, logging_metrics=[SMAPE()]),
            dict(
                moving_avg=4,
                individual=True,
            ),
            dict(
                moving_avg=5,
                individual=False,
                logging_metrics=[SMAPE()],
            ),
        ]
