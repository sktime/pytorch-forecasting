"""
Package container for LightTS model.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class LightTS_pkg_v2(Base_pkg):
    """
    Package container describing the LightTS model.

    This class registers metadata, links the model implementation,
    and provides helper utilities used during testing.
    """

    _tags = {
        "info:name": "LightTS",
        "info:compute": 2,
        "authors": ["Sylver-Icy"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """
        Return the LightTS model class.

        Returns
        -------
        type
            The LightTS model implementation.
        """
        from pytorch_forecasting.models.lightts._lightts_v2 import LightTS

        return LightTS

    @classmethod
    def get_datamodule_cls(cls):
        """
        Return the datamodule used for LightTS training and evaluation.

        Returns
        -------
        type
            The TslibDataModule class.
        """
        from pytorch_forecasting.data._tslib_data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def _get_test_datamodule_from(cls, trainer_kwargs):
        """
        Build test dataloaders from trainer configuration.

        Parameters
        ----------
        trainer_kwargs : dict
            Trainer configuration containing dataloader options.

        Returns
        -------
        dict
            Dictionary containing train, validation, and test dataloaders
            along with the initialized datamodule.
        """
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
        Provide parameter configurations used for automated model tests.

        Returns
        -------
        list of dict
            Different model parameter combinations used during testing.
        """
        from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss

        params = [
            {},
            dict(
                d_model=128,
                chunk_size=4,
                logging_metrics=[SMAPE()],
                loss=MAE(),
            ),
            dict(
                d_model=64,
                chunk_size=2,
                dropout=0.2,
            ),
            dict(
                d_model=96,
                chunk_size=8,
                loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            ),
        ]

        default_dm_cfg = {"context_length": 12, "prediction_length": 4}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = default_dm_cfg

        return params
