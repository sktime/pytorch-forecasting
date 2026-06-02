"""
Packages container for DLinear model.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class DLinear_pkg_v2(Base_pkg):
    """DLinear package container."""

    _tags = {
        "info:name": "DLinear",
        "info:compute": 2,
        "authors": ["PranavBhatP"],
        "info:y_type": ["numeric"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.dlinear._dlinear_v2 import DLinear

        return DLinear

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def get_test_train_params(cls):
        """
        Return testing parameter settings for the trainer.

        Parameters
        ----------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
        """

        from pytorch_forecasting.metrics import SMAPE

        params = [
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
            dict(
                optimizer="adamw",
                lr_scheduler="cosine_annealing",
                lr_scheduler_params={"T_max": 5},
            ),
            dict(
                optimizer="adagrad",
                optimizer_params={"lr": 1e-3},
            ),
        ]

        default_dm_cfg = {"context_length": 8, "prediction_length": 2}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)

            param["datamodule_cfg"] = default_dm_cfg

        return params
