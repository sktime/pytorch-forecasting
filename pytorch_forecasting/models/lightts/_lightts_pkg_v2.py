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
        "authors": ["wuhaixu2016", "Sylver-Icy"],
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
