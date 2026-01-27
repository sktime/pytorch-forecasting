"""
Metadata container for xLSTMTime v2.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class xLSTMTime_v2_pkg_v2(Base_pkg):
    """xLSTMTime metadata container."""

    _tags = {
        "info:name": "xLSTMTime",
        "authors": ["lucifer4073"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.xlstm._xlstm_v2 import xLSTMTime_v2

        return xLSTMTime_v2

    @classmethod
    def name(cls):
        """Return the name of the model."""
        return "xLSTMTime_v2"

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data._tslib_data_module import TslibDataModule

        return TslibDataModule

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
        from pytorch_forecasting.metrics import QuantileLoss

        params = [
            {},
            dict(
                hidden_size=64,
                xlstm_type="slstm",
            ),
            dict(datamodule_cfg=dict(context_length=12, prediction_length=3)),
            dict(
                xlstm_type="slstm",
                num_layers=1,
                decomposition_kernel=13,
            ),
            dict(
                hidden_size=256,
                xlstm_type="mlstm",
                num_layers=3,
                decomposition_kernel=25,
                dropout=0.2,
            ),
            dict(
                hidden_size=64,
                xlstm_type="mlstm",
                num_layers=2,
                dropout=0.1,
            ),
        ]

        default_dm_cfg = {"context_length": 12, "prediction_length": 4}
        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = default_dm_cfg

        return params
