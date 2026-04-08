"""
Metadata container for SegRNN v2.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class SegRNN_pkg_v2(Base_pkg):
    """SegRNN metadata container."""

    _tags = {
        "info:name": "SegRNN",
        "authors": ["lucifer4073"],
        "capability:exogenous": False,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.segrnn._segrnn_v2 import SegRNN

        return SegRNN

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
        params : dict or list of dict
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance,
            i.e., ``SegRNN(**params)`` or ``SegRNN(**params[i])`` creates a valid
            test instance. ``create_test_instance`` uses the first (or only) dict.

        Notes
        -----
        ``seg_len`` must evenly divide both ``context_length`` and
        ``prediction_length``.  All test configs are carefully aligned to
        respect this constraint so that no segment-length warnings are raised
        during CI runs.
        """
        params = [
            dict(
                seg_len=4,
                d_model=32,
            ),
            dict(
                seg_len=4,
                d_model=64,
                datamodule_cfg=dict(context_length=12, prediction_length=8),
            ),
            dict(
                seg_len=3,
                d_model=16,
                dropout=0.05,
                datamodule_cfg=dict(context_length=12, prediction_length=6),
            ),
            dict(
                seg_len=2,
                d_model=256,
                dropout=0.1,
            ),
        ]

        default_dm_cfg = {"context_length": 12, "prediction_length": 4}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            param["datamodule_cfg"] = {**default_dm_cfg, **current_dm_cfg}

        return params
