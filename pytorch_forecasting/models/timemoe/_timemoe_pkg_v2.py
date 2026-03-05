"""
Metadata container for TimeMoE v2.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class TimeMoE_pkg_v2(Base_pkg):
    """TimeMoE metadata container."""

    _tags = {
        "info:name": "TimeMoE",
        "authors": ["lucifer4073"],
        "capability:exogenous": False,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "python_dependencies": ["transformers<=4.40.1"],
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.timemoe._timemoe_v2 import TimeMoE

        return TimeMoE

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
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``TimeMoE(**params)`` or ``TimeMoE(**params[i])`` creates a valid test
            instance. ``create_test_instance`` uses the first (or only) dictionary.

        Notes
        -----
        All test configurations share ``context_length=12`` and
        ``prediction_length=4`` as the minimum sensible defaults for the
        pretrained backbone.  ``training=True`` is kept in all cases to
        avoid downloading and fine-tuning large weights during CI runs.
        """
        params = [
            dict(
                task_name="zero_shot_forecast",
                training=True,
            ),
            dict(
                task_name="zero_shot_forecast",
                training=True,
                datamodule_cfg=dict(context_length=16, prediction_length=4),
            ),
        ]

        default_dm_cfg = {"context_length": 12, "prediction_length": 4}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            merged = {**default_dm_cfg, **current_dm_cfg}
            param["datamodule_cfg"] = merged

        return params
