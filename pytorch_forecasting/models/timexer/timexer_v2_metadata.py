"""TimeXer metadata container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class TimeXerMetadata(_BasePtForecaster):
    """TimeXer metadata container."""

    _tags = {
        "info:name": "TimeXer",
        "object_type": "ptf-v2",
        "authors": ["PranavBhatP"],
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models import TimeXer

        return TimeXer

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
        import torch.nn as nn

        return [
            dict(
                loss=nn.L1Loss(),
                context_length=30,
                prediction_length=1,
                d_model=32,
                n_heads=2,
                e_layers=1,
                d_ff=64,
                patch_length=1,
                task_name="long_term_forecast",
                features="MS",
            ),
        ]
