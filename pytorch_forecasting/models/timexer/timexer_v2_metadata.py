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
        return [
            dict(
                context_length=4,
                prediction_length=3,
            ),
        ]
