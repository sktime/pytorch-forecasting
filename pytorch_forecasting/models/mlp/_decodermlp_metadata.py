"""DecoderMLP metadata container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class DecoderMLPMetadata(_BasePtForecaster):
    """DecoderMLP metadata container."""

    _tags = {
        "info:name": "DecoderMLP",
        "info:compute": 1,
        "authors": ["jdb78"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": True,
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models import DecoderMLP

        return DecoderMLP

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
        from torchmetrics import MeanSquaredError

        from pytorch_forecasting.metrics import (
            MAE,
            CrossEntropy,
            MultiLoss,
            QuantileLoss,
        )

        return [
            {},
            dict(train_only=True),
            dict(
                loss=MultiLoss([QuantileLoss(), MAE()]),
                data_loader_kwargs=dict(
                    time_varying_unknown_reals=["volume", "discount"],
                    target=["volume", "discount"],
                ),
            ),
            dict(
                loss=CrossEntropy(),
                data_loader_kwargs=dict(
                    target="agency",
                ),
            ),
            dict(loss=MeanSquaredError()),
            dict(
                loss=MeanSquaredError(),
                data_loader_kwargs=dict(min_prediction_length=1, min_encoder_length=1),
            ),
        ]
