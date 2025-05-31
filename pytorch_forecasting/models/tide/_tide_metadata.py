"""TiDE metadata container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class TiDEModelMetadata(_BasePtForecaster):
    """Metadata container for TiDE Model."""

    _tags = {
        "info:name": "TiDEModel",
        "info:compute": 3,
        "authors": ["Sohaib-Ahmed21"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.tide import TiDEModel

        return TiDEModel

    @classmethod
    def get_test_train_params(cls):
        """Return testing parameter settings for the trainer.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """

        from pytorch_forecasting.data.encoders import GroupNormalizer
        from pytorch_forecasting.metrics import SMAPE

        return [
            {
                "data_loader_kwargs": dict(
                    add_relative_time_idx=False,
                    # must include this everytime since the data_loader_default_kwargs
                    # include this to be True.
                )
            },
            {
                "temporal_decoder_hidden": 16,
                "data_loader_kwargs": dict(add_relative_time_idx=False),
            },
            {
                "dropout": 0.2,
                "use_layer_norm": True,
                "loss": SMAPE(),
                "data_loader_kwargs": dict(
                    target_normalizer=GroupNormalizer(
                        groups=["agency", "sku"], transformation="softplus"
                    ),
                    add_relative_time_idx=False,
                ),
            },
        ]
