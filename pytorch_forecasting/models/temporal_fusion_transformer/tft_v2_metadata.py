"""TFT metadata container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class TFTMetadata(_BasePtForecaster):
    """TFT metadata container."""

    _tags = {
        "info:name": "TFT",
        "object_type": "ptf-v2",
        "authors": ["phoeenniixx"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 import TFT

        return TFT

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
            {},
            dict(
                hidden_size=25,
                attention_head_size=5,
            ),
        ]


class TemporalFusionTransformerMetadata(_BasePtForecaster):
    """TFT metadata container."""

    _tags = {
        "info:name": "TemporalFusionTransformerM",
        "object_type": "ptf-v2",
        "authors": ["jdb78"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.temporal_fusion_transformer._tft_ver2 import (
            TemporalFusionTransformer,
        )

        return TemporalFusionTransformer

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
            {},
            dict(
                hidden_size=25,
                attention_head_size=5,
                data_loader_kwargs={
                    "add_relative_time_idx": False,
                },
            ),
        ]
