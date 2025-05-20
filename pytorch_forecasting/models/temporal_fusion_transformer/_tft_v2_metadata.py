"""TFT metadata container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class TFTMetadata(_BasePtForecaster):
    """TFT metadata container."""

    _tags = {
        "info:name": "TFT",
        "authors": ["jdb78"],
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
