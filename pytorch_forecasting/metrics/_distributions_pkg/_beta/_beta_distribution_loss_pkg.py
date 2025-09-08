"""
Package container for the Beta distribution loss metric.
"""

from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric
from pytorch_forecasting.data.encoders import GroupNormalizer

class BetaDistributionLoss_pkg(_BasePtMetric):
    """
    Beta distribution loss metric for distribution forecasts.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "beta",
        "info:metric_name": "BetaDistributionLoss",
        "requires:data_type": "beta_distribution_forecast",
        "clip_target": True,
        "data_loader_kwargs": {
            "target_normalizer": GroupNormalizer(
                groups=["agency", "sku"], transformation="logit"
            )
        },
        "compatible_pred_types": ["distr"],
        "compatible_y_types": ["numeric"],
        "expected_loss_ndim": 2,
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.distributions import BetaDistributionLoss

        return BetaDistributionLoss

    @classmethod
    def get_encoder(cls):
        """
        Returns a TorchNormalizer instance for rescaling parameters.
        """
        return TorchNormalizer(transformation="logit")
    
    @classmethod
    def _get_test_dataloaders_from(cls, params=None):
        """
        Return test dataloaders configured for BetaDistributionLoss.
        """
        from pytorch_forecasting.tests._data_scenarios import data_with_covariates, make_dataloaders

        if params is None:
            params = {}
        clip_target = cls._tags.get("clip_target", False)
        data_loader_kwargs = cls._tags.get("data_loader_kwargs", {}).copy()
        data_loader_kwargs.update(params.get("data_loader_kwargs", {}))

        data = data_with_covariates()
        if clip_target:
            data["target"] = data["target"].clip(1e-4, 1 - 1e-4)
        dataloaders = make_dataloaders(data, **data_loader_kwargs)
        return dataloaders
