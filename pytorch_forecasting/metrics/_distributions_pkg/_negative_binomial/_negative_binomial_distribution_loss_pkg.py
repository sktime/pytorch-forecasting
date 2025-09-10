"""
Package container for the Negative Binomial distribution loss metric.
"""

from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics.base_metrics._base_object import _BasePtMetric


class NegativeBinomialDistributionLoss_pkg(_BasePtMetric):
    """
    Negative binomial distribution loss metric for distribution forecasts.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "negative_binomial",
        "info:metric_name": "NegativeBinomialDistributionLoss",
        "requires:data_type": "negative_binomial_distribution_forecast",
        "clip_target": False,
        "data_loader_kwargs": {
            "target_normalizer": GroupNormalizer(groups=["agency", "sku"], center=False)
        },
        "info:pred_type": ["distr"],
        "info:y_type": ["numeric"],
        "expected_loss_ndim": 2,
    }

    @classmethod
    def get_cls(cls):
        from pytorch_forecasting.metrics.distributions import (
            NegativeBinomialDistributionLoss,
        )

        return NegativeBinomialDistributionLoss

    @classmethod
    def get_encoder(cls):
        """
        Returns a TorchNormalizer instance for rescaling parameters.
        """
        return TorchNormalizer(center=False)

    @classmethod
    def _get_test_dataloaders_from(cls, params=None):
        """
        Returns test dataloaders configured for NegativeBinomialDistributionLoss.
        """
        from pytorch_forecasting.tests._data_scenarios import (
            data_with_covariates,
            make_dataloaders,
        )

        if params is None:
            params = {}
        clip_target = cls._tags.get("clip_target", False)
        data_loader_kwargs = cls._tags.get("data_loader_kwargs", {}).copy()
        data_loader_kwargs.update(params.get("data_loader_kwargs", {}))

        data = data_with_covariates()
        if clip_target:
            data["target"] = data["target"].clip(1e-4, None)
        dataloaders = make_dataloaders(data, **data_loader_kwargs)
        return dataloaders
