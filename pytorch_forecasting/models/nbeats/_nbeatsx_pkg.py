"""NBeatsX package container."""

from importlib import import_module

import torch

from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import PoissonLoss, TweedieLoss
from pytorch_forecasting.models.base._base_object import _BasePtForecaster
from pytorch_forecasting.tests._data_scenarios import (
    data_with_covariates,
    make_dataloaders,
)


class NBeatsX_pkg(_BasePtForecaster):
    """NBEATSx with exogenous variables."""

    _tags = {
        "info:name": "NBeatsX",
        "info:compute": 1,
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
        "authors": [],
        "capability:exogenous": True,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get the model class."""
        # Load after package initialization to avoid a model/package import cycle.
        return import_module("pytorch_forecasting.models.nbeats._nbeatsx").NBeatsX

    @classmethod
    def get_base_test_params(cls):
        """Return small configurations for model tests."""
        return [{"widths": [8], "num_blocks": [1], "num_block_layers": [1]}]

    @classmethod
    def _get_test_dataloaders_from(cls, params):
        """Return fixed-length dataloaders with covariates."""
        kwargs = {
            "time_varying_known_reals": ["time_idx", "price_regular"],
            "static_categoricals": ["agency", "sku"],
        }
        if isinstance(params.get("loss"), (PoissonLoss, TweedieLoss)):
            # These losses apply the inverse log link themselves.
            kwargs["target_normalizer"] = GroupNormalizer(
                groups=["agency", "sku"], transformation={"forward": torch.log1p}
            )
        kwargs.update(params.get("data_loader_kwargs", {}))
        return make_dataloaders(data_with_covariates(), **kwargs)
