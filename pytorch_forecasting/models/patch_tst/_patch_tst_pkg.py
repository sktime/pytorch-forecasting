"""PatchTST package container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class PatchTST_pkg(_BasePtForecaster):
    """PatchTST package container."""

    _tags = {
        "info:name": "PatchTST",
        "info:compute": 3,
        "info:pred_type": ["point", "quantile"],
        "info:y_type": ["numeric"],
        "authors": ["nareshmethuku"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models import PatchTST

        return PatchTST

    @classmethod
    def get_base_test_params(cls):
        """
        Return testing parameter settings for the trainer.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
        """

        from pytorch_forecasting.data.encoders import GroupNormalizer

        return [
            {
                # Basic test params
                "d_model": 16,
                "n_heads": 2,
                "patch_len": 4,
                "stride": 4,
                "dropout": 0.1,
            },
            {
                "d_model": 32,
                "n_heads": 4,
                "patch_len": 8,
                "stride": 8,
                "dropout": 0.2,
                "activation": "gelu",
            },
            {
                "d_model": 16,
                "n_heads": 2,
                "patch_len": 2,
                "stride": 2,
                "dropout": 0.1,
            },
            {
                "d_model": 24,
                "n_heads": 3,
                "patch_len": 4,
                "stride": 2,
                "dropout": 0.15,
                "data_loader_kwargs": dict(
                    target_normalizer=GroupNormalizer(
                        groups=["agency", "sku"], transformation="softplus"
                    ),
                ),
            },
        ]

    @classmethod
    def _get_test_dataloaders_from(cls, params):
        """
        Get dataloaders from parameters.

        Parameters
        ----------
        params: dict
            Parameters to create dataloaders.
            One of the elements in the list returned by ``get_test_train_params``.

        Returns
        -------
        dataloaders: Dict[str, DataLoader]
            Dict of dataloaders created from the parameters.
            Train, validation, and test dataloaders created from the parameters.
        """
        loss = params.get("loss", None)
        data_loader_kwargs = params.get("data_loader_kwargs", {})

        from pytorch_forecasting.metrics import (
            NegativeBinomialDistributionLoss,
            PoissonLoss,
            TweedieLoss,
        )
        from pytorch_forecasting.tests._conftest import make_dataloaders
        from pytorch_forecasting.tests._data_scenarios import data_with_covariates

        dwc = data_with_covariates()

        if isinstance(loss, NegativeBinomialDistributionLoss):
            dwc = dwc.assign(volume=lambda x: x.volume.round())

        dwc = dwc.copy()
        if isinstance(loss, TweedieLoss | PoissonLoss):
            dwc["target"] = dwc["volume"].clip(1e-3, 1.0)
        else:
            dwc["target"] = dwc["volume"]

        data_loader_default_kwargs = dict(
            target="target",
            time_varying_known_reals=["price_actual"],
            time_varying_unknown_reals=["target"],
            static_categoricals=["agency"],
            add_relative_time_idx=True,
        )
        data_loader_default_kwargs.update(data_loader_kwargs)
        dataloaders_w_covariates = make_dataloaders(dwc, **data_loader_default_kwargs)
        return dataloaders_w_covariates
