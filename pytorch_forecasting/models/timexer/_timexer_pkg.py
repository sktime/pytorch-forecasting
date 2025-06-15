"""TimeXer package container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class TimeXer_pkg(_BasePtForecaster):
    """TimeXer package container."""

    _tags = {
        "info:name": "TimeXer",
        "info:compute": 3,
        "authors": ["PranavBhatP"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models import TimeXer

        return TimeXer

    @classmethod
    def get_test_train_params(cls):
        """
        Return testing parameter settings for the trainer.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
        """

        from pytorch_forecasting.data.encoders import GroupNormalizer, MultiNormalizer
        from pytorch_forecasting.metrics import SMAPE, QuantileLoss

        return [
            {
                # Basic test params
                "hidden_size": 16,
                "patch_length": 1,
                "n_heads": 2,
                "e_layers": 1,
                "d_ff": 32,
                "dropout": 0.1,
            },
            {
                "hidden_size": 32,
                "n_heads": 4,
                "e_layers": 2,
                "d_ff": 64,
                "patch_length": 4,
                "dropout": 0.2,
                "activation": "gelu",
            },
            {
                "hidden_size": 16,
                "n_heads": 2,
                "e_layers": 1,
                "d_ff": 32,
                "patch_length": 2,
                "dropout": 0.1,
                "loss": QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            },
            {
                "hidden_size": 24,
                "n_heads": 3,
                "e_layers": 1,
                "d_ff": 48,
                "patch_length": 3,
                "dropout": 0.15,
                "loss": SMAPE(),
                "data_loader_kwargs": dict(
                    target_normalizer=GroupNormalizer(
                        groups=["agency", "sku"], transformation="softplus"
                    ),
                ),
            },
            {
                "hidden_size": 16,
                "n_heads": 2,
                "e_layers": 1,
                "d_ff": 32,
                "patch_length": 2,
                "dropout": 0.1,
                "features": "M",
                "data_loader_kwargs": dict(
                    target=["volume", "price_regular"],
                    time_varying_unknown_reals=["volume", "price_regular"],
                    target_normalizer=MultiNormalizer(
                        [
                            GroupNormalizer(groups=["agency", "sku"]),
                            GroupNormalizer(groups=["agency", "sku"]),
                        ]
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
        clip_target = params.get("clip_target", False)
        data_loader_kwargs = params.get("data_loader_kwargs", {})

        from pytorch_forecasting.metrics import NegativeBinomialDistributionLoss
        from pytorch_forecasting.tests._conftest import make_dataloaders
        from pytorch_forecasting.tests._data_scenarios import data_with_covariates

        dwc = data_with_covariates()

        if isinstance(loss, NegativeBinomialDistributionLoss):
            dwc = dwc.assign(volume=lambda x: x.volume.round())

        dwc = dwc.copy()
        if clip_target:
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
