"""DeepAR metadata container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class DeepARMetadata(_BasePtForecaster):
    """DeepAR metadata container."""

    _tags = {
        "info:name": "DeepAR",
        "info:compute": 3,
        "authors": ["jdb78"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models import DeepAR

        return DeepAR

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
        from pytorch_forecasting.data.encoders import GroupNormalizer
        from pytorch_forecasting.metrics import (
            BetaDistributionLoss,
            ImplicitQuantileNetworkDistributionLoss,
            LogNormalDistributionLoss,
            MultivariateNormalDistributionLoss,
            NegativeBinomialDistributionLoss,
        )

        return [
            {},
            {"cell_type": "GRU", "n_plotting_samples": 100},
            dict(
                loss=LogNormalDistributionLoss(),
                clip_target=True,
                data_loader_kwargs=dict(
                    target_normalizer=GroupNormalizer(
                        groups=["agency", "sku"], transformation="log"
                    )
                ),
                cell_type="LSTM",
                n_plotting_samples=100,
            ),
            dict(
                loss=NegativeBinomialDistributionLoss(),
                clip_target=False,
                data_loader_kwargs=dict(
                    target_normalizer=GroupNormalizer(
                        groups=["agency", "sku"], center=False
                    )
                ),
                cell_type="LSTM",
                n_plotting_samples=100,
            ),
            dict(
                loss=BetaDistributionLoss(),
                clip_target=True,
                data_loader_kwargs=dict(
                    target_normalizer=GroupNormalizer(
                        groups=["agency", "sku"], transformation="logit"
                    )
                ),
                cell_type="LSTM",
                n_plotting_samples=100,
            ),
            dict(
                data_loader_kwargs=dict(
                    lags={"volume": [2, 5]},
                    target="volume",
                    time_varying_unknown_reals=["volume"],
                    min_encoder_length=2,
                ),
                cell_type="LSTM",
                n_plotting_samples=100,
            ),
            dict(
                data_loader_kwargs=dict(
                    time_varying_unknown_reals=["volume", "discount"],
                    target=["volume", "discount"],
                    lags={"volume": [2], "discount": [2]},
                ),
                cell_type="LSTM",
                n_plotting_samples=100,
            ),
            dict(
                loss=ImplicitQuantileNetworkDistributionLoss(hidden_size=8),
                cell_type="LSTM",
                n_plotting_samples=100,
            ),
            dict(
                loss=MultivariateNormalDistributionLoss(),
                cell_type="LSTM",
                n_plotting_samples=100,
                trainer_kwargs=dict(accelerator="cpu"),
            ),
            dict(
                loss=MultivariateNormalDistributionLoss(),
                data_loader_kwargs=dict(
                    target_normalizer=GroupNormalizer(
                        groups=["agency", "sku"], transformation="log1p"
                    )
                ),
                cell_type="LSTM",
                n_plotting_samples=100,
                trainer_kwargs=dict(accelerator="cpu"),
            ),
        ]

    @classmethod
    def _get_test_dataloaders_from(cls, params):
        """Get dataloaders from parameters.

        Parameters
        ----------
        params : dict
            Parameters to create dataloaders.
            One of the elements in the list returned by ``get_test_train_params``.

        Returns
        -------
        dataloaders : tuple of three torch.utils.data.DataLoader
            List of dataloaders created from the parameters.
            Train, validation, and test dataloaders, in this order.
        """
        trainer_kwargs = params.get("trainer_kwargs", {})
        clip_target = params.get("clip_target", False)
        data_loader_kwargs = params.get("data_loader_kwargs", {})

        from pytorch_forecasting.metrics import NegativeBinomialDistributionLoss
        from pytorch_forecasting.tests._conftest import make_dataloaders
        from pytorch_forecasting.tests._data_scenarios import data_with_covariates

        dwc = data_with_covariates()

        if "loss" in trainer_kwargs and isinstance(
            trainer_kwargs["loss"], NegativeBinomialDistributionLoss
        ):
            data_with_covariates = data_with_covariates.assign(
                volume=lambda x: x.volume.round()
            )

        data_with_covariates = data_with_covariates.copy()
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
