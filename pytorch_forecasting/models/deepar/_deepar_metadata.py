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
