"""DeepAR metadata container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class DeepARMetadata(_BasePtForecaster):
    """DeepAR metadata container."""

    _tags = {
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
        "info:compute": 3,
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models import DeepAR

        return DeepAR

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the skbase object.

        ``get_test_params`` is a unified interface point to store
        parameter settings for testing purposes. This function is also
        used in ``create_test_instance`` and ``create_test_instances_and_names``
        to construct test instances.

        ``get_test_params`` should return a single ``dict``, or a ``list`` of ``dict``.

        Each ``dict`` is a parameter configuration for testing,
        and can be used to construct an "interesting" test instance.
        A call to ``cls(**params)`` should
        be valid for all dictionaries ``params`` in the return of ``get_test_params``.

        The ``get_test_params`` need not return fixed lists of dictionaries,
        it can also return dynamic or stochastic parameter settings.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

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
            {"cell_type": "GRU"},
            dict(
                loss=LogNormalDistributionLoss(),
                clip_target=True,
                data_loader_kwargs=dict(
                    target_normalizer=GroupNormalizer(
                        groups=["agency", "sku"], transformation="log"
                    )
                ),
            ),
            dict(
                loss=NegativeBinomialDistributionLoss(),
                clip_target=False,
                data_loader_kwargs=dict(
                    target_normalizer=GroupNormalizer(
                        groups=["agency", "sku"], center=False
                    )
                ),
            ),
            dict(
                loss=BetaDistributionLoss(),
                clip_target=True,
                data_loader_kwargs=dict(
                    target_normalizer=GroupNormalizer(
                        groups=["agency", "sku"], transformation="logit"
                    )
                ),
            ),
            dict(
                data_loader_kwargs=dict(
                    lags={"volume": [2, 5]},
                    target="volume",
                    time_varying_unknown_reals=["volume"],
                    min_encoder_length=2,
                )
            ),
            dict(
                data_loader_kwargs=dict(
                    time_varying_unknown_reals=["volume", "discount"],
                    target=["volume", "discount"],
                    lags={"volume": [2], "discount": [2]},
                )
            ),
            dict(
                loss=ImplicitQuantileNetworkDistributionLoss(hidden_size=8),
            ),
            dict(
                loss=MultivariateNormalDistributionLoss(),
                trainer_kwargs=dict(accelerator="cpu"),
            ),
            dict(
                loss=MultivariateNormalDistributionLoss(),
                data_loader_kwargs=dict(
                    target_normalizer=GroupNormalizer(
                        groups=["agency", "sku"], transformation="log1p"
                    )
                ),
                trainer_kwargs=dict(accelerator="cpu"),
            ),
        ]
