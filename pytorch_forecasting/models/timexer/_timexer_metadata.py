"""TimeXer metadata container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class TimeXerMetadata(_BasePtForecaster):
    """TimeXer metdata container."""

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
            {},
            {
                "d_model": 32,
                "n_heads": 4,
                "e_layers": 2,
                "d_ff": 64,
                "patch_length": 4,
                "dropout": 0.2,
                "activation": "gelu",
            },
            {
                "d_model": 16,
                "n_heads": 2,
                "e_layers": 1,
                "d_ff": 32,
                "patch_length": 2,
                "dropout": 0.1,
                "loss": QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            },
            {
                "d_model": 24,
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
                "d_model": 16,
                "n_heads": 2,
                "e_layers": 1,
                "d_ff": 32,
                "patch_length": 2,
                "dropout": 0.1,
                "features": "M",
                "data_loader_kwargs": dict(
                    target=["volume", "volume2"],
                    time_varying_unknown_reals=["volume", "volume2"],
                    target_normalizer=MultiNormalizer(
                        [
                            GroupNormalizer(groups=["agency", "sku"]),
                            GroupNormalizer(groups=["agency", "sku"]),
                        ]
                    ),
                    data_provider_transform=lambda df: df.assign(
                        volume2=df.volume * 0.5
                    ),  # noqa: E501
                ),
            },
        ]
