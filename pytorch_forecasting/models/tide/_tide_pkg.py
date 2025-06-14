"""TiDE package container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class TiDEModel_pkg(_BasePtForecaster):
    """Package container for TiDE Model."""

    _tags = {
        "info:name": "TiDEModel",
        "info:compute": 3,
        "authors": ["Sohaib-Ahmed21"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.tide import TiDEModel

        return TiDEModel

    @classmethod
    def get_test_train_params(cls):
        """Return testing parameter settings for the trainer.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """

        from pytorch_forecasting.data.encoders import GroupNormalizer
        from pytorch_forecasting.metrics import SMAPE

        params = [
            {
                "data_loader_kwargs": dict(
                    add_relative_time_idx=False,
                    # must include this everytime since the data_loader_default_kwargs
                    # include this to be True.
                )
            },
            {
                "temporal_decoder_hidden": 16,
                "data_loader_kwargs": dict(add_relative_time_idx=False),
            },
            {
                "dropout": 0.2,
                "use_layer_norm": True,
                "loss": SMAPE(),
                "data_loader_kwargs": dict(
                    target_normalizer=GroupNormalizer(
                        groups=["agency", "sku"], transformation="softplus"
                    ),
                    add_relative_time_idx=False,
                ),
            },
        ]
        defaults = {"hidden_size": 5}
        for param in params:
            param.update(defaults)
        return params

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
        dataloaders : dict with keys "train", "val", "test", values torch DataLoader
            Dict of dataloaders created from the parameters.
            Train, validation, and test dataloaders.
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
