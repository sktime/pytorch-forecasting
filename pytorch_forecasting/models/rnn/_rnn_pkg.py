"""RecurrentNetwork package container."""

from pytorch_forecasting.models.base import _BasePtForecaster


class RecurrentNetwork_pkg(_BasePtForecaster):
    """RecurrentNetwork package container."""

    _tags = {
        "info:name": "RecurrentNetwork",
        "info:compute": 2,
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
        "authors": ["jdb78"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": True,
        "tests:skip_by_name": [
            "test_integration[RecurrentNetwork-base_params-2-PoissonLoss]"
        ],
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models import RecurrentNetwork

        return RecurrentNetwork

    @classmethod
    def get_base_test_params(cls):
        """Return testing parameter settings for the trainer.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return [
            {},
            {"cell_type": "GRU"},
            dict(
                data_loader_kwargs=dict(
                    lags={"volume": [2, 5]},
                    target="volume",
                    time_varying_unknown_reals=["volume"],
                    min_encoder_length=2,
                )
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
        dataloaders : dict with keys "train", "val", "test", values torch DataLoader
            Dict of dataloaders created from the parameters.
            Train, validation, and test dataloaders.
        """
        loss = params.get("loss", None)
        clip_target = params.get("clip_target", False)
        data_loader_kwargs = params.get("data_loader_kwargs", {})

        from pytorch_forecasting.metrics import (
            PoissonLoss,
            TweedieLoss,
        )
        from pytorch_forecasting.tests._conftest import make_dataloaders
        from pytorch_forecasting.tests._data_scenarios import data_with_covariates

        dwc = data_with_covariates()

        if isinstance(loss, (TweedieLoss, PoissonLoss)):
            clip_target = True

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
