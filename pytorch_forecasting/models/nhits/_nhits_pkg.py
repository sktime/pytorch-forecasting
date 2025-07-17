"""NHiTS package container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class NHiTS_pkg(_BasePtForecaster):
    """NHiTS package container."""

    _tags = {
        "info:name": "NHiTS",
        "info:compute": 1,
        "info:pred_type": ["distr", "point", "quantile"],
        "info:y_type": ["category", "numeric"],
        "authors": ["jdb78"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
        "tests:skip_by_name": ["test_integration"],  # still need to debug the tests
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models import NHiTS

        return NHiTS

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
            {
                "backcast_loss_ratio": 0.5,
                "data_loader_kwargs": dict(add_relative_time_idx=False),
            },
            {"hidden_size": 16},
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
            Train, validation, and test dataloaders, in this order.
        """

        loss = params.get("loss", None)
        data_loader_kwargs = params.get("data_loader_kwargs", {})

        from pytorch_forecasting.metrics import (
            ImplicitQuantileNetworkDistributionLoss,
            QuantileLoss,
        )
        from pytorch_forecasting.tests._data_scenarios import (
            data_with_covariates,
            dataloaders_fixed_window_without_covariates,
            make_dataloaders,
        )

        if isinstance(loss, (QuantileLoss, ImplicitQuantileNetworkDistributionLoss)):
            dwc = data_with_covariates()
            dl_default_kwargs = dict(
                target="volume",
                time_varying_unknown_reals=["volume"],
                add_relative_time_idx=True,
            )
            dl_default_kwargs.update(data_loader_kwargs)
            dataloaders_with_covariates = make_dataloaders(dwc, **dl_default_kwargs)
            return dataloaders_with_covariates

        if "backcast_loss_ratio" in params:
            dwc = data_with_covariates()
            dl_default_kwargs = dict(
                target="volume",
                time_varying_unknown_reals=["volume"],
                add_relative_time_idx=True,
            )
            dl_default_kwargs.update(data_loader_kwargs)
            dataloaders_with_covariates = make_dataloaders(dwc, **dl_default_kwargs)
            return dataloaders_with_covariates

        return dataloaders_fixed_window_without_covariates()
