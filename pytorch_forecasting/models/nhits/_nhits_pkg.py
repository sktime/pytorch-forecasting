"""NHiTS package container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class NHiTS_pkg(_BasePtForecaster):
    """NHiTS package container."""

    _tags = {
        "info:name": "NHiTS",
        "info:compute": 1,
        "info:pred_type": ["distr", "point", "quantile"],
        "info:y_type": ["numeric"],
        "authors": ["jdb78"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
        "python_dependencies": ["cpflows"],
        "tests:skip_by_name": [
            "test_integration[NHiTS-base_params-0-NormalDistributionLoss]",
            "test_integration[NHiTS-base_params-0-MultivariateNormalDistributionLoss]",
            "test_integration[NHiTS-base_params-0-NegativeBinomialDistributionLoss]",
        ],
    }

    @classmethod
    def get_cls(cls):
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
        clip_target = params.get("clip_target", False)

        from pytorch_forecasting.metrics import (
            MQF2DistributionLoss,
        )
        from pytorch_forecasting.tests._data_scenarios import (
            data_with_covariates,
            dataloaders_fixed_window_without_covariates,
            make_dataloaders,
        )

        # Use fixed window dataloaders for MultivariateNormalDistributionLoss
        if hasattr(
            loss, "get_class_tag"
        ) and "multivariate_normal" in loss.get_class_tag("distribution_type", ""):
            return dataloaders_fixed_window_without_covariates()

        # For other distribution losses, use covariates and apply preprocessing
        distr_types = {"log_normal", "negative_binomial", "mqf2", "beta"}
        if (
            hasattr(loss, "get_class_tag")
            and loss.get_class_tag("distribution_type", "") in distr_types
        ):
            dwc = data_with_covariates()
            if clip_target:
                dwc["target"] = dwc["volume"].clip(1e-3, 1.0)
            else:
                dwc["target"] = dwc["volume"]
            dl_default_kwargs = dict(
                target="volume",
                time_varying_unknown_reals=["volume"],
                add_relative_time_idx=False,
            )
            dl_default_kwargs.update(data_loader_kwargs)

            if loss.get_class_tag("distribution_type", "") == "negative_binomial":
                dwc = dwc.assign(volume=lambda x: x.volume.round())
            elif loss.get_class_tag("distribution_type", "") == "log_normal":
                dwc["volume"] = dwc["volume"].clip(1e-3, 1.0)
            return make_dataloaders(dwc, **dl_default_kwargs)

        if (
            hasattr(loss, "get_class_tag")
            and loss.get_class_tag("info:metric_name", "") == "TweedieLoss"
        ):
            dwc = data_with_covariates()
            dl_default_kwargs = dict(
                target="target",
                time_varying_unknown_reals=["target"],
                add_relative_time_idx=False,
            )
            dl_default_kwargs.update(data_loader_kwargs)
            return make_dataloaders(dwc, **dl_default_kwargs)

        return dataloaders_fixed_window_without_covariates()
