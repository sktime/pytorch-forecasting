"""NBeats package container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class NBeats_pkg(_BasePtForecaster):
    """NBeats package container."""

    _tags = {
        "info:name": "NBeats",
        "info:compute": 1,
        "info:compatible_loss": ["point"],
        "authors": ["jdb78"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
        "tests:skip_by_name": [
            "test_integration[NBeats-base_params-0-CrossEntropy]",
            "test_integration[NBeats-base_params-1-CrossEntropy]",
        ],
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models import NBeats

        return NBeats

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
        return [{}, {"backcast_loss_ratio": 1.0}]

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
        from pytorch_forecasting.metrics import CrossEntropy, TweedieLoss
        from pytorch_forecasting.tests._data_scenarios import (
            data_with_covariates,
            dataloaders_fixed_window_without_covariates,
            make_dataloaders,
        )

        if isinstance(loss, (CrossEntropy, TweedieLoss)):
            dwc = data_with_covariates()
            dwc.assign(target=lambda x: x.volume)
            dl_default_kwargs = dict(
                target="target",
                time_varying_unknown_reals=["target"],
                add_relative_time_idx=False,
            )
            dl_default_kwargs.update(data_loader_kwargs)
            dataloaders_with_covariates = make_dataloaders(dwc, **dl_default_kwargs)
            return dataloaders_with_covariates

        return dataloaders_fixed_window_without_covariates()
