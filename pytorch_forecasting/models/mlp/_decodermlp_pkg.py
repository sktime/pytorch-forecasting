"""DecoderMLP package container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class DecoderMLP_pkg(_BasePtForecaster):
    """DecoderMLP package container."""

    _tags = {
        "info:name": "DecoderMLP",
        "info:compute": 1,
        "authors": ["jdb78"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": True,
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models import DecoderMLP

        return DecoderMLP

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
        from torchmetrics import MeanSquaredError

        from pytorch_forecasting.metrics import (
            MAE,
            CrossEntropy,
            MultiLoss,
            QuantileLoss,
        )

        return [
            {},
            dict(
                loss=MultiLoss([QuantileLoss(), MAE()]),
                data_loader_kwargs=dict(
                    time_varying_unknown_reals=["volume", "discount"],
                    target=["volume", "discount"],
                ),
            ),
            dict(
                loss=CrossEntropy(),
                data_loader_kwargs=dict(
                    target="agency",
                ),
            ),
            dict(loss=MeanSquaredError()),
            dict(
                loss=MeanSquaredError(),
                data_loader_kwargs=dict(min_prediction_length=1, min_encoder_length=1),
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
            Train, validation, and test dataloaders, in this order.
        """
        data_loader_kwargs = params.get("data_loader_kwargs", {})

        from pytorch_forecasting.tests._data_scenarios import (
            data_with_covariates,
            make_dataloaders,
        )

        dwc = data_with_covariates()
        dwc.assign(target=lambda x: x.volume)
        dl_default_kwargs = dict(
            target="target",
            time_varying_known_reals=["price_actual"],
            time_varying_unknown_reals=["target"],
            static_categoricals=["agency"],
            add_relative_time_idx=True,
        )
        dl_default_kwargs.update(data_loader_kwargs)
        dataloaders_with_covariates = make_dataloaders(dwc, **dl_default_kwargs)
        return dataloaders_with_covariates
