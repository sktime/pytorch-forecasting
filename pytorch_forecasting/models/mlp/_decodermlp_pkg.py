"""DecoderMLP package container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class DecoderMLP_pkg(_BasePtForecaster):
    """DecoderMLP package container.

    Examples
    --------
    >>> # Usage example for DecoderMLP (v1 API)
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pytorch_forecasting import DecoderMLP, TimeSeriesDataSet
    >>> from pytorch_forecasting.data import NaNLabelEncoder
    >>> from pytorch_forecasting.data.examples import generate_ar_data
    >>> from pytorch_forecasting.metrics import QuantileLoss
    >>>
    >>> # Generate synthetic time series data
    >>> data = generate_ar_data(seasonality=10.0, timesteps=400, n_series=10, seed=42)
    >>> data["static"] = 2
    >>> data = data.astype(dict(series=str))
    >>>
    >>> # Define time series parameters
    >>> max_encoder_length = 60
    >>> max_prediction_length = 20
    >>> training_cutoff = data["time_idx"].max() - max_prediction_length
    >>>
    >>> # Create training dataset
    >>> training = TimeSeriesDataSet(
    ...     data[lambda x: x.time_idx <= training_cutoff],
    ...     time_idx="time_idx",
    ...     target="value",
    ...     categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
    ...     group_ids=["series"],
    ...     static_categoricals=["series"],
    ...     time_varying_unknown_reals=["value"],
    ...     max_encoder_length=max_encoder_length,
    ...     max_prediction_length=max_prediction_length,
    ... )
    >>>
    >>> # Create validation dataset
    >>> validation = TimeSeriesDataSet.from_dataset(
    ...     training, data, min_prediction_idx=training_cutoff + 1
    ... )
    >>>
    >>> # Create dataloaders
    >>> batch_size = 32
    >>> train_dataloader = training.to_dataloader(
    ...     train=True, batch_size=batch_size, num_workers=0
    ... )
    >>> val_dataloader = validation.to_dataloader(
    ...     train=False, batch_size=batch_size, num_workers=0
    ... )
    >>>
    >>> # Initialize DecoderMLP model from dataset
    >>> model = DecoderMLP.from_dataset(
    ...     training,
    ...     learning_rate=1e-2,
    ...     hidden_size=16,
    ...     n_hidden_layers=2,
    ...     loss=QuantileLoss(),
    ...     optimizer="Adam",
    ... )
    >>>
    >>> # Training with Lightning - skip in doctest
    >>> # import lightning as L  # doctest: +SKIP
    >>> # trainer = L.Trainer(max_epochs=1, accelerator="cpu")  # doctest: +SKIP
    >>> # trainer.fit(model, train_dataloaders=train_dataloader,  # doctest: +SKIP
    >>> #             val_dataloaders=val_dataloader)  # doctest: +SKIP
    >>>
    >>> # Make predictions - skip in doctest
    >>> # predictions = model.predict(val_dataloader,  # doctest: +SKIP
    >>> #             trainer_kwargs=dict(accelerator="cpu"))  # doctest: +SKIP
    """

    _tags = {
        "info:name": "DecoderMLP",
        "info:compute": 1,
        "info:pred_type": ["distr", "point", "quantile"],
        "info:y_type": ["category", "numeric"],
        "authors": ["jdb78"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": True,
        "python_dependencies": ["cpflows"],
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models import DecoderMLP

        return DecoderMLP

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
            dict(
                data_loader_kwargs=dict(min_prediction_length=2, min_encoder_length=2),
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
        loss = params.get("loss", None)
        import inspect

        from pytorch_forecasting.metrics import (
            CrossEntropy,
            MQF2DistributionLoss,
            NegativeBinomialDistributionLoss,
        )
        from pytorch_forecasting.tests._data_scenarios import (
            data_with_covariates,
            make_dataloaders,
        )

        dwc = data_with_covariates()
        dwc.assign(target=lambda x: x.volume)
        if isinstance(loss, NegativeBinomialDistributionLoss):
            dwc = dwc.assign(target=lambda x: x.volume.round())
        # todo: still need some debugging to add the MQF2DistributionLoss
        # elif inspect.isclass(loss) and issubclass(loss, MQF2DistributionLoss):
        #     dwc = dwc.assign(volume=lambda x: x.volume.round())
        #     data_loader_kwargs["target"] = "volume"
        #     data_loader_kwargs["time_varying_unknown_reals"] = ["volume"]
        elif isinstance(loss, CrossEntropy):
            data_loader_kwargs["target"] = "agency"
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
