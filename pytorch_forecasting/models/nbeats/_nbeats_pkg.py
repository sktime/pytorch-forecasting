"""NBeats package container."""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class NBeats_pkg(_BasePtForecaster):
    """NBeats package container.

    Examples
    --------
    >>> # Usage example for NBeats (v1 API)
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pytorch_forecasting import NBeats, TimeSeriesDataSet
    >>> from pytorch_forecasting.data import NaNLabelEncoder
    >>> from pytorch_forecasting.data.examples import generate_ar_data
    >>>
    >>> # Generate synthetic time series data
    >>> data = generate_ar_data(seasonality=10.0, timesteps=400, n_series=10, seed=42)
    >>> data["static"] = 2
    >>> data = data.astype(dict(series=str))
    >>>
    >>> # Define time series parameters
    >>> max_encoder_length = 150
    >>> max_prediction_length = 20
    >>> training_cutoff = data["time_idx"].max() - max_prediction_length
    >>>
    >>> # Create training dataset
    >>> training = TimeSeriesDataSet(
    ...     data[lambda x: x.time_idx < training_cutoff],
    ...     time_idx="time_idx",
    ...     target="value",
    ...     categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
    ...     group_ids=["series"],
    ...     min_encoder_length=max_encoder_length,
    ...     max_encoder_length=max_encoder_length,
    ...     max_prediction_length=max_prediction_length,
    ...     min_prediction_length=max_prediction_length,
    ...     time_varying_unknown_reals=["value"],
    ...     randomize_length=None,
    ...     add_relative_time_idx=False,
    ...     add_target_scales=False,
    ... )
    >>>
    >>> # Create validation dataset
    >>> validation = TimeSeriesDataSet.from_dataset(
    ...     training, data, min_prediction_idx=training_cutoff
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
    >>> # Initialize NBeats model from dataset
    >>> model = NBeats.from_dataset(
    ...     training,
    ...     learning_rate=3e-2,
    ...     weight_decay=1e-2,
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
    >>> #            trainer_kwargs=dict(accelerator="cpu"))  # doctest: +SKIP
    """

    _tags = {
        "info:name": "NBeats",
        "info:compute": 1,
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
        "authors": ["jdb78"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
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
        from pytorch_forecasting.metrics import TweedieLoss
        from pytorch_forecasting.tests._data_scenarios import (
            data_with_covariates,
            dataloaders_fixed_window_without_covariates,
            make_dataloaders,
        )

        if isinstance(loss, TweedieLoss):
            dwc = data_with_covariates()
            dl_default_kwargs = dict(
                target="target",
                time_varying_unknown_reals=["target"],
                add_relative_time_idx=False,
            )
            dl_default_kwargs.update(data_loader_kwargs)
            dataloaders_with_covariates = make_dataloaders(dwc, **dl_default_kwargs)
            return dataloaders_with_covariates

        return dataloaders_fixed_window_without_covariates()
