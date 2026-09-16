"""FreTS v2 package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class FreTS_pkg_v2(Base_pkg):
    """FreTS v2 package container.

    Examples
    --------
    >>> # Package-level usage for FreTS
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pytorch_forecasting.data import TimeSeries
    >>> from pytorch_forecasting.data.data_module import (
    ...     EncoderDecoderTimeSeriesDataModule,
    ... )
    >>> from pytorch_forecasting.models.frets import FreTS_pkg_v2
    >>> from pytorch_forecasting.metrics import MAE
    >>>
    >>> # Create minimal synthetic time series
    >>> rng = np.random.default_rng(42)
    >>> rows = []
    >>> for group in range(2):
    ...     for t in range(20):
    ...         rows.append({
    ...             "group": f"series_{group}",
    ...             "time_idx": int(t),
    ...             "target": float(rng.normal() + t * 0.05),
    ...         })
    >>> df = pd.DataFrame(rows)
    >>>
    >>> # Create TimeSeries object
    >>> ts = TimeSeries(
    ...     data=df,
    ...     time="time_idx",
    ...     target="target",
    ...     group=["group"],
    ...     known=["time_idx"],
    ... )
    >>>
    >>> # Create data module
    >>> dm = EncoderDecoderTimeSeriesDataModule(
    ...     time_series_dataset=ts,
    ...     max_encoder_length=6,
    ...     max_prediction_length=3,
    ...     batch_size=4,
    ... )
    >>> dm.setup("fit")
    >>>
    >>> # Create FreTS model via package interface
    >>> pkg = FreTS_pkg_v2(
    ...     model_cfg={
    ...         "embed_size": 16,
    ...         "hidden_size": 32,
    ...         "channel_independence": True,
    ...         "loss": MAE(),
    ...     },
    ...     trainer_cfg={"max_epochs": 1, "accelerator": "cpu"},
    ...     datamodule_cfg={"max_encoder_length": 6, "max_prediction_length": 3},
    ... )
    >>> # Training requires Lightning - skip in doctest
    >>> # pkg.fit(dm)  # doctest: +SKIP
    >>> # Predictions also skipped for doctest safety
    >>> # preds = pkg.predict(dm)  # doctest: +SKIP
    """

    _tags = {
        "info:name": "FreTS",
        "info:compute": 2,
        "info:y_type": ["numeric"],
        "authors": ["echo-xiao"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class.

        Returns
        -------
        FreTS : type
            The model class.
        """
        from pytorch_forecasting.models.frets._frets_v2 import FreTS

        return FreTS

    @classmethod
    def get_datamodule_cls(cls):
        """Get datamodule class used for training.

        Returns
        -------
        EncoderDecoderTimeSeriesDataModule : type
            The datamodule class.
        """
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    @classmethod
    def get_test_train_params(cls):
        """Return testing parameter settings for the trainer.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
            Each dict is passed as ``model_cfg`` to the package constructor.
            The key ``"datamodule_cfg"`` inside each dict is forwarded to
            the datamodule constructor.
        """
        from pytorch_forecasting.metrics import MAE, RMSE, SMAPE

        params = [
            {},
            dict(
                embed_size=32,
                hidden_size=64,
                channel_independence=True,
            ),
            dict(
                embed_size=64,
                hidden_size=128,
                channel_independence=False,
                logging_metrics=[SMAPE()],
            ),
            dict(
                embed_size=16,
                hidden_size=32,
                loss=MAE(),
            ),
            dict(
                embed_size=16,
                hidden_size=32,
                sparsity_threshold=0.0,
                loss=RMSE(),
            ),
            dict(
                embed_size=16,
                hidden_size=32,
                optimizer="adamw",
                lr_scheduler="cosine_annealing",
                lr_scheduler_params=dict(T_max=2),
            ),
            dict(
                embed_size=16,
                hidden_size=32,
                channel_independence=False,
                datamodule_cfg=dict(max_encoder_length=12, max_prediction_length=4),
            ),
        ]

        default_dm_cfg = {
            "max_encoder_length": 6,
            "max_prediction_length": 3,
        }

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            dm_cfg = default_dm_cfg.copy()
            dm_cfg.update(current_dm_cfg)
            param["datamodule_cfg"] = dm_cfg

        return params
