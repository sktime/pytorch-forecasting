"""TFT package container."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class TFT_pkg_v2(Base_pkg):
    """TFT package container.

    Examples
    --------
    >>> # Package-level usage for TFT (Temporal Fusion Transformer)
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pytorch_forecasting.data import TimeSeries
    >>> from pytorch_forecasting.data.data_module import (
    ...     EncoderDecoderTimeSeriesDataModule,
    ... )
    >>> from pytorch_forecasting.models.temporal_fusion_transformer import (
    ...     TFT_pkg_v2,
    ... )
    >>> from pytorch_forecasting.metrics import SMAPE
    >>>
    >>> # Create minimal synthetic time series data
    >>> rng = np.random.default_rng(42)
    >>> rows = []
    >>> for group in range(2):
    ...     for t in range(20):
    ...         rows.append({
    ...             "group": f"series_{group}",
    ...             "time_idx": int(t),
    ...             "target": float(rng.normal() + t * 0.1),
    ...             "static_cov": int(group),
    ...         })
    >>> df = pd.DataFrame(rows)
    >>>
    >>> # Create TimeSeries object
    >>> ts = TimeSeries(
    ...     data=df,
    ...     time="time_idx",
    ...     target="target",
    ...     group=["group"],
    ...     num=["static_cov"],
    ...     known=["time_idx"],
    ...     static=["static_cov"],
    ... )
    >>>
    >>> # Create data module
    >>> dm = EncoderDecoderTimeSeriesDataModule(
    ...     time_series_dataset=ts,
    ...     max_encoder_length=4,
    ...     max_prediction_length=2,
    ...     batch_size=4,
    ... )
    >>> dm.setup("fit")
    >>>
    >>> # Create model using package interface
    >>> pkg = TFT_pkg_v2(
    ...     model_cfg={
    ...         "hidden_size": 8,
    ...         "attention_head_size": 2,
    ...         "loss": SMAPE(),
    ...     },
    ...     trainer_cfg={"max_epochs": 1, "accelerator": "cpu"},
    ...     datamodule_cfg={"max_encoder_length": 4, "max_prediction_length": 2},
    ... )
    >>> # Training requires Lightning - skip in doctest
    >>> # pkg.fit(dm)  # doctest: +SKIP
    >>> # Predictions also skipped for doctest safety
    >>> # preds = pkg.predict(dm)  # doctest: +SKIP
    """

    _tags = {
        "info:name": "TFT",
        "authors": ["phoeenniixx"],
        "info:compute": 3,
        "info:y_type": ["numeric"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 import TFT

        return TFT

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

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
        import torch.nn as nn

        params = [
            {},
            dict(
                hidden_size=25,
                attention_head_size=5,
            ),
            dict(
                loss=nn.MSELoss(),
                hidden_size=16,
                attention_head_size=4,
            ),
            dict(
                loss=nn.GaussianNLLLoss(),
                output_size=2,
                hidden_size=16,
                attention_head_size=2,
            ),
            dict(datamodule_cfg=dict(max_encoder_length=5, max_prediction_length=3)),
            dict(
                hidden_size=24,
                attention_head_size=8,
                datamodule_cfg=dict(
                    max_encoder_length=5,
                    max_prediction_length=3,
                    add_relative_time_idx=False,
                ),
            ),
            dict(
                hidden_size=12,
                datamodule_cfg=dict(max_encoder_length=7, max_prediction_length=10),
            ),
            dict(attention_head_size=2),
            dict(
                optimizer="adamw",
                lr_scheduler="cosine_annealing",
                lr_scheduler_params={"T_max": 5},
            ),
            dict(
                optimizer="adagrad",
                optimizer_params={"lr": 1e-3},
            ),
        ]

        default_dm_cfg = {"max_encoder_length": 4, "max_prediction_length": 3}

        for param in params:
            current_dm_cfg = param.get("datamodule_cfg", {})
            default_dm_cfg.update(current_dm_cfg)

            param["datamodule_cfg"] = default_dm_cfg

        return params
