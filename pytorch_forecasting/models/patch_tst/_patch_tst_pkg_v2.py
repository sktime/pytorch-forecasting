"""PatchTST package container for V2."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class PatchTST_pkg_v2(Base_pkg):
    """PatchTST package container for V2.

    Examples
    --------
    >>> # Package-level usage for PatchTST
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pytorch_forecasting.data import TimeSeries
    >>> from pytorch_forecasting.data.data_module import (
    ...     EncoderDecoderTimeSeriesDataModule,
    ... )
    >>> from pytorch_forecasting.models.patch_tst import PatchTST_pkg_v2
    >>> from pytorch_forecasting.metrics import MAE
    >>>
    >>> # Create minimal synthetic time series
    >>> rng = np.random.default_rng(42)
    >>> rows = []
    >>> for group in range(2):
    ...     for t in range(24):
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
    >>> # Create data module with patch-compatible settings
    >>> dm = EncoderDecoderTimeSeriesDataModule(
    ...     time_series_dataset=ts,
    ...     max_encoder_length=16,
    ...     max_prediction_length=3,
    ...     batch_size=4,
    ... )
    >>> dm.setup("fit")
    >>>
    >>> # Create PatchTST model via package interface
    >>> pkg = PatchTST_pkg_v2(
    ...     model_cfg={
    ...         "hidden_size": 16,
    ...         "n_heads": 2,
    ...         "patch_len": 4,
    ...         "stride": 4,
    ...         "loss": MAE(),
    ...     },
    ...     trainer_cfg={"max_epochs": 1, "accelerator": "cpu"},
    ...     datamodule_cfg={"max_encoder_length": 16, "max_prediction_length": 3},
    ... )
    >>> # Training requires Lightning - skip in doctest
    >>> # pkg.fit(dm)  # doctest: +SKIP
    >>> # Predictions also skipped for doctest safety
    >>> # preds = pkg.predict(dm)  # doctest: +SKIP
    """

    _tags = {
        "info:name": "PatchTST_v2",
        "info:compute": 3,
        "info:y_type": ["numeric"],
        "authors": ["nareshmethuku"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
        "tests:skip_by_name": ["test_integration"],
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.patch_tst._patch_tst_v2 import PatchTST_v2

        return PatchTST_v2

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
        from pytorch_forecasting.data.encoders import GroupNormalizer

        return [
            {
                "hidden_size": 16,
                "n_heads": 2,
                "patch_len": 4,
                "stride": 4,
                "dropout": 0.1,
                "datamodule_cfg": {
                    "max_encoder_length": 16,
                    "max_prediction_length": 3,
                },
            },
            {
                "hidden_size": 32,
                "n_heads": 4,
                "patch_len": 8,
                "stride": 8,
                "dropout": 0.2,
                "datamodule_cfg": {
                    "max_encoder_length": 16,
                    "max_prediction_length": 3,
                },
            },
            {
                "hidden_size": 16,
                "n_heads": 2,
                "patch_len": 2,
                "stride": 2,
                "dropout": 0.1,
                "datamodule_cfg": {"max_encoder_length": 4, "max_prediction_length": 2},
            },
            {
                "hidden_size": 24,
                "n_heads": 3,
                "patch_len": 4,
                "stride": 2,
                "dropout": 0.15,
                "datamodule_cfg": dict(
                    max_encoder_length=6,
                    max_prediction_length=3,
                ),
            },
        ]
