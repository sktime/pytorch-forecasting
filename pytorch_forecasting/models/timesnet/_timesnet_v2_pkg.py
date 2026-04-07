"""TimesNet package layer for PyTorch Forecasting v2."""

from pytorch_forecasting.base._base_pkg import Base_pkg


class TimesNet_pkg_v2(Base_pkg):
    """Package-layer wrapper for TimesNet_v2."""

    _tags = {
        "info:name": "TimesNet_v2",
        "authors": ["pytorch_forecasting_contributors"],
    }

    @classmethod
    def get_cls(cls):
        """Return the underlying ``TimesNet_v2`` model class."""
        from pytorch_forecasting.models.timesnet._timesnet_v2 import TimesNet_v2

        return TimesNet_v2

    @classmethod
    def get_datamodule_cls(cls):
        """Return the DataModule class used to build train / val loaders."""
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    @classmethod
    def get_test_train_params(cls):
        """Return a list of parameter dicts for unit / integration testing.

        Each dict is a valid ``TimesNet_pkg_v2(**params)`` call.
        ``Base_pkg`` machinery merges ``datamodule_cfg`` with a default
        encoder / prediction length before constructing the DataModule.

        Returns
        -------
        list of dict
            Three parameter sets covering:

            * minimal architecture (fast, default loss MAE)
            * medium architecture with explicit MAE loss and custom lengths
            * small architecture with MAPE loss and short sequences
        """
        from pytorch_forecasting.metrics import MAE, MAPE

        params = [
            {},
            dict(
                e_layers=1,
                d_model=16,
                top_k=3,
                d_ff=16,
                num_kernels=4,
                cat_cardinalities=[],
                cat_embedding_dim=8,
            ),
            dict(
                e_layers=2,
                d_model=32,
                top_k=5,
                d_ff=32,
                num_kernels=6,
                cat_cardinalities=[],
                cat_embedding_dim=16,
                loss=MAE(),
                datamodule_cfg=dict(
                    max_encoder_length=5,
                    max_prediction_length=3,
                ),
            ),
            dict(
                e_layers=1,
                d_model=16,
                top_k=2,
                d_ff=16,
                num_kernels=4,
                cat_cardinalities=[],
                cat_embedding_dim=8,
                loss=MAPE(),
                datamodule_cfg=dict(
                    max_encoder_length=4,
                    max_prediction_length=2,
                ),
            ),
        ]

        default_dm_cfg = {"max_encoder_length": 4, "max_prediction_length": 3}

        for param in params:
            merged_dm_cfg = dict(default_dm_cfg)
            merged_dm_cfg.update(param.get("datamodule_cfg", {}))
            param["datamodule_cfg"] = merged_dm_cfg

        return params
