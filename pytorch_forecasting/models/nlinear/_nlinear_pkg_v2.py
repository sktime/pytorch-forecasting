"""
Packages container for NLinear model.
"""

from pytorch_forecasting.base._base_pkg import Base_pkg


class NLinear_pkg_v2(Base_pkg):
    """NLinear package container."""

    _tags = {
        "info:name": "NLinear",
        "info:compute": 2,
        "authors": ["mixiancmx", "Sylver.Icy"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": True,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    def get_cls(cls):
        """Get model class."""
        from pytorch_forecasting.models.nlinear._nlinear_v2 import NLinear

        return NLinear

    @classmethod
    def get_datamodule_cls(cls):
        """Get the underlying DataModule class."""
        from pytorch_forecasting.data._tslib_data_module import TslibDataModule

        return TslibDataModule

    @classmethod
    def get_test_dataset_from(cls, **kwargs):
        """Create target-only datasets for NLinear package tests.

        Constructs train/predict TimeSeries objects containing only the target
        column and group identifiers — no exogenous, categorical, or static
        features — matching NLinear's target-history-only input contract.
        """
        from pytorch_forecasting.data import TimeSeries
        from pytorch_forecasting.tests._data_scenarios import data_with_covariates_v2

        training_cutoff = "2016-09-01"
        group_cols = ["agency_encoded", "sku_encoded"]
        cols = ["time_idx"] + group_cols + ["target"]

        raw_data = data_with_covariates_v2()
        training_data = raw_data.loc[raw_data["date"] < training_cutoff, cols].copy()
        validation_data = raw_data[cols].copy()

        ts_kwargs = dict(
            time="time_idx",
            group=group_cols,
            target=["target"],
            num=[],
            cat=None,
            known=[],
            unknown=["target"],
            static=[],
        )

        return {
            "train": TimeSeries(training_data, **ts_kwargs),
            "predict": TimeSeries(validation_data, **ts_kwargs),
        }

    @classmethod
    def get_test_train_params(cls):
        """Return testing parameter settings for the trainer."""
        params = [
            {},
            dict(datamodule_cfg=dict(context_length=12, prediction_length=3)),
            dict(datamodule_cfg=dict(context_length=16, prediction_length=4)),
        ]

        for param in params:
            dm_cfg = {"context_length": 8, "prediction_length": 2}
            dm_cfg.update(param.get("datamodule_cfg", {}))
            param["datamodule_cfg"] = dm_cfg

        return params
