"""Tests for v2 HyperparameterTuner."""

import numpy as np
import pandas as pd
import pytest

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import (
    EncoderDecoderTimeSeriesDataModule,
    TslibDataModule,
)
from pytorch_forecasting.models.dlinear._dlinear_v2 import DLinear
from pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 import TFT
from pytorch_forecasting.tuning.hyperparameter_tuner import HyperparameterTuner

optuna = pytest.importorskip("optuna")


@pytest.fixture(scope="module")
def dummy_ts():
    n_steps = 60
    dfs = [
        pd.DataFrame(
            {
                "time_idx": np.arange(n_steps),
                "series_id": g,
                "value": np.sin(2 * np.pi * np.arange(n_steps) / 20)
                + np.random.normal(0, 0.1, n_steps),
            }
        )
        for g in range(5)
    ]
    return TimeSeries(
        pd.concat(dfs, ignore_index=True),
        time="time_idx",
        target=["value"],
        group=["series_id"],
        num=[],
        cat=[],
        known=["time_idx"],
        unknown=["value"],
    )


@pytest.fixture(scope="module")
def encoder_decoder_datamodule(dummy_ts):
    return EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=dummy_ts,
        max_encoder_length=10,
        max_prediction_length=5,
        batch_size=4,
        train_val_test_split=(0.6, 0.2, 0.2),
    )


@pytest.fixture(scope="module")
def tslib_datamodule(dummy_ts):
    return TslibDataModule(
        time_series_dataset=dummy_ts,
        context_length=10,
        prediction_length=5,
        batch_size=4,
        train_val_test_split=(0.6, 0.2, 0.2),
    )


@pytest.fixture
def trial():
    return optuna.create_study().ask()


class TestInitAndValidation:
    """Constructor validation and DataModule resolution."""

    def test_accepts_prebuilt_encoder_decoder_datamodule(
        self, encoder_decoder_datamodule
    ):
        tuner = HyperparameterTuner(model_cls=TFT, data=encoder_decoder_datamodule)
        assert tuner.datamodule is encoder_decoder_datamodule
        assert "max_encoder_length" in tuner._metadata

    def test_accepts_prebuilt_tslib_datamodule(self, tslib_datamodule):
        tuner = HyperparameterTuner(model_cls=DLinear, data=tslib_datamodule)
        assert tuner.datamodule is tslib_datamodule
        assert "context_length" in tuner._metadata

    def test_accepts_raw_timeseries_dataset(self, dummy_ts):
        """Raw TimeSeries requires explicit datamodule_cls."""
        tuner = HyperparameterTuner(
            model_cls=TFT,
            data=dummy_ts,
            datamodule_cls=EncoderDecoderTimeSeriesDataModule,
        )
        assert tuner.datamodule is not None
        assert "max_encoder_length" in tuner._metadata

    def test_raw_timeseries_without_datamodule_cls_raises(self, dummy_ts):
        """Passing raw TimeSeries without datamodule_cls raises ValueError."""
        with pytest.raises(ValueError, match="'datamodule_cls' must be specified"):
            HyperparameterTuner(model_cls=TFT, data=dummy_ts)

    def test_invalid_datamodule_cls_type_raises(self, dummy_ts):
        """Passing a non-subclass of LightningDataModule raises TypeError."""
        with pytest.raises(
            TypeError, match="must be a subclass of LightningDataModule"
        ):
            HyperparameterTuner(model_cls=TFT, data=dummy_ts, datamodule_cls=str)

    def test_mismatched_datamodule_instance_raises(self, tslib_datamodule):
        """
        Passing a prebuilt DataModule that contradicts
        explicit datamodule_cls raises TypeError.
        """
        with pytest.raises(
            TypeError,
            match=(
                "Expected datamodule of type EncoderDecoderTimeSeriesDataModule, "
                "got TslibDataModule"
            ),
        ):
            HyperparameterTuner(
                model_cls=TFT,
                data=tslib_datamodule,
                datamodule_cls=EncoderDecoderTimeSeriesDataModule,
            )


class TestHyperparameterDiscovery:
    """Search space auto-discovery and parameter filtering."""

    def test_discovers_tft_default_types(self, encoder_decoder_datamodule, trial):
        tuner = HyperparameterTuner(model_cls=TFT, data=encoder_decoder_datamodule)
        cfg = tuner._discover_hyperparameters(trial)
        assert isinstance(cfg["dropout"], float)
        assert isinstance(cfg["hidden_size"], int)
        assert isinstance(cfg["attention_head_size"], int)
        assert "metadata" not in cfg

    def test_discovers_dlinear_default_types(self, tslib_datamodule, trial):
        tuner = HyperparameterTuner(model_cls=DLinear, data=tslib_datamodule)
        cfg = tuner._discover_hyperparameters(trial)
        assert isinstance(cfg["moving_avg"], int)
        assert isinstance(cfg["individual"], bool)
        assert "metadata" not in cfg

    def test_excludes_fixed_and_base_params(self, encoder_decoder_datamodule, trial):
        tuner = HyperparameterTuner(
            model_cls=TFT, data=encoder_decoder_datamodule, hidden_size=128
        )
        cfg = tuner._discover_hyperparameters(trial)
        assert "hidden_size" not in cfg
        for base_p in ("loss", "optimizer", "optimizer_params", "lr_scheduler"):
            assert base_p not in cfg

    def test_custom_range_overrides(self, encoder_decoder_datamodule, trial):
        tuner = HyperparameterTuner(model_cls=TFT, data=encoder_decoder_datamodule)
        cfg = tuner._discover_hyperparameters(
            trial, custom_ranges={"hidden_size": [32, 64], "dropout": (0.0, 1.0)}
        )
        assert cfg["hidden_size"] in [32, 64]
        assert 0.0 <= cfg["dropout"] <= 1.0

    def test_fixed_loss_preserved(self, encoder_decoder_datamodule, trial):
        import torch.nn as nn

        tuner = HyperparameterTuner(
            model_cls=TFT, data=encoder_decoder_datamodule, loss=nn.L1Loss()
        )
        cfg = tuner._discover_hyperparameters(trial)
        assert "loss" not in cfg
        assert isinstance(tuner.fixed_hparams["loss"], nn.L1Loss)


class TestOptimize:
    @pytest.mark.slow
    def test_optimize_tft(self, encoder_decoder_datamodule):
        tuner = HyperparameterTuner(
            model_cls=TFT,
            data=encoder_decoder_datamodule,
            hidden_size=8,
            attention_head_size=2,
            output_size=1,
        )
        study = tuner.optimize(n_trials=1, max_epochs=1, timeout=120)
        assert isinstance(study, optuna.Study)
        assert len(study.trials) == 1

    @pytest.mark.slow
    def test_optimize_dlinear(self, tslib_datamodule):
        tuner = HyperparameterTuner(
            model_cls=DLinear,
            data=tslib_datamodule,
            moving_avg=5,
            individual=False,
        )
        study = tuner.optimize(n_trials=1, max_epochs=1, timeout=120)
        assert isinstance(study, optuna.Study)
        assert len(study.trials) == 1
