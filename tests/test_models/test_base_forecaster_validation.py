"""Early trainer and datamodule validation for the v2 forecaster API."""

from unittest.mock import Mock

from lightning import Trainer
import numpy as np
import pandas as pd
import pytest

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import (
    EncoderDecoderTimeSeriesDataModule,
    TslibDataModule,
)
from pytorch_forecasting.models.dlinear._dlinear_forecaster_v2 import DLinearForecaster


@pytest.mark.parametrize("trainer", [{"max_epochs": 1}, "cpu", 42, Trainer])
def test_invalid_constructor_trainer(trainer, monkeypatch):
    """Bad constructor input is rejected before checkpoint configuration is read."""
    load_config = Mock(side_effect=AssertionError("configuration was loaded"))
    monkeypatch.setattr(DLinearForecaster, "_load_config", load_config)
    with pytest.raises(
        TypeError, match=r"`trainer` must be Trainer.*lightning.Trainer"
    ):
        DLinearForecaster(trainer=trainer)
    load_config.assert_not_called()


@pytest.mark.parametrize(
    "datamodule,match",
    [
        (TslibDataModule, r"`datamodule`.*instance.*not the class TslibDataModule"),
        ({"context_length": 8}, r"`datamodule` must be TslibDataModule, got dict"),
        (
            EncoderDecoderTimeSeriesDataModule(
                max_encoder_length=8, max_prediction_length=2
            ),
            r"`datamodule` must be TslibDataModule, "
            r"got EncoderDecoder.*get_datamodule_cls",
        ),
    ],
)
def test_invalid_constructor_datamodule(datamodule, match, monkeypatch):
    """Reject class objects, config dictionaries, and the wrong datamodule family."""
    load_config = Mock(side_effect=AssertionError("configuration was loaded"))
    monkeypatch.setattr(DLinearForecaster, "_load_config", load_config)
    with pytest.raises(TypeError, match=match):
        DLinearForecaster(datamodule=datamodule)
    load_config.assert_not_called()


@pytest.mark.parametrize("trainer", [{"max_epochs": 1}, "cpu", 42, Trainer])
def test_invalid_fit_trainer(trainer, monkeypatch):
    """Reject a fit override before constructing data or model objects."""
    forecaster = DLinearForecaster()
    build = Mock(side_effect=AssertionError("datamodule was built"))
    monkeypatch.setattr(forecaster, "_build_datamodule", build)
    with pytest.raises(TypeError, match=r"`trainer` must be Trainer"):
        forecaster.fit(None, trainer=trainer, save_ckpt=False)
    build.assert_not_called()
    assert forecaster.model is None
    assert forecaster.trainer_ is None
    assert forecaster.datamodule_ is None


@pytest.mark.parametrize("parameter", ["trainer", "datamodule"])
def test_changed_configuration_checked_before_fit(parameter, monkeypatch):
    """Revalidate stored parameters changed after construction."""
    forecaster = DLinearForecaster()
    setattr(forecaster, parameter, {})
    build = Mock(side_effect=AssertionError("datamodule was built"))
    monkeypatch.setattr(forecaster, "_build_datamodule", build)
    with pytest.raises(TypeError, match=f"`{parameter}` must be"):
        forecaster.fit(None, save_ckpt=False)
    build.assert_not_called()


def test_defaults_and_subclasses_accepted():
    """None defaults and compatible trainer/datamodule subclasses remain valid."""

    class CustomDataModule(TslibDataModule):
        pass

    class CustomTrainer(Trainer):
        pass

    defaults = DLinearForecaster()
    assert defaults.trainer is None
    assert defaults.datamodule is None
    trainer = CustomTrainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    dm = CustomDataModule(context_length=8, prediction_length=2)
    forecaster = DLinearForecaster(trainer=trainer, datamodule=dm)
    assert forecaster.trainer is trainer
    assert forecaster.datamodule is dm
    forecaster._validate_trainer_and_datamodule(trainer)


@pytest.mark.parametrize("use_fit_override", [False, True])
def test_valid_trainer_fits(use_fit_override):
    """A valid trainer still trains, whether configured or passed to fit."""
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    data = TimeSeries(
        pd.DataFrame({"time": np.arange(100), "y": np.sin(np.arange(100))}),
        time="time",
        target="y",
    )
    forecaster = DLinearForecaster(
        trainer=None if use_fit_override else trainer,
        datamodule=TslibDataModule(
            context_length=8, prediction_length=2, batch_size=16
        ),
    )
    forecaster.fit(data, trainer=trainer if use_fit_override else None, save_ckpt=False)
    assert forecaster._is_fitted
    assert forecaster.trainer_.global_step == 1
