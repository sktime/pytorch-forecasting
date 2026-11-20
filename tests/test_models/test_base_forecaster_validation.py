"""Early trainer and datamodule validation for the v2 forecaster API."""

from lightning import Trainer
import pytest

from pytorch_forecasting.data.data_module import (
    EncoderDecoderTimeSeriesDataModule,
    TslibDataModule,
)
from pytorch_forecasting.models.dlinear._dlinear_forecaster_v2 import DLinearForecaster


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"trainer": {"max_epochs": 1}}, r"trainer.*trainer settings"),
        ({"trainer": Trainer}, r"trainer.*pass the configured instance"),
        ({"datamodule": TslibDataModule}, r"datamodule.*class TslibDataModule"),
        (
            {
                "datamodule": EncoderDecoderTimeSeriesDataModule(
                    max_encoder_length=8, max_prediction_length=2
                )
            },
            r"datamodule.*must be TslibDataModule",
        ),
    ],
)
def test_rejects_invalid_constructor_inputs(kwargs, match):
    """Reject invalid trainers and datamodules at construction."""
    with pytest.raises(TypeError, match=match):
        DLinearForecaster(**kwargs)


def test_fit_rejects_invalid_trainer_override():
    """Reject an invalid trainer override before using the training data."""
    forecaster = DLinearForecaster()
    with pytest.raises(TypeError, match=r"trainer.*must be Trainer"):
        forecaster.fit(None, trainer={"max_epochs": 1}, save_ckpt=False)
