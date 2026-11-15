"""Tests for NBEATSx with the v2 forecaster API."""

from lightning.pytorch import Trainer
import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
from pytorch_forecasting.models import NBeatsXForecaster


@pytest.fixture
def data():
    time = np.tile(np.arange(12), 4)
    group = np.repeat(np.arange(4), 12)
    return TimeSeries(
        pd.DataFrame(
            {
                "time": time,
                "group": group,
                "observed": np.sin(time),
                "observed_category": time % 3,
                "known": np.cos(time),
                "known_category": time % 2,
                "static": group % 2,
                "value": 2.0 + np.sin(time) + time / 10,
            }
        ),
        time="time",
        group="group",
        target="value",
        known=["known", "known_category"],
        cat=["observed_category", "known_category", "static"],
        static=["static"],
    )


def test_forecaster_exogenous_fit_predict(data):
    model = NBeatsXForecaster(
        stack_types=["exogenous"],
        widths=[8],
        num_blocks=[1],
        num_block_layers=[1],
        datamodule=EncoderDecoderTimeSeriesDataModule(
            max_encoder_length=4,
            max_prediction_length=2,
            train_val_test_split=(0.75, 0.25),
            batch_size=4,
        ),
        trainer=Trainer(
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        ),
    )
    model.fit(data, save_ckpt=False)
    prediction = model.predict(data, return_info=["x"])
    assert isinstance(prediction, TimeSeries)
    assert prediction.metadata.is_prediction
    assert np.isfinite(prediction.to_pandas()["value"]).all()

    x = prediction.info["x"]
    block = model.model.net_blocks[0]
    with torch.no_grad():
        block.theta_b_fc.weight.zero_()
        block.theta_b_fc.bias.copy_(torch.tensor([2.0, 3.0]))
        block.theta_f_fc.weight.zero_()
        block.theta_f_fc.bias.copy_(torch.tensor([4.0, 5.0]))
        out = model.model(x)
    torch.testing.assert_close(
        out["backcast"].squeeze(-1),
        2 * x["encoder_cont"][..., 1] + 3 * x["encoder_cat"][..., 1],
    )
    torch.testing.assert_close(
        out["prediction"].squeeze(-1),
        4 * x["decoder_cont"][..., 0] + 5 * x["decoder_cat"][..., 0],
    )


@pytest.mark.parametrize(
    ("metadata_change", "match"),
    [
        ({"target": 2}, "single continuous target"),
        ({"target_is_categorical": True}, "single continuous target"),
        ({"min_encoder_length": 2}, "fixed encoder and prediction"),
    ],
)
def test_forecaster_validates_metadata(data, metadata_change, match):
    datamodule = EncoderDecoderTimeSeriesDataModule(time_series=data)
    metadata = dict(datamodule.metadata)
    metadata.update(metadata_change)
    with pytest.raises(ValueError, match=match):
        NBeatsXForecaster.get_cls()(metadata=metadata)
