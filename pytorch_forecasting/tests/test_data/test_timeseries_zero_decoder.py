import numpy as np
import pandas as pd
import torch

from pytorch_forecasting.data.timeseries import TimeSeriesDataSet


def test_timeseries_dataset_zero_prediction_length():
    """Test TimeSeriesDataSet with zero prediction length."""
    data = pd.DataFrame(
        dict(
            value=np.random.rand(30),
            group=np.repeat([0, 1], 15),
            time_idx=np.tile(np.arange(15), 2),
        )
    )

    # Test creation
    dataset = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        min_encoder_length=5,
        max_encoder_length=5,
        min_prediction_length=0,
        max_prediction_length=0,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["value"],
    )

    assert dataset.min_prediction_length == 0
    assert dataset.max_prediction_length == 0

    # Test __getitem__
    x, y = dataset[0]
    assert x["encoder_length"] == 5
    assert x["decoder_length"] == 0
    assert x["x_cat"].shape == (5, 0)
    assert x["x_cont"].shape == (5, 2)
    assert x["encoder_target"].shape == (5,)
    assert y[0].shape == (0,)

    # Test dataloader
    dataloader = dataset.to_dataloader(batch_size=4)
    batch = next(iter(dataloader))
    assert batch[0]["encoder_lengths"].shape == (4,)
    assert (batch[0]["encoder_lengths"] == 5).all()
    assert batch[0]["decoder_lengths"].shape == (4,)
    assert (batch[0]["decoder_lengths"] == 0).all()
    assert batch[0]["decoder_target"].shape == (4, 0)


def test_timeseries_dataset_predict_mode_zero_prediction_length():
    """Test TimeSeriesDataSet in predict mode with zero prediction length."""
    data = pd.DataFrame(
        dict(
            value=np.random.rand(30),
            group=np.repeat([0, 1], 15),
            time_idx=np.tile(np.arange(15), 2),
        )
    )

    dataset = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        min_encoder_length=5,
        max_encoder_length=5,
        min_prediction_length=0,
        max_prediction_length=0,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["value"],
        predict_mode=True,
    )

    assert len(dataset) == 2
    for i in range(len(dataset)):
        x, y = dataset[i]
        assert x["decoder_length"] == 0
        assert x["encoder_length"] == 5
        # Should be at the end of the series
        assert x["encoder_time_idx_start"] + x["encoder_length"] - 1 == 14


def test_timeseries_dataset_mixed_prediction_length_including_zero():
    """Test TimeSeriesDataSet with min_prediction_length=0 and max > 0."""
    data = pd.DataFrame(
        dict(
            value=np.random.rand(30),
            group=np.repeat([0, 1], 15),
            time_idx=np.tile(np.arange(15), 2),
        )
    )

    dataset = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        min_encoder_length=5,
        max_encoder_length=5,
        min_prediction_length=0,
        max_prediction_length=2,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["value"],
        predict_mode=False,
    )

    # Verify that some samples have 0 decoder length
    decoder_lengths = [dataset[i][0]["decoder_length"] for i in range(len(dataset))]
    assert 0 in decoder_lengths
    assert 1 in decoder_lengths
    assert 2 in decoder_lengths
