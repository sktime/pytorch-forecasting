import numpy as np
import pandas as pd
import torch

from pytorch_forecasting.data.data_module import (
    EncoderDecoderTimeSeriesDataModule,
)
from pytorch_forecasting.data.encoders import EncoderNormalizer
from pytorch_forecasting.data.timeseries import TimeSeries


def _make_sample_timeseries():
    num_groups = 10
    seq_length = 100

    groups = []
    times = []
    values = []

    rng = np.random.default_rng(42)

    for g in range(num_groups):
        for t in range(seq_length):
            groups.append(g)
            times.append(pd.Timestamp("2020-01-01") + pd.Timedelta(days=t))
            values.append(10 + 0.1 * t + 5 * np.sin(t / 10) + g * 2 + rng.normal(0, 1))

    df = pd.DataFrame(
        {
            "group": groups,
            "time": times,
            "target": values,
        }
    )

    return TimeSeries(
        data=df,
        time="time",
        target="target",
        group=["group"],
        num=[],
        cat=[],
        known=[],
    )


def _module(dataset, normalizer):
    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=dataset,
        max_encoder_length=20,
        max_prediction_length=5,
        batch_size=2,
        target_normalizer=normalizer,
    )
    dm.setup(stage="fit")
    return dm


def _window_targets(dm):
    series_idx, start, enc_length, pred_length = dm.train_dataset.windows[0]

    raw = dm.train_dataset.preprocessed_data[series_idx]["target"].float()

    return (
        raw[start : start + enc_length],
        raw[start + enc_length : start + enc_length + pred_length],
    )


def test_encoder_normalizer_scales_decoder_target():
    dataset = _make_sample_timeseries()
    dm = _module(dataset, EncoderNormalizer())

    x, y = dm.train_dataset[0]

    encoder_raw, decoder_raw = _window_targets(dm)

    expected = (
        (decoder_raw - encoder_raw.mean()) / encoder_raw.std(unbiased=True)
    ).squeeze(-1)

    # Encoder target was normalised.
    assert abs(float(x["target_past"].mean())) < 1e-4

    # Decoder target must use the encoder's fitted statistics.
    torch.testing.assert_close(
        y.float(),
        expected,
        rtol=1e-4,
        atol=1e-3,
    )

    # Decoder target must not remain in raw scale.
    assert not torch.allclose(
        y.float(),
        decoder_raw.squeeze(-1),
        atol=1e-3,
    ), "y is still raw"
