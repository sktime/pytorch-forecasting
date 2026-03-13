import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer, MultiNormalizer
from pytorch_forecasting.models import LSTMModel


def make_dataset(n_targets=1):
    data = pd.DataFrame(
        {
            "time_idx": np.tile(np.arange(20), 2),
            "group": np.repeat([0, 1], 20),
        }
    )

    if n_targets == 1:
        data["target"] = np.random.rand(len(data))
        targets = "target"
        unknowns = ["target"]
    else:
        for i in range(n_targets):
            data[f"target{i}"] = np.random.rand(len(data))
        targets = [f"target{i}" for i in range(n_targets)]
        unknowns = targets

    dataset = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        group_ids=["group"],
        target=targets,
        time_varying_unknown_reals=unknowns,
        min_encoder_length=5,
        max_encoder_length=5,
        min_prediction_length=2,
        max_prediction_length=2,
        target_normalizer=(
            EncoderNormalizer()
            if n_targets == 1
            else MultiNormalizer([EncoderNormalizer()] * n_targets)
        ),
    )

    return dataset


def test_baseline_lstm_univariate_forward():
    dataset = make_dataset(n_targets=1)

    model = LSTMModel.from_dataset(
        dataset,
        hidden_size=8,
        n_layers=1,
    )

    loader = dataset.to_dataloader(train=True, batch_size=4)
    x, y = next(iter(loader))

    out = model(x)["prediction"]

    assert out.ndim == 3
    assert out.shape[-1] == 1


def test_baseline_lstm_univariate_eval_forward():
    dataset = make_dataset(n_targets=1)

    model = LSTMModel.from_dataset(
        dataset,
        hidden_size=8,
        n_layers=1,
    )
    model.eval()

    loader = dataset.to_dataloader(train=False, batch_size=4)
    x, y = next(iter(loader))

    with torch.no_grad():
        out = model(x)["prediction"]

    assert out.ndim == 3
    assert out.shape[1] == dataset.max_prediction_length
    assert out.shape[-1] == 1


def test_baseline_lstm_multitarget_forward():
    dataset = make_dataset(n_targets=2)

    model = LSTMModel.from_dataset(
        dataset,
        hidden_size=8,
        n_layers=1,
    )

    loader = dataset.to_dataloader(train=True, batch_size=4)
    x, y = next(iter(loader))

    out = model(x)["prediction"]

    # Multi-target returns list of tensors (one per target) for MultiLoss
    if isinstance(out, list):
        assert len(out) == 2
        out = torch.cat(out, dim=-1)  # (batch, time, n_targets)
    assert out.ndim == 3
    assert out.shape[-1] == 2


def test_baseline_lstm_multitarget_loss():
    dataset = make_dataset(n_targets=2)

    model = LSTMModel.from_dataset(
        dataset,
        hidden_size=8,
        n_layers=1,
    )

    loader = dataset.to_dataloader(train=True, batch_size=4)
    x, y = next(iter(loader))

    out = model(x)["prediction"]
    loss = model.loss(out, y)

    assert torch.is_tensor(loss)
    assert loss.ndim == 0


def test_baseline_lstm_backward():
    dataset = make_dataset(n_targets=2)

    model = LSTMModel.from_dataset(
        dataset,
        hidden_size=8,
        n_layers=1,
    )

    loader = dataset.to_dataloader(train=True, batch_size=4)
    x, y = next(iter(loader))

    out = model(x)["prediction"]
    loss = model.loss(out, y)

    loss.backward()  # should not raise


def test_prediction_length():
    dataset = make_dataset(n_targets=2)

    model = LSTMModel.from_dataset(dataset)
    model.eval()

    loader = dataset.to_dataloader(train=False, batch_size=4)
    x, y = next(iter(loader))

    with torch.no_grad():
        out = model(x)["prediction"]
    # Multi-target returns list of tensors; use first for time dim
    out_t = out[0] if isinstance(out, list) else out
    assert out_t.shape[1] == dataset.max_prediction_length
