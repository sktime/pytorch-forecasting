"""Unit tests for TimesNet v2 layers and model."""

import pytest
import torch

from pytorch_forecasting.layers._blocks import TimesBlock, fft_for_period
from pytorch_forecasting.layers._convolution import InceptionBlockV1
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_forecasting.models.timesnet import TimesNet


def _make_metadata(
    context_length: int = 8,
    prediction_length: int = 2,
    cont_dim: int = 2,
    target_dim: int = 1,
) -> dict:
    """Build minimal TslibBaseModel metadata for unit tests."""
    return {
        "context_length": context_length,
        "prediction_length": prediction_length,
        "n_features": {
            "continuous": cont_dim,
            "categorical": 0,
            "target": target_dim,
        },
        "feature_names": {
            "continuous": [f"cont_{i}" for i in range(cont_dim)],
            "categorical": [],
            "target": [f"target_{i}" for i in range(target_dim)],
        },
        "features": "M" if target_dim > 1 else "MS",
    }


@pytest.mark.parametrize("init_weight", [True, False])
def test_inception_block_v1_shape(init_weight):
    block = InceptionBlockV1(
        in_channels=8, out_channels=16, num_kernels=3, init_weight=init_weight
    )
    x = torch.randn(2, 8, 6, 4)
    out = block(x)
    assert out.shape == (2, 16, 6, 4)


@pytest.mark.parametrize("seq_len", [1, 3, 16])
def test_fft_for_period_edge_cases(seq_len):
    x = torch.randn(2, seq_len, 8)
    periods, amplitudes = fft_for_period(x, k=5)
    assert len(periods) >= 1
    assert all(p >= 1 for p in periods)
    assert amplitudes.shape == (2, len(periods))


def test_fft_for_period_all_zeros():
    x = torch.zeros(2, 8, 4)
    periods, amplitudes = fft_for_period(x, k=3)
    assert len(periods) >= 1
    assert all(p >= 1 for p in periods)
    assert amplitudes.shape == (2, len(periods))


@pytest.mark.parametrize("seq_len", [10, 13])
def test_times_block_shape(seq_len):
    block = TimesBlock(d_model=16, d_ff=16, top_k=3, num_kernels=2)
    x = torch.randn(2, seq_len, 16)
    out = block(x)
    assert out.shape == (2, seq_len, 16)


@pytest.mark.parametrize("cont_dim,target_dim", [(0, 1), (2, 1), (2, 3)])
def test_timesnet_point_forward_shape(cont_dim, target_dim):
    metadata = _make_metadata(
        context_length=8,
        prediction_length=3,
        cont_dim=cont_dim,
        target_dim=target_dim,
    )
    model = TimesNet(
        loss=MAE(),
        d_model=16,
        d_ff=16,
        e_layers=1,
        top_k=2,
        num_kernels=2,
        dropout=0.0,
        metadata=metadata,
    )
    batch = {"history_target": torch.randn(2, 8, target_dim)}
    if cont_dim > 0:
        batch["history_cont"] = torch.randn(2, 8, cont_dim)

    out = model(batch)
    assert out["prediction"].shape == (2, 3, target_dim)


def test_timesnet_quantile_forward_shape():
    loss = QuantileLoss()
    metadata = _make_metadata(
        context_length=8, prediction_length=3, cont_dim=1, target_dim=1
    )
    model = TimesNet(
        loss=loss,
        d_model=16,
        d_ff=16,
        e_layers=1,
        top_k=2,
        num_kernels=2,
        metadata=metadata,
    )
    batch = {
        "history_cont": torch.randn(2, 8, 1),
        "history_target": torch.randn(2, 8, 1),
    }
    out = model(batch)
    assert out["prediction"].shape == (2, 3, len(loss.quantiles))


def test_timesnet_quantile_multitarget_raises():
    metadata = _make_metadata(cont_dim=1, target_dim=2)
    with pytest.raises(ValueError, match="single target"):
        TimesNet(
            loss=QuantileLoss(),
            d_model=16,
            d_ff=16,
            e_layers=1,
            top_k=2,
            num_kernels=2,
            metadata=metadata,
        )


def test_timesnet_missing_target_raises():
    metadata = _make_metadata(cont_dim=1, target_dim=1)
    model = TimesNet(
        loss=MAE(),
        d_model=16,
        d_ff=16,
        e_layers=1,
        top_k=2,
        num_kernels=2,
        metadata=metadata,
    )
    with pytest.raises(ValueError, match="No target history"):
        model({"history_cont": torch.randn(2, 8, 1)})
