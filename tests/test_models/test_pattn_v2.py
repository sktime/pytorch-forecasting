import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import TslibDataModule
from pytorch_forecasting.data.examples import load_toydata
from pytorch_forecasting.layers import AttentionLayer, FullAttention, PAttnEncoderLayer
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
from pytorch_forecasting.models.pattn import PAttn


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing using v2."""
    data_df = load_toydata(num_series=2, seq_length=24)

    ts = TimeSeries(
        data=data_df,
        time="time_idx",
        target="y",
        group=["series_id"],
        num=["x", "future_known_feature", "static_feature"],
        cat=["category", "static_feature_cat"],
        known=["future_known_feature"],
        unknown=["x", "category"],
        static=["static_feature", "static_feature_cat"],
    )

    dm = TslibDataModule(ts, context_length=16, prediction_length=4, batch_size=4)
    dm.setup()

    return {"data_module": dm, "time_series": ts}


def test_quantile_loss_output(sample_dataset):
    """Test PAttn output shape with quantile loss."""
    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]
    metadata = dm.metadata

    quantiles = [0.1, 0.5, 0.9]
    model = PAttn(
        loss=QuantileLoss(quantiles=quantiles),
        d_model=32,
        n_heads=4,
        e_layers=1,
        d_ff=64,
        patch_len=8,
        stride=4,
        dropout=0.1,
        logging_metrics=[SMAPE(), MAE()],
        metadata=metadata,
    )

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    assert output["prediction"].shape == (
        dm.batch_size,
        metadata["prediction_length"],
        len(quantiles),
    )


def test_univariate_forecast():
    """Test univariate forecasting with PAttn."""
    n_samples = 100
    time_idx = np.arange(n_samples)
    values = np.sin(2 * np.pi * time_idx / 20)

    series = pd.DataFrame({"time_idx": time_idx, "series_id": 0, "value": values})
    ts = TimeSeries(
        series,
        time="time_idx",
        group=["series_id"],
        target=["value"],
        num=[],
        cat=[],
        known=["time_idx"],
        unknown=["value"],
    )

    dm = TslibDataModule(ts, context_length=16, prediction_length=4, batch_size=4)
    dm.setup()
    metadata = dm.metadata

    model = PAttn(
        loss=MAE(),
        d_model=32,
        n_heads=4,
        patch_len=8,
        stride=4,
        metadata=metadata,
    )

    batch = next(iter(dm.train_dataloader()))[0]
    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    assert output["prediction"].shape == (
        dm.batch_size,
        metadata["prediction_length"],
        1,
    )


def test_multivariate_target_forecast():
    """Test multivariate target point forecasting with PAttn."""
    n_samples = 60
    time_idx = np.arange(n_samples)
    df = pd.DataFrame(
        {
            "time_idx": time_idx,
            "series_id": 0,
            "y1": np.sin(time_idx / 5.0),
            "y2": np.cos(time_idx / 5.0),
            "cov": np.linspace(0.0, 1.0, n_samples),
        }
    )
    ts = TimeSeries(
        df,
        time="time_idx",
        group=["series_id"],
        target=["y1", "y2"],
        num=["cov"],
        known=["cov"],
        unknown=["y1", "y2"],
    )
    dm = TslibDataModule(ts, context_length=12, prediction_length=3, batch_size=4)
    dm.setup()

    model = PAttn(
        loss=MAE(),
        d_model=16,
        n_heads=4,
        patch_len=6,
        stride=3,
        metadata=dm.metadata,
    )
    batch = next(iter(dm.train_dataloader()))[0]
    with torch.no_grad():
        output = model(batch)

    assert output["prediction"].shape == (dm.batch_size, 3, 2)


def test_prepare_input_data(sample_dataset):
    """Test preparation of continuous and target historical data."""
    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]

    model = PAttn(loss=MAE(), d_model=16, n_heads=4, metadata=dm.metadata)
    input_data, target_indices = model._prepare_input_data(batch)

    assert input_data.shape[-1] == (
        batch["history_cont"].shape[-1] + batch["history_target"].shape[-1]
    )
    assert target_indices.tolist() == [batch["history_cont"].shape[-1]]
    assert target_indices.dtype == torch.long
    assert target_indices.device == input_data.device

    batch_2d = {**batch, "history_target": batch["history_target"].squeeze(-1)}
    input_2d, target_indices_2d = model._prepare_input_data(batch_2d)
    assert input_2d.shape == input_data.shape
    assert target_indices_2d.tolist() == target_indices.tolist()


def test_encoder_all_channels(sample_dataset):
    """Test _encoder returns all channels when target_indices is None."""
    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]

    model = PAttn(loss=MAE(), d_model=16, n_heads=4, metadata=dm.metadata)
    input_data, _ = model._prepare_input_data(batch)

    with torch.no_grad():
        output = model._encoder(input_data, target_indices=None)

    assert output.shape == (
        dm.batch_size,
        dm.metadata["prediction_length"],
        input_data.shape[-1],
    )


def test_prepare_input_data_error_for_no_history(sample_dataset):
    """Test _prepare_input_data without required target history."""
    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]

    model = PAttn(loss=MAE(), d_model=16, n_heads=4, metadata=dm.metadata)
    batch_without_target = {k: v for k, v in batch.items() if k != "history_target"}

    with pytest.raises(
        ValueError,
        match="No target history found in the input dictionary.",
    ):
        model._prepare_input_data(batch_without_target)


def test_quantile_loss_error_on_multiple_targets(sample_dataset):
    """Test error for quantile forecasting with multiple targets."""
    dm = sample_dataset["data_module"]
    model = PAttn(
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        d_model=16,
        n_heads=4,
        metadata=dm.metadata,
    )

    n_features = model.cont_dim + model.target_dim
    x = torch.randn(dm.batch_size, dm.metadata["context_length"], n_features)
    target_indices = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="Quantile forecasting currently only supports a single target.",
    ):
        model._encoder(x, target_indices)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"d_model": 15, "n_heads": 4}, "divisible by n_heads"),
        ({"patch_len": 0}, "patch_len must be positive"),
        ({"stride": 0}, "stride must be positive"),
        ({"e_layers": 0}, "e_layers must be positive"),
        ({"activation": "tanh"}, "activation must be"),
        ({"patch_len": 25, "stride": 4}, "at least patch_len"),
    ],
)
def test_invalid_init_params(sample_dataset, kwargs, match):
    """Test validation errors for invalid hyperparameters."""
    metadata = sample_dataset["data_module"].metadata
    with pytest.raises(ValueError, match=match):
        PAttn(loss=MAE(), metadata=metadata, **kwargs)


def test_forward_with_target_scale(sample_dataset):
    """Test that prediction uses target scaling when target_scale is provided."""
    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]

    model = PAttn(loss=MAE(), d_model=16, n_heads=4, metadata=dm.metadata)
    batch["target_scale"] = {
        "scale": torch.ones(batch["history_target"].shape[0], 1),
        "center": torch.zeros(batch["history_target"].shape[0], 1),
    }

    captured = {}

    def mock_transform_output(prediction, target_scale):
        captured["prediction"] = prediction
        captured["target_scale"] = target_scale
        return prediction + 1

    model.transform_output = mock_transform_output
    prediction = model(batch)["prediction"]

    assert captured["target_scale"] is batch["target_scale"]
    assert torch.allclose(prediction, captured["prediction"] + 1)


def test_pattn_encoder_layer_output_shape():
    """Test PAttnEncoderLayer output shape and invalid activation guard."""
    layer = PAttnEncoderLayer(
        AttentionLayer(
            FullAttention(mask_flag=False, factor=5, output_attention=False),
            d_model=16,
            n_heads=4,
        ),
        d_model=16,
        d_ff=32,
        dropout=0.0,
        activation="gelu",
    )
    x = torch.randn(3, 5, 16)
    out, attn = layer(x)
    assert out.shape == x.shape
    assert attn is None

    with pytest.raises(ValueError, match="activation must be"):
        PAttnEncoderLayer(
            AttentionLayer(FullAttention(mask_flag=False), d_model=16, n_heads=4),
            d_model=16,
            activation="invalid",
        )
