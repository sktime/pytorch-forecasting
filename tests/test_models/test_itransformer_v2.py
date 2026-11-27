"""
Basic test framework for iTransformer v2 model.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import TslibDataModule
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
from pytorch_forecasting.models.itransformer import ITransformer


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing using v2."""
    n_samples = 100
    n_series = 3

    time_idx = np.arange(n_samples)

    series_data = []
    for i in range(n_series):
        trend = 0.1 * time_idx
        seasonality = 10 * np.sin(2 * np.pi * time_idx / 20)
        noise = np.random.normal(0, 1, n_samples)
        values = trend + seasonality + noise

        series = pd.DataFrame(
            {
                "time_idx": time_idx,
                "series_id": i,
                "value": values,
                "feat1": np.random.normal(0, 1, n_samples),
                "feat2": np.random.normal(0, 1, n_samples),
            }
        )
        series_data.append(series)

    data = pd.concat(series_data).reset_index(drop=True)

    ts = TimeSeries(
        data,
        time="time_idx",
        group=["series_id"],
        target=["value"],
        num=["feat1", "feat2"],
        cat=[],
        known=["time_idx"],
        unknown=["value", "feat1", "feat2"],
    )

    dm = TslibDataModule(ts, context_length=16, prediction_length=4, batch_size=4)
    dm.setup()

    return {"data_module": dm, "time_series": ts}


def test_itransformer_init(sample_dataset):
    """Test ITransformer initialization."""
    dm = sample_dataset["data_module"]
    metadata = dm.metadata
    loss = MAE()
    model = ITransformer(
        loss=loss,
        d_model=16,
        n_heads=2,
        e_layers=1,
        d_ff=32,
        dropout=0.1,
        activation="gelu",
        use_norm=True,
        metadata=metadata,
    )

    assert model.d_model == 16
    assert model.n_heads == 2
    assert model.e_layers == 1
    assert model.d_ff == 32
    assert model.use_norm is True
    assert model.n_quantiles is None


def test_itransformer_forward(sample_dataset):
    """Test forward pass of ITransformer."""
    dm = sample_dataset["data_module"]
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    metadata = dm.metadata
    model = ITransformer(
        loss=MAE(),
        d_model=16,
        n_heads=2,
        e_layers=1,
        d_ff=32,
        metadata=metadata,
    )

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    assert output["prediction"].shape == (
        dm.batch_size,
        metadata["prediction_length"],
        metadata["n_features"]["target"],
    )


def test_itransformer_quantile_loss(sample_dataset):
    """Test ITransformer output shape with quantile loss."""
    dm = sample_dataset["data_module"]
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    metadata = dm.metadata
    quantiles = [0.1, 0.5, 0.9]

    model = ITransformer(
        loss=QuantileLoss(quantiles=quantiles),
        d_model=16,
        n_heads=2,
        e_layers=1,
        d_ff=32,
        logging_metrics=[SMAPE(), MAE()],
        metadata=metadata,
    )

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    pred = output["prediction"]
    assert pred.shape == (
        dm.batch_size,
        metadata["prediction_length"],
        len(quantiles),
    )


def test_itransformer_forward_with_target_scale(sample_dataset):
    """Test that prediction uses target scaling when target_scale is provided."""
    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]

    model = ITransformer(
        loss=MAE(),
        d_model=16,
        n_heads=2,
        e_layers=1,
        d_ff=32,
        metadata=dm.metadata,
    )

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

    assert torch.equal(
        captured["target_scale"]["scale"], batch["target_scale"]["scale"]
    )
    assert torch.allclose(prediction, captured["prediction"] + 1)


def test_itransformer_n_heads_divisibility_error(sample_dataset):
    """Test ValueError when d_model is not divisible by n_heads."""
    dm = sample_dataset["data_module"]
    with pytest.raises(ValueError, match="must be divisible by n_heads"):
        ITransformer(
            loss=MAE(),
            d_model=17,
            n_heads=2,
            metadata=dm.metadata,
        )


def test_itransformer_context_length_mismatch_error(sample_dataset):
    """Test ValueError when input sequence length does not match context_length."""
    dm = sample_dataset["data_module"]
    model = ITransformer(
        loss=MAE(),
        d_model=16,
        n_heads=2,
        metadata=dm.metadata,
    )
    # Provide input with sequence length 20 instead of configured 16
    wrong_batch = {"history_target": torch.randn(2, 20, 1)}
    with pytest.raises(ValueError, match="does not match configured context_length"):
        model(wrong_batch)
