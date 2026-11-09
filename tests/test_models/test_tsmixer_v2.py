import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import TslibDataModule
from pytorch_forecasting.data.examples import load_toydata
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
from pytorch_forecasting.models.tsmixer._tsmixer_v2 import TSMixer


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
    """Test TSMixer output shape with quantile loss."""

    dm = sample_dataset["data_module"]

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    metadata = dm.metadata

    quantiles = [0.1, 0.5, 0.9]

    model = TSMixer(
        loss=QuantileLoss(quantiles=quantiles),
        d_model=32,
        e_layers=2,
        dropout=0.1,
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


def test_univariate_forecast():
    """Test univariate forecasting with TSMixer."""

    n_samples = 100
    time_idx = np.arange(n_samples)
    values = np.sin(2 * np.pi * time_idx / 20) + np.random.normal(0, 0.1, n_samples)

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

    model = TSMixer(
        loss=MAE(),
        d_model=32,
        e_layers=1,
        dropout=0.1,
        metadata=metadata,
    )

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    assert output["prediction"].shape == (
        dm.batch_size,
        metadata["prediction_length"],
        1,
    )


def test_prepare_input_data(sample_dataset):
    """Test preparation of continuous and target historical data."""

    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]  # One sample batch

    model = TSMixer(
        loss=MAE(),
        metadata=dm.metadata,
    )

    input_data, target_indices = model._prepare_input_data(batch)

    assert input_data.shape[-1] == (
        batch["history_cont"].shape[-1] + batch["history_target"].shape[-1]
    )

    assert target_indices.tolist() == [batch["history_cont"].shape[-1]]

    assert target_indices.dtype == torch.long
    assert target_indices.device == input_data.device


def test_prepare_input_data_error_for_no_history(sample_dataset):
    """Test _prepare_input_data without the required history."""

    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]

    model = TSMixer(
        loss=MAE(),
        metadata=dm.metadata,
    )

    batch_without_target = {
        key: value for key, value in batch.items() if key != "history_target"
    }

    with pytest.raises(
        ValueError,
        match="No target history found in the input dictionary.",
    ):
        model._prepare_input_data(batch_without_target)


def test_quantile_loss_error_on_multiple_targets(sample_dataset):
    """Test error for quantile forecasting with multiple targets."""

    dm = sample_dataset["data_module"]

    model = TSMixer(
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        metadata=dm.metadata,
    )

    batch_size = dm.batch_size
    context_length = dm.metadata["context_length"]
    n_features = model.enc_in

    x = torch.randn(batch_size, context_length, n_features)

    target_indices = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="Quantile forecasting currently only supports a single target.",
    ):
        model._encoder(x, target_indices)


def test_forward_with_target_scale(sample_dataset):
    """Test that prediction uses target scaling when target_scale is provided."""
    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]

    model = TSMixer(
        loss=MAE(),
        metadata=dm.metadata,
    )

    batch["target_scale"] = torch.ones(batch["history_target"].shape[0], 2)

    captured = {}

    def mock_transform_output(prediction, target_scale):
        captured["prediction"] = prediction
        captured["target_scale"] = target_scale
        return prediction + 1

    model.transform_output = mock_transform_output

    prediction = model(batch)["prediction"]

    assert torch.equal(captured["target_scale"], batch["target_scale"])
    assert torch.allclose(prediction, captured["prediction"] + 1)
