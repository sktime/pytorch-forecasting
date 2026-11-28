import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import TslibDataModule
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
from pytorch_forecasting.models.film._film_v2 import FiLM


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


@pytest.fixture
def model_with_logging_metrics(sample_dataset):
    """FiLM instance used to test BaseModel logging_metrics registration."""
    dm = sample_dataset["data_module"]
    with pytest.warns(UserWarning):
        model = FiLM(
            loss=MAE(),
            window_size=8,
            modes=2,
            logging_metrics=[SMAPE(), MAE()],
            metadata=dm.metadata,
        )
    return model


@pytest.mark.parametrize(
    "window_size, modes",
    [
        (8, 2),
        (16, 4),
    ],
)
def test_film_init(window_size, modes, sample_dataset):
    """Test FiLM initialization."""

    dm = sample_dataset["data_module"]

    metadata = dm.metadata
    loss = MAE()
    model = FiLM(
        loss=loss,
        window_size=window_size,
        modes=modes,
        metadata=metadata,
    )

    assert model.window_size == window_size
    assert model.modes == modes
    assert model.n_quantiles is None
    assert len(model.legts) == len(model.multiscale)
    assert len(model.spec_convs) == len(model.multiscale)


def test_film_forward(sample_dataset):
    """Test forward pass of FiLM."""

    dm = sample_dataset["data_module"]

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    metadata = dm.metadata

    model = FiLM(loss=MAE(), window_size=8, modes=2, metadata=metadata)

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    assert output["prediction"].shape[0] == dm.batch_size
    assert output["prediction"].shape[1] == metadata["prediction_length"]


def test_quantile_loss_output(sample_dataset):
    """Test FiLM output shape with quantile loss."""

    dm = sample_dataset["data_module"]

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    metadata = dm.metadata

    quantiles = [0.1, 0.5, 0.9]

    model = FiLM(
        loss=QuantileLoss(quantiles=quantiles),
        window_size=8,
        modes=2,
        logging_metrics=[SMAPE(), MAE()],
        metadata=metadata,
    )

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    pred = output["prediction"]
    assert pred.ndim == 3
    assert pred.shape[-1] == len(quantiles)
    assert pred.shape[1] == metadata["prediction_length"]


def test_univariate_forecast():
    """Test univariate forecasting with FiLM."""

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

    model = FiLM(loss=MAE(), window_size=8, modes=2, metadata=metadata)

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    assert output["prediction"].shape[0] == dm.batch_size
    assert output["prediction"].shape[1] == metadata["prediction_length"]


def test_prepare_input_data(sample_dataset):
    """Test preparation of continuous and target historical data."""

    dm = sample_dataset["data_module"]
    batch = next(iter(dm.train_dataloader()))[0]

    model = FiLM(
        loss=MAE(),
        window_size=8,
        modes=2,
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

    model = FiLM(
        loss=MAE(),
        window_size=8,
        modes=2,
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

    model = FiLM(
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        window_size=8,
        modes=2,
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

    model = FiLM(
        loss=MAE(),
        window_size=8,
        modes=2,
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


def test_logging_metrics_is_module_list(model_with_logging_metrics):
    """logging_metrics must be registered as nn.ModuleList so .to() propagates."""
    import torch.nn as nn

    assert isinstance(model_with_logging_metrics.logging_metrics, nn.ModuleList)


def test_logging_metrics_device_propagation(model_with_logging_metrics):
    """Metric state tensors must follow the model when moved to a device."""
    model_with_logging_metrics.to("meta")
    for metric in model_with_logging_metrics.logging_metrics:
        for state_name in metric._defaults:
            val = getattr(metric, state_name)
            if isinstance(val, torch.Tensor):
                assert val.device.type == "meta"
