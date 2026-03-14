import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
from pytorch_forecasting.metrics import MAE, SMAPE
from pytorch_forecasting.models.scinet._scinet_pkg_v2 import SCINet_v2_pkg_v2
from pytorch_forecasting.models.scinet._scinet_v2 import SCINet_v2

CONTEXT_LENGTH = 8
PREDICTION_LENGTH = 4
BATCH_SIZE = 4
N_SERIES = 3
N_SAMPLES = 80


@pytest.fixture
def sample_datamodule():
    """Create a sample EncoderDecoderTimeSeriesDataModule for testing.

    Returns
    -------
    dm : EncoderDecoderTimeSeriesDataModule
        Configured data module with synthetic univariate time series.
    """
    time_idx = np.arange(N_SAMPLES)
    series_data = []
    for i in range(N_SERIES):
        values = np.sin(2 * np.pi * time_idx / 20) + np.random.normal(0, 0.1, N_SAMPLES)
        series_data.append(
            pd.DataFrame({"time_idx": time_idx, "series_id": i, "value": values})
        )
    data = pd.concat(series_data).reset_index(drop=True)

    ts = TimeSeries(
        data,
        time="time_idx",
        group=["series_id"],
        target=["value"],
        num=[],
        cat=[],
        known=[],
        unknown=["value"],
    )

    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=ts,
        max_encoder_length=CONTEXT_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,
        batch_size=BATCH_SIZE,
    )
    dm.setup("fit")
    return dm


def test_scinet_v2_forward_shapes(sample_datamodule):
    """Test that forward pass returns correct output shapes.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    """
    dm = sample_datamodule
    model = SCINet_v2(loss=MAE(), metadata=dm.metadata)

    batch_x, _ = next(iter(dm.train_dataloader()))
    with torch.no_grad():
        out = model(batch_x)

    assert "prediction" in out
    pred = out["prediction"]
    assert pred.ndim == 3, f"prediction must be 3D, got {pred.ndim}D"
    assert pred.shape[1] == PREDICTION_LENGTH
    assert pred.shape[2] == 1


def test_scinet_v2_training_step(sample_datamodule):
    """Test that training_step returns a scalar loss with gradients.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    """
    dm = sample_datamodule
    model = SCINet_v2(loss=MAE(), metadata=dm.metadata)

    batch = next(iter(dm.train_dataloader()))
    result = model.training_step(batch, batch_idx=0)

    assert "loss" in result
    loss = result["loss"]
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    loss.backward()


def test_scinet_v2_validation_step(sample_datamodule):
    """Test that validation_step returns a scalar val_loss.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    """
    dm = sample_datamodule
    model = SCINet_v2(loss=MAE(), metadata=dm.metadata)

    batch = next(iter(dm.train_dataloader()))
    result = model.validation_step(batch, batch_idx=0)

    assert "val_loss" in result
    loss = result["val_loss"]
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


@pytest.mark.parametrize("num_stacks", [1, 2])
def test_scinet_v2_num_stacks(sample_datamodule, num_stacks):
    """Test that different num_stacks run without error.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    num_stacks : int
        Number of stacked SCITree modules.
    """
    dm = sample_datamodule
    model = SCINet_v2(loss=MAE(), metadata=dm.metadata, num_stacks=num_stacks)

    batch_x, _ = next(iter(dm.train_dataloader()))
    with torch.no_grad():
        out = model(batch_x)

    assert out["prediction"].shape[1] == PREDICTION_LENGTH


@pytest.mark.parametrize("num_levels", [1, 2, 3])
def test_scinet_v2_num_levels(sample_datamodule, num_levels):
    """Test that different num_levels run without error.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    num_levels : int
        Depth of the binary decomposition tree.
    """
    dm = sample_datamodule
    model = SCINet_v2(loss=MAE(), metadata=dm.metadata, num_levels=num_levels)

    batch_x, _ = next(iter(dm.train_dataloader()))
    with torch.no_grad():
        out = model(batch_x)

    assert out["prediction"].shape[1] == PREDICTION_LENGTH


def test_scinet_v2_invalid_context_length():
    """Test that invalid context_length raises ValueError."""
    metadata = {
        "max_encoder_length": 7,  # not divisible by 2^3=8
        "max_prediction_length": 4,
        "target": 1,
    }
    with pytest.raises(ValueError, match="divisible"):
        SCINet_v2(metadata=metadata, num_levels=3)


def test_scinet_v2_default_loss(sample_datamodule):
    """Test that loss=None falls back to MAE as default.

    Parameters
    ----------
    sample_datamodule : EncoderDecoderTimeSeriesDataModule
        Fixture providing the data module.
    """
    dm = sample_datamodule
    model = SCINet_v2(metadata=dm.metadata)
    assert isinstance(model.loss, MAE)


def test_scinet_v2_pkg_get_cls():
    """Test that SCINet_v2_pkg_v2.get_cls() returns SCINet_v2."""
    assert SCINet_v2_pkg_v2.get_cls() is SCINet_v2


def test_scinet_v2_pkg_naming_convention():
    """Test that pkg class name follows the convention <model>_pkg_v2."""
    model_cls = SCINet_v2_pkg_v2.get_cls()
    expected_pkg_name = model_cls.__name__ + "_pkg_v2"
    assert SCINet_v2_pkg_v2.__name__ == expected_pkg_name


def test_scinet_v2_pkg_test_train_params():
    """Test that get_test_train_params returns a non-empty list of dicts."""
    params = SCINet_v2_pkg_v2.get_test_train_params()
    assert isinstance(params, list)
    assert len(params) > 0
    for p in params:
        assert isinstance(p, dict)
        assert "datamodule_cfg" in p
