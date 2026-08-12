import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
import torch

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data.data_module import (
    EncoderDecoderTimeSeriesDataModule,
)
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
from pytorch_forecasting.models.nbeats._nbeats_pkg_v2 import NBeats_pkg_v2
from pytorch_forecasting.models.nbeats._nbeats_v2 import NBeats_v2


@pytest.fixture
def sample_dataset():
    """Create sample univariate dataset and datamodule for testing NBeats v2."""
    n_samples = 60
    n_series = 4
    data_list = []

    for i in range(n_series):
        time_idx = np.arange(n_samples)
        trend = 0.05 * time_idx + i
        seasonality = 5 * np.sin(2 * np.pi * time_idx / 12)
        noise = np.random.normal(0, 0.5, n_samples)
        values = trend + seasonality + noise

        data_list.append(
            pd.DataFrame(
                {
                    "time_idx": time_idx,
                    "series_id": f"s_{i}",
                    "value": values.astype(np.float32),
                }
            )
        )

    df = pd.concat(data_list, ignore_index=True)

    ts = TimeSeries(
        df,
        time="time_idx",
        group=["series_id"],
        target=["value"],
        num=[],
        cat=[],
        known=["time_idx"],
        unknown=["value"],
    )

    dm = EncoderDecoderTimeSeriesDataModule(
        ts,
        max_encoder_length=16,
        max_prediction_length=4,
        batch_size=4,
        train_val_test_split=(0.5, 0.25, 0.25),
    )
    dm.setup("fit")
    dm.setup("test")
    return {"data_module": dm, "time_series": ts}


def test_nbeats_v2_init(sample_dataset):
    """Test initialization of NBeats_v2 model."""
    dm = sample_dataset["data_module"]
    loss = MAE()
    model = NBeats_v2(
        loss=loss,
        stack_types=["generic"],
        num_blocks=[2],
        num_block_layers=[3],
        widths=[32],
        backcast_loss_ratio=1.0,
        metadata=dm.metadata,
    )

    assert model.context_length == 16
    assert model.prediction_length == 4
    assert model.n_quantiles == 1
    assert len(model.net_blocks) == 2


def test_nbeats_v2_invalid_params():
    """Test validation errors for invalid parameters."""
    with pytest.raises(ValueError, match="dropout must be non-negative"):
        NBeats_v2(loss=MAE(), dropout=-0.1)

    with pytest.raises(ValueError, match="backcast_loss_ratio must be non-negative"):
        NBeats_v2(loss=MAE(), backcast_loss_ratio=-0.5)

    with pytest.warns(UserWarning, match="dropout is greater than 0.3"):
        model = NBeats_v2(loss=MAE(), dropout=0.5)
        assert model.hparams.dropout == 0.3


def test_nbeats_v2_forward(sample_dataset):
    """Test forward pass of NBeats_v2."""
    dm = sample_dataset["data_module"]
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    model = NBeats_v2(
        loss=MAE(),
        stack_types=["trend", "seasonality"],
        num_blocks=[2, 2],
        widths=[16, 32],
        metadata=dm.metadata,
    )

    with torch.no_grad():
        output = model(batch)

    assert "prediction" in output
    assert "backcast" in output
    assert "trend" in output
    assert "seasonality" in output
    assert "generic" in output

    pred = output["prediction"]
    assert pred.shape[0] == dm.batch_size
    assert pred.shape[1] == dm.metadata["max_prediction_length"]
    assert pred.shape[2] == 1


def test_nbeats_v2_quantile_loss_forward(sample_dataset):
    """Test forward pass with QuantileLoss."""
    dm = sample_dataset["data_module"]
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    quantiles = [0.1, 0.5, 0.9]
    loss = QuantileLoss(quantiles=quantiles)
    model = NBeats_v2(
        loss=loss,
        stack_types=["generic"],
        num_blocks=[1],
        widths=[16],
        metadata=dm.metadata,
    )

    with torch.no_grad():
        output = model(batch)

    assert output["prediction"].shape[-1] == len(quantiles)
    assert output["backcast"].shape[-1] == len(quantiles)


def test_nbeats_v2_training_step(sample_dataset):
    """Test training and validation step with combined backcast loss."""
    dm = sample_dataset["data_module"]
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))

    model = NBeats_v2(
        loss=MAE(),
        stack_types=["trend", "seasonality"],
        num_blocks=[1, 1],
        widths=[16, 16],
        backcast_loss_ratio=1.0,
        logging_metrics=[SMAPE()],
        metadata=dm.metadata,
    )

    train_out = model.training_step(batch, 0)
    assert "loss" in train_out
    assert isinstance(train_out["loss"], torch.Tensor)

    val_out = model.validation_step(batch, 0)
    assert "val_loss" in val_out

    test_out = model.test_step(batch, 0)
    assert "test_loss" in test_out


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required package matplotlib not installed",
)
def test_nbeats_v2_interpretation_plot(sample_dataset):
    """Test interpretation plotting."""
    import matplotlib.pyplot as plt

    dm = sample_dataset["data_module"]
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))[0]

    model = NBeats_v2(
        loss=MAE(),
        stack_types=["trend", "seasonality"],
        num_blocks=[1, 1],
        widths=[16, 16],
        metadata=dm.metadata,
    )

    with torch.no_grad():
        output = model(batch)

    fig = model.plot_interpretation(batch, output, idx=0)
    assert fig is not None
    plt.close(fig)

    fig2 = model.plot_interpretation(
        batch, output, idx=0, plot_seasonality_and_generic_on_secondary_axis=True
    )
    assert fig2 is not None
    plt.close(fig2)


def test_nbeats_v2_transform_output():
    """Test scale transformation."""
    model = NBeats_v2(loss=MAE())
    y_hat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # No scale
    assert torch.equal(model.transform_output(y_hat, None), y_hat)

    # Tensor scale
    scale = torch.tensor([2.0, 2.0])
    res = model.transform_output(y_hat, scale)
    assert res.shape == y_hat.shape

    # Dict scale
    scale_dict = {"scale": torch.tensor(2.0), "center": torch.tensor(1.0)}
    res_dict = model.transform_output(y_hat, scale_dict)
    assert torch.equal(res_dict, y_hat * 2.0 + 1.0)


def test_nbeats_pkg_v2_interface(sample_dataset):
    """Test NBeats_pkg_v2 wrapper interface."""
    pkg = NBeats_pkg_v2()
    assert pkg.get_cls() == NBeats_v2
    assert pkg.get_datamodule_cls() == EncoderDecoderTimeSeriesDataModule

    test_params = pkg.get_test_train_params()
    assert len(test_params) > 0
    assert "datamodule_cfg" in test_params[0]
