import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pytorch_forecasting.metrics import MAE

if not _check_soft_dependencies("uni2ts", severity="none"):
    pytest.skip("uni2ts not installed", allow_module_level=True)

import torch  # noqa: E402

from pytorch_forecasting.data import TimeSeries  # noqa: E402
from pytorch_forecasting.data._tslib_data_module import TslibDataModule  # noqa: E402
from pytorch_forecasting.models.moirai_moe._moirai_moe_v2 import MoiraiMoE  # noqa: E402


@pytest.fixture
def sample_dataset():
    """Small synthetic dataset for Moirai-MoE smoke tests."""
    n_samples = 80
    n_series = 2
    time_idx = np.arange(n_samples)

    frames = []
    for i in range(n_series):
        values = (
            0.1 * time_idx
            + 5 * np.sin(2 * np.pi * time_idx / 20)
            + np.random.normal(0, 1, n_samples)
        )
        frames.append(
            pd.DataFrame(
                {
                    "time_idx": time_idx,
                    "series_id": i,
                    "value": values,
                }
            )
        )
    data = pd.concat(frames).reset_index(drop=True)

    ts = TimeSeries(
        data,
        time="time_idx",
        group=["series_id"],
        target=["value"],
        num=[],
        cat=[],
        known=["time_idx"],
        unknown=["value"],
    )

    dm = TslibDataModule(ts, context_length=32, prediction_length=8, batch_size=2)
    dm.setup()
    return {"data_module": dm, "time_series": ts}


def test_moirai_moe_init(sample_dataset):
    """Model constructs with TslibDataModule metadata."""
    dm = sample_dataset["data_module"]
    with pytest.warns(UserWarning):
        model = MoiraiMoE(
            loss=MAE(),
            training=False,
            num_samples=5,
            metadata=dm.metadata,
        )
    assert model.pretrained_model_name == "Salesforce/moirai-moe-1.0-R-small"
    assert model.patch_size == 16


def test_moirai_moe_forward_shape(sample_dataset):
    """Forward pass produces (batch, prediction_length, 1) predictions."""
    dm = sample_dataset["data_module"]
    with pytest.warns(UserWarning):
        model = MoiraiMoE(
            loss=MAE(),
            training=False,
            num_samples=5,
            metadata=dm.metadata,
        )

    batch = next(iter(dm.train_dataloader()))[0]
    with torch.no_grad():
        out = model(batch)

    assert "prediction" in out
    pred = out["prediction"]
    assert pred.shape[0] == batch["history_target"].shape[0]
    assert pred.shape[1] == dm.metadata["prediction_length"]
