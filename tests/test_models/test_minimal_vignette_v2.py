import shutil

import numpy as np
import pandas as pd
import pytest

from pytorch_forecasting._registry import all_objects
from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.models.scinet import SCINetForecaster


@pytest.fixture
def minimal_time_series():
    """Create minimal synthetic TimeSeries data for vignette execution."""
    n_groups = 2
    n_timesteps = 50
    frame = pd.DataFrame(
        {
            "group": np.repeat(np.arange(n_groups), n_timesteps),
            "time": np.tile(np.arange(n_timesteps), n_groups),
            "target": 100.0 + np.tile(np.arange(n_timesteps), n_groups).astype(float),
            "feature": np.tile(np.arange(n_timesteps), n_groups).astype(float),
        }
    )
    return TimeSeries(
        frame,
        time="time",
        target="target",
        group=["group"],
        num=["feature"],
        known=["feature"],
    )


V2_FORECASTERS = all_objects(object_types="forecaster_pytorch_v2")


@pytest.mark.parametrize("forecaster_name,forecaster_cls", V2_FORECASTERS)
def test_minimal_vignette_all_models_defaults(
    forecaster_name, forecaster_cls, minimal_time_series, tmp_path
):
    """Test running a minimal vignette for each v2 model with all default parameters.

    Ensures that default model and default datamodule configurations are mutually
    compatible and run end-to-end (issue #2469).
    """
    fast_trainer_kwargs = {
        "accelerator": "cpu",
        "max_epochs": 1,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
        "enable_checkpointing": False,
        "default_root_dir": str(tmp_path),
    }

    # Initialize model with all defaults
    forecaster = forecaster_cls(trainer_kwargs=fast_trainer_kwargs)

    # Fit on data with default datamodule
    forecaster.fit(minimal_time_series, save_ckpt=False)

    # Predict with defaults
    predictions = forecaster.predict(minimal_time_series)

    assert isinstance(predictions, TimeSeries)
    assert predictions.metadata.is_prediction
    assert len(predictions) > 0

    shutil.rmtree(tmp_path, ignore_errors=True)


def test_scinet_default_compatibility_regression(tmp_path):
    """Regression test for issue #2469 / PR #2434 discussion_r4231012615.

    Default SCINetForecaster and default datamodule must be compatible without
    raising ValueError about context_length not being divisible by 2 ** num_levels.
    """
    frame = pd.DataFrame(
        {
            "group": np.repeat(np.arange(10), 40),
            "time": np.tile(np.arange(40), 10),
            "target": 100.0 + np.tile(np.arange(40), 10),
            "feature": np.tile(np.arange(40), 10).astype(float),
        }
    )
    data = TimeSeries(
        frame,
        time="time",
        target="target",
        group=["group"],
        num=["feature"],
    )

    fast_trainer_kwargs = {
        "accelerator": "cpu",
        "max_epochs": 1,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
        "enable_checkpointing": False,
        "default_root_dir": str(tmp_path),
    }

    forecaster = SCINetForecaster(trainer_kwargs=fast_trainer_kwargs)
    forecaster.fit(data, save_ckpt=False)
    predictions = forecaster.predict(data)

    assert isinstance(predictions, TimeSeries)
    assert predictions.metadata.is_prediction

    shutil.rmtree(tmp_path, ignore_errors=True)
