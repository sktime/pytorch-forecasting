import numpy as np
import pandas as pd
import pytest

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.data._tslib_data_module import TslibDataModule


@pytest.fixture
def sample_tslib_dataset():
    """Create a minimal dataset for TslibDataModule tests."""
    n_samples = 50
    n_series = 3

    series_data = []
    for i in range(n_series):
        time_idx = np.arange(n_samples)
        values = np.sin(2 * np.pi * time_idx / 20) + np.random.normal(0, 0.1, n_samples)
        series = pd.DataFrame(
            {
                "time_idx": time_idx,
                "series_id": i,
                "value": values,
                "feat1": np.random.normal(0, 1, n_samples),
            }
        )
        series_data.append(series)

    data = pd.concat(series_data).reset_index(drop=True)

    ts = TimeSeries(
        data,
        time="time_idx",
        group=["series_id"],
        target=["value"],
        num=["feat1"],
        known=["time_idx"],
        unknown=["value", "feat1"],
    )

    return ts


def test_setup_reentrant_guard(sample_tslib_dataset):
    """Regression test for issue #2218.

    Verifies that calling setup('fit') a second time does NOT rebuild the
    train and val datasets from scratch. The guard on line 723 must check the
    correct attribute names ('train_dataset' / 'val_dataset') so that it
    detects previously created datasets and skips re-creation.
    """
    dm = TslibDataModule(
        sample_tslib_dataset,
        context_length=16,
        prediction_length=4,
        batch_size=4,
    )

    dm.setup("fit")

    # Capture the dataset objects created on the first call
    train_dataset_first = dm.train_dataset
    val_dataset_first = dm.val_dataset

    # A second call to setup("fit") must hit the guard and return without
    # recreating the datasets (i.e. the same objects should be returned).
    dm.setup("fit")

    assert dm.train_dataset is train_dataset_first, (
        "setup('fit') must not recreate train_dataset when it already exists. "
        "Check that the hasattr guard uses 'train_dataset', not '_train_dataset'."
    )
    assert dm.val_dataset is val_dataset_first, (
        "setup('fit') must not recreate val_dataset when it already exists. "
        "Check that the hasattr guard uses 'val_dataset', not '_val_dataset'."
    )
