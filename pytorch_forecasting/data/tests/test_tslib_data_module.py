# TODO: Remove this file
import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data.data_module import TslibDataModule
from pytorch_forecasting.data.timeseries import TimeSeries
from pytorch_forecasting.tests._data_scenarios import make_datamodule_test_timeseries


@pytest.fixture(scope="session")
def sample_timeseries_data():
    """Fixture to generate a sample TimeSeries."""

    np.random.seed(42)
    n_series = 20
    n_timesteps = 50

    data = []

    for series_id in range(n_series):
        for time_idx in range(n_timesteps):
            # Generate a target variable with some noise
            target = (
                10
                + 0.1 * time_idx
                + np.sin(2 * np.pi * time_idx / 12)
                + np.random.randn() * 0.5
            )  # noqa: E501

            cat_a = np.random.choice([0, 1, 2])

            feature_1 = np.random.randn() + time_idx * 0.01
            feature_2 = target * 0.8 + np.random.randn() * 0.2
            feature_3 = np.sin(time_idx / 5) + np.random.randn() * 0.1

            static_feature = series_id * 2.5

            data.append(
                {
                    "series_id": series_id,
                    "time_idx": time_idx,
                    "target": target,
                    "cat_a": cat_a,
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "feature_3": feature_3,
                    "static_feature": static_feature,
                }
            )

    df = pd.DataFrame(data)

    time_series = TimeSeries(
        data=df,
        time="time_idx",
        target="target",
        group=["series_id"],
        num=["feature_1", "feature_2", "feature_3"],
        cat=["cat_a"],
        unknown=["feature_2", "target", "cat_a"],
        static=["static_feature"],
        known=["feature_1", "feature_3"],
    )
    return time_series


@pytest.fixture
def tslib_data_module(sample_timeseries_data):
    """Fixture for TSLibDataModule."""
    return TslibDataModule(
        time_series_dataset=sample_timeseries_data,
        context_length=8,
        prediction_length=4,
        batch_size=2,  # Smaller batch size for faster testing
        num_workers=0,  # Avoid multiprocessing issues in tests
    )


def test_tslib_metadata_feature_names_and_indices(tslib_data_module):
    """Tslib metadata contains expected keys, feature groups, and length fields."""
    metadata = tslib_data_module.metadata

    assert isinstance(metadata, dict)

    assert "feature_names" in metadata
    assert "feature_indices" in metadata
    assert "n_features" in metadata
    assert "context_length" in metadata
    assert "prediction_length" in metadata
    assert "freq" in metadata
    assert "features" in metadata

    for group in (
        "categorical",
        "continuous",
        "static",
        "known",
        "unknown",
        "target",
        "all",
        "static_categorical",
        "static_continuous",
    ):
        assert group in metadata["feature_names"]
    for group in (
        "categorical",
        "continuous",
        "static",
        "known",
        "unknown",
        "target",
    ):
        assert group in metadata["feature_indices"]

    assert metadata["context_length"] == tslib_data_module._context_length()
    assert metadata["prediction_length"] == tslib_data_module._prediction_length()


def test_tslib_metadata_n_features_count(tslib_data_module):
    """n_features entries match the length of each feature_names group."""
    metadata = tslib_data_module.metadata
    for key in metadata["n_features"]:
        assert metadata["n_features"][key] == len(metadata["feature_names"][key])


def test_tslib_static_features_metadata():
    """Metadata reports static categorical and continuous feature counts."""
    ts = make_datamodule_test_timeseries(
        include_static=True,
        include_static_categorical=True,
    )
    dm = TslibDataModule(
        time_series_dataset=ts,
        context_length=8,
        prediction_length=4,
        batch_size=2,
        num_workers=0,
    )
    dm.setup(stage="fit")
    metadata = dm.metadata

    assert metadata["n_features"]["static_categorical"] == 1
    assert metadata["n_features"]["static_continuous"] == 1
