import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data._tslib_data_module import TslibDataModule
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.data.timeseries import TimeSeries


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


def test_init(sample_timeseries_data):
    """Test the initialization of the data module."""

    tslib_dm = TslibDataModule(
        time_series_dataset=sample_timeseries_data,
        context_length=32,
        prediction_length=16,
        batch_size=8,
    )

    assert tslib_dm.time_series_dataset == sample_timeseries_data
    assert tslib_dm.context_length == 32
    assert tslib_dm.prediction_length == 16
    assert tslib_dm.batch_size == 8
    assert tslib_dm.train_val_test_split == (0.7, 0.15, 0.15)

    assert isinstance(tslib_dm.time_series_metadata, dict)
    assert "cols" in tslib_dm.time_series_metadata


def test_prepare_metadata(tslib_data_module):
    """Test the metadata preparation to ensure correct metadata extraction
    and structure."""

    metadata = tslib_data_module.metadata

    assert isinstance(metadata, dict)

    assert "feature_names" in metadata
    assert "feature_indices" in metadata
    assert "n_features" in metadata
    assert "context_length" in metadata
    assert "prediction_length" in metadata
    assert "freq" in metadata
    assert "features" in metadata

    assert "categorical" in metadata["feature_names"]
    assert "continuous" in metadata["feature_names"]
    assert "static" in metadata["feature_names"]
    assert "known" in metadata["feature_names"]
    assert "unknown" in metadata["feature_names"]
    assert "target" in metadata["feature_names"]
    assert "all" in metadata["feature_names"]
    assert "static_categorical" in metadata["feature_names"]
    assert "static_continuous" in metadata["feature_names"]

    assert "categorical" in metadata["feature_indices"]
    assert "continuous" in metadata["feature_indices"]
    assert "static" in metadata["feature_indices"]
    assert "known" in metadata["feature_indices"]
    assert "unknown" in metadata["feature_indices"]
    assert "target" in metadata["feature_indices"]

    for k in metadata["n_features"]:
        assert k in metadata["n_features"]
        assert metadata["n_features"][k] == len(metadata["feature_names"][k])

    assert metadata["context_length"] == tslib_data_module.context_length
    assert metadata["prediction_length"] == tslib_data_module.prediction_length


def test_setup(tslib_data_module):
    """Test the setup method to ensure datamodule is setup for training,
    testing, and validation."""

    tslib_data_module.setup(stage="fit")
    assert hasattr(tslib_data_module, "train_dataset")
    assert hasattr(tslib_data_module, "val_dataset")
    assert len(tslib_data_module._train_windows) > 0
    assert len(tslib_data_module._val_windows) > 0

    tslib_data_module.setup(stage="test")
    assert hasattr(tslib_data_module, "test_dataset")
    assert len(tslib_data_module._test_windows) > 0

    tslib_data_module.setup(stage="predict")
    assert hasattr(tslib_data_module, "predict_dataset")
    assert len(tslib_data_module._predict_windows) > 0


def test_train_dataloader(tslib_data_module):
    """Test the train dataloader to ensure it returns the batches of the data,
    and all hyperparameters are correctly set."""

    tslib_data_module.setup(stage="fit")
    train_data_loader = tslib_data_module.train_dataloader()

    assert hasattr(train_data_loader, "batch_size")
    assert train_data_loader.batch_size == tslib_data_module.batch_size
    assert train_data_loader.num_workers == tslib_data_module.num_workers

    val_data_loader = tslib_data_module.val_dataloader()
    assert hasattr(val_data_loader, "batch_size")


def test_test_dataloader(tslib_data_module):
    """Test the test dataloader to ensure it returns the batches of the data,
    and all hyperparameters are correctly set."""

    tslib_data_module.setup(stage="test")
    test_data_loader = tslib_data_module.test_dataloader()

    assert hasattr(test_data_loader, "batch_size")
    assert test_data_loader.batch_size == tslib_data_module.batch_size
    assert test_data_loader.num_workers == tslib_data_module.num_workers


def test_predict_dataloader(tslib_data_module):
    """Test the predict dataloader to ensure it returns the batches of the data,
    and all hyperparameters are correctly set."""

    tslib_data_module.setup(stage="predict")
    predict_data_loader = tslib_data_module.predict_dataloader()

    assert hasattr(predict_data_loader, "batch_size")
    assert predict_data_loader.batch_size == tslib_data_module.batch_size
    assert predict_data_loader.num_workers == tslib_data_module.num_workers


def test_tslib_dataset(tslib_data_module):
    """Test the _TslibDataset to ensure it is correctly initialized
    and ensure correct outputs from __getitem__."""

    tslib_data_module.setup(stage="fit")
    assert hasattr(tslib_data_module, "train_dataset")
    train_dataset = tslib_data_module.train_dataset

    assert len(train_dataset) > 0, "The train dataset is empty!"

    sample_x, sample_y = train_dataset[0]

    assert isinstance(sample_x, dict), "Sample x should be a dictionary."
    assert isinstance(sample_y, torch.Tensor), "Sample y should be a PyTorch tensor."

    expected_keys = [
        "history_cont",
        "history_cat",
        "future_cont",
        "future_cat",
        "history_length",
        "future_length",
        "history_mask",
        "future_mask",
        "groups",
        "history_time_idx",
        "future_time_idx",
        "future_target",
        "future_target_len",
    ]

    for key in expected_keys:
        assert key in sample_x, f"Key '{key}' not found in sample_x."

    context_length = tslib_data_module.context_length
    prediction_length = tslib_data_module.prediction_length
    metadata = tslib_data_module.metadata

    assert sample_x["history_cont"].shape[0] == context_length
    assert sample_x["history_cat"].shape[0] == context_length
    assert sample_x["future_cont"].shape[0] == prediction_length
    assert sample_x["future_cat"].shape[0] == prediction_length
    assert sample_x["history_target"].shape[0] == context_length
    assert sample_x["future_target"].shape[0] == prediction_length

    known_cat_count = len(
        [
            name
            for name in metadata["feature_names"]["known"]
            if name in metadata["feature_names"]["categorical"]
        ]
    )
    known_cont_count = len(
        [
            name
            for name in metadata["feature_names"]["known"]
            if name in metadata["feature_names"]["continuous"]
        ]
    )

    print(sample_x["future_cont"].shape)

    assert sample_x["future_cont"].shape[1] == known_cont_count
    assert sample_x["future_cat"].shape[1] == known_cat_count

    assert sample_y.shape[0] == prediction_length

    assert sample_x["history_cont"].dtype == torch.float32
    assert sample_x["future_cont"].dtype == torch.float32
    assert sample_x["history_target"].dtype == torch.float32

    assert sample_y.dtype == torch.float32


def test_collate_fn(tslib_data_module):
    """Test the collate function in the TslibDataModule to ensure it correctly
    collates the data into batches and properly handles stacking of batches."""

    tslib_data_module.setup(stage="fit")
    batch_size = 2

    batches = [tslib_data_module.train_dataset[i] for i in range(batch_size)]

    x_batch, y_batch = tslib_data_module.collate_fn(batches)

    for key in x_batch:
        assert x_batch[key].shape[0] == batch_size

    metadata = tslib_data_module.metadata

    known_cat_count = len(
        [
            name
            for name in metadata["feature_names"]["known"]
            if name in metadata["feature_names"]["categorical"]
        ]
    )
    known_cont_count = len(
        [
            name
            for name in metadata["feature_names"]["known"]
            if name in metadata["feature_names"]["continuous"]
        ]
    )

    assert x_batch["future_cont"].shape[2] == known_cont_count
    assert x_batch["future_cat"].shape[2] == known_cat_count
    # print(x_batch["future_cont"].shape)
    assert y_batch.shape[0] == batch_size
    assert y_batch.shape[1] == tslib_data_module.prediction_length


def test_create_windows(tslib_data_module):
    """Test the _create_windows method to ensures correct creation
    of windows for training, validation and testing."""

    tslib_data_module.setup(stage="fit")
    train_indices = tslib_data_module._train_indices
    train_windows = tslib_data_module._create_windows(train_indices)

    assert len(train_windows) > 0, "No training windows created!"

    for windows in train_windows:
        assert isinstance(windows, tuple), "Windows should be a tuple."

        assert len(windows) == 4, "Each window should have 4 elements."

        series_idx, start_idx, context_length, prediction_length = windows

        assert isinstance(series_idx, int), "series_idx should be an integer."

        assert isinstance(start_idx, int), "start_idx should be an integer."

        assert (
            context_length == tslib_data_module.context_length
        ), "context_length should match the datamodule's context_length."

        assert (
            prediction_length == tslib_data_module.prediction_length
        ), "prediction_length should match the datamodule's prediction_length."

        assert (
            0 <= series_idx < len(tslib_data_module.time_series_dataset)
        ), "series_idx should be within the range of the dataset length."

        min_required_length = context_length + prediction_length

        time_series_dataset = tslib_data_module.time_series_dataset
        # print(type(time_series_dataset[series_idx]))
        sample = time_series_dataset[series_idx]

        if "t" in sample:
            series_length = len(sample["t"])
        elif "y" in sample:
            series_length = len(sample["y"])
        else:
            series_length = len(sample)
        assert (
            start_idx + min_required_length <= series_length
        ), "Window extended beyond series length."

    all_indices = torch.arange(len(tslib_data_module.time_series_dataset))
    all_windows = tslib_data_module._create_windows(all_indices)
    assert len(all_windows) >= len(
        train_windows
    ), "Should have more windows than all indices."

    empty_windows = tslib_data_module._create_windows(torch.tensor([]))

    assert len(empty_windows) == 0, "Should return empty list for empty index."


def test_dataloader_pipeline(tslib_data_module):
    """Test for a single iteration of the dataloader pipeline to
    perform batch retrival and ensure correct data shapes and types."""

    tslib_data_module.setup(stage="fit")
    train_dataloader = tslib_data_module.train_dataloader()

    x_batch, y_batch = next(iter(train_dataloader))

    assert isinstance(x_batch, dict), "x_batch should be a dictionary."
    assert isinstance(y_batch, torch.Tensor), "y_batch should be a PyTorch tensor."

    assert x_batch["history_cont"].shape[1] == tslib_data_module.context_length
    assert x_batch["history_cat"].shape[1] == tslib_data_module.context_length

    metadata = tslib_data_module.metadata

    known_cat_count = len(
        [
            name
            for name in metadata["feature_names"]["known"]
            if name in metadata["feature_names"]["categorical"]
        ]
    )

    known_cont_count = len(
        [
            name
            for name in metadata["feature_names"]["known"]
            if name in metadata["feature_names"]["continuous"]
        ]
    )

    assert x_batch["future_cont"].shape[0] == tslib_data_module.batch_size
    assert x_batch["future_cat"].shape[2] == known_cat_count
    assert x_batch["future_cont"].shape[2] == known_cont_count
    assert x_batch["future_cont"].shape[0] == tslib_data_module.batch_size

    assert y_batch.shape[0] == tslib_data_module.batch_size
    assert y_batch.shape[1] == tslib_data_module.prediction_length


def test_different_split_ratios(sample_timeseries_data):
    """Test the TslibDataModule with different train/val/test split ratios."""

    custom_split = (0.6, 0.2, 0.2)
    dm_custom = TslibDataModule(
        time_series_dataset=sample_timeseries_data,
        context_length=8,
        prediction_length=4,
        batch_size=2,
        train_val_test_split=custom_split,
    )

    dm_custom.setup(stage="fit")

    total_series = len(sample_timeseries_data)
    expected_train = int(total_series * 0.6)
    expected_val = int(total_series * 0.2)
    expected_test = total_series - expected_train - expected_val

    assert len(dm_custom._train_indices) == expected_train
    assert len(dm_custom._val_indices) == expected_val
    assert len(dm_custom._test_indices) == expected_test

    assert dm_custom.train_val_test_split == custom_split

    total_split = (
        len(dm_custom._train_indices)
        + len(dm_custom._val_indices)
        + len(dm_custom._test_indices)
    )
    assert (
        total_split == total_series
    ), "Total split indices should match the dataset length."


def test_preprocess_data(tslib_data_module, sample_timeseries_data):
    """Test the preprocess_data method.
    Ensures alignment and presence of all required features."""

    if not hasattr(tslib_data_module, "_indices"):
        tslib_data_module.setup()

    sample_series_idx = tslib_data_module._train_indices[0]
    processed = tslib_data_module._preprocess_data(sample_series_idx)

    assert "features" in processed
    assert "target" in processed
    assert "static" in processed
    assert "group" in processed
    assert "time_mask" in processed
    assert "continuous" in processed["features"]
    assert "categorical" in processed["features"]
    assert "length" in processed
    assert "timestep" in processed

    original_sample = sample_timeseries_data[sample_series_idx]

    expected_length = len(original_sample["t"])

    assert processed["features"]["categorical"].shape[0] == expected_length
    assert processed["features"]["continuous"].shape[0] == expected_length
    assert processed["target"].shape[0] == expected_length


def test_static_features(tslib_data_module):
    """Test with static features included.

    Validates the static feature support in the TslibDataModule."""

    tslib_data_module.setup(stage="fit")

    metadata = tslib_data_module.metadata

    assert metadata["n_features"]["static_continuous"] == 1

    x, y = tslib_data_module.train_dataset[0]

    assert "static_continuous_features" in x
    assert (
        x["static_continuous_features"].shape[1]
        == metadata["n_features"]["static_continuous"]
    )


def test_multivariate_target():
    """Test with multivariate target (multiple target columns).

    Verifies correct handling of multivariate targets in data pipeline."""
    df = pd.DataFrame(
        {
            "group": np.repeat([0, 1], 50),
            "time": np.tile(pd.date_range("2020-01-01", periods=50), 2),
            "target1": np.random.normal(0, 1, 100),
            "target2": np.random.normal(5, 2, 100),
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
        }
    )

    ts = TimeSeries(
        data=df,
        time="time",
        target=["target1", "target2"],
        group=["group"],
        num=["feature1", "feature2"],
    )

    dm = TslibDataModule(
        time_series_dataset=ts,
        context_length=8,
        prediction_length=4,
        batch_size=2,
    )

    dm.setup(stage="fit")

    x, y = dm.train_dataset[0]

    assert (
        y.shape[-1] == 2
    ), "Target should have two dimensions for n_features for multivariate target."
