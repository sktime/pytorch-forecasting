import numpy as np
import pandas as pd
import pytest

from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
from pytorch_forecasting.data.timeseries import TimeSeries


@pytest.fixture
def sample_timeseries_data():
    """Create a sample time series dataset with only numerical values."""
    num_groups = 10
    seq_length = 100

    groups = []
    times = []
    values = []
    categorical_feature = []
    continuous_feature1 = []
    continuous_feature2 = []
    known_future = []

    for g in range(num_groups):
        for t in range(seq_length):
            groups.append(g)
            times.append(pd.Timestamp("2020-01-01") + pd.Timedelta(days=t))

            value = 10 + 0.1 * t + 5 * np.sin(t / 10) + g * 2 + np.random.normal(0, 1)
            values.append(value)

            categorical_feature.append(np.random.choice([0, 1, 2]))

            continuous_feature1.append(np.random.normal(g, 1))
            continuous_feature2.append(value * 0.5 + np.random.normal(0, 0.5))

            known_future.append(t % 7)

    df = pd.DataFrame(
        {
            "group": groups,
            "time": times,
            "target": values,
            "cat_feat": categorical_feature,
            "cont_feat1": continuous_feature1,
            "cont_feat2": continuous_feature2,
            "known_future": known_future,
        }
    )

    time_series = TimeSeries(
        data=df,
        time="time",
        target="target",
        group=["group"],
        num=["cont_feat1", "cont_feat2", "known_future"],
        cat=["cat_feat"],
        known=["known_future"],
    )

    return time_series


@pytest.fixture
def data_module(sample_timeseries_data):
    """Create a data module instance."""
    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=sample_timeseries_data,
        max_encoder_length=24,
        max_prediction_length=12,
        batch_size=4,
        train_val_test_split=(0.7, 0.15, 0.15),
    )
    return dm


def test_init(sample_timeseries_data):
    """Test the initialization of the data module.

    Verifies hyperparameter assignment and basic time_series_metadata creation."""
    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=sample_timeseries_data,
        max_encoder_length=24,
        max_prediction_length=12,
        batch_size=8,
    )

    assert dm.max_encoder_length == 24
    assert dm.max_prediction_length == 12
    assert dm._min_encoder_length == 24
    assert dm._min_prediction_length == 12
    assert dm.batch_size == 8
    assert dm.train_val_test_split == (0.7, 0.15, 0.15)

    assert isinstance(dm.time_series_metadata, dict)
    assert "cols" in dm.time_series_metadata


def test_prepare_metadata(data_module):
    """Test the metadata preparation method.

    Ensures that internal metadata keys are created correctly."""
    metadata = data_module._prepare_metadata()

    assert "encoder_cat" in metadata
    assert "encoder_cont" in metadata
    assert "decoder_cat" in metadata
    assert "decoder_cont" in metadata
    assert "target" in metadata
    assert "max_encoder_length" in metadata
    assert "max_prediction_length" in metadata

    assert metadata["max_encoder_length"] == 24
    assert metadata["max_prediction_length"] == 12


def test_metadata_property(data_module):
    """Test the metadata property.

    Confirms caching behavior and correct feature counts."""
    metadata = data_module.metadata

    # Should return the same object when called multiple times (caching)
    assert data_module.metadata is metadata

    assert metadata["encoder_cat"] == 1  # cat_feat
    assert metadata["encoder_cont"] == 3  # cont_feat1, cont_feat2, known_future
    assert metadata["decoder_cat"] == 0  # No categorical features marked as known
    assert metadata["decoder_cont"] == 1  # Only known_future marked as known


def test_setup(data_module):
    """Test the setup method that prepares the datasets."""
    data_module.setup(stage="fit")
    print(data_module._val_indices)
    assert hasattr(data_module, "train_dataset")
    assert hasattr(data_module, "val_dataset")
    assert len(data_module.train_windows) > 0
    assert len(data_module.val_windows) > 0

    data_module.setup(stage="test")
    assert hasattr(data_module, "test_dataset")
    assert len(data_module.test_windows) > 0

    data_module.setup(stage="predict")
    assert hasattr(data_module, "predict_dataset")
    assert len(data_module.predict_windows) > 0


def test_create_windows(data_module):
    """Test the window creation logic.

    Validates window structure and length settings."""
    data_module.setup()

    windows = data_module._create_windows(data_module._train_indices)

    assert len(windows) > 0

    for window in windows:
        assert len(window) == 4
        assert window[2] == data_module.max_encoder_length
        assert window[3] == data_module.max_prediction_length


def test_dataloader_creation(data_module):
    """Test that dataloaders are created correctly.

    Checks batch sizes and dataloader instantiation across all stages."""
    data_module.setup()

    train_loader = data_module.train_dataloader()
    assert train_loader.batch_size == data_module.batch_size
    assert train_loader.num_workers == data_module.num_workers

    val_loader = data_module.val_dataloader()
    assert val_loader.batch_size == data_module.batch_size

    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    assert test_loader.batch_size == data_module.batch_size

    data_module.setup(stage="predict")
    predict_loader = data_module.predict_dataloader()
    assert predict_loader.batch_size == data_module.batch_size


def test_processed_dataset(data_module):
    """Test the internal ProcessedEncoderDecoderDataset class.

    Verifies sample structure and tensor dimensions for encoder/decoder inputs."""
    data_module.setup()

    assert len(data_module.train_dataset) == len(data_module.train_windows)
    assert len(data_module.val_dataset) == len(data_module.val_windows)

    x, y = data_module.train_dataset[0]

    required_keys = [
        "encoder_cat",
        "encoder_cont",
        "decoder_cat",
        "decoder_cont",
        "encoder_lengths",
        "decoder_lengths",
        "decoder_target_lengths",
        "groups",
        "encoder_time_idx",
        "decoder_time_idx",
        "target_scale",
        "encoder_mask",
        "decoder_mask",
    ]

    for key in required_keys:
        assert key in x

    assert x["encoder_cat"].shape[0] == data_module.max_encoder_length
    assert x["decoder_cat"].shape[0] == data_module.max_prediction_length

    metadata = data_module.time_series_metadata
    known_cat_count = len(
        [
            col
            for col in metadata["cols"]["x"]
            if metadata["col_type"].get(col) == "C"
            and metadata["col_known"].get(col) == "K"
        ]
    )

    known_cont_count = len(
        [
            col
            for col in metadata["cols"]["x"]
            if metadata["col_type"].get(col) == "F"
            and metadata["col_known"].get(col) == "K"
        ]
    )

    assert x["decoder_cat"].shape[1] == known_cat_count
    assert x["decoder_cont"].shape[1] == known_cont_count

    assert y.shape[0] == data_module.max_prediction_length


def test_collate_fn(data_module):
    """Test the collate function that combines batch samples.

    Ensures proper stacking of dictionary keys and batch outputs."""
    data_module.setup()

    batch_size = 3
    batch = [data_module.train_dataset[i] for i in range(batch_size)]

    x_batch, y_batch = data_module.collate_fn(batch)

    for key in x_batch:
        assert x_batch[key].shape[0] == batch_size

    metadata = data_module.time_series_metadata
    known_cat_count = len(
        [
            col
            for col in metadata["cols"]["x"]
            if metadata["col_type"].get(col) == "C"
            and metadata["col_known"].get(col) == "K"
        ]
    )

    known_cont_count = len(
        [
            col
            for col in metadata["cols"]["x"]
            if metadata["col_type"].get(col) == "F"
            and metadata["col_known"].get(col) == "K"
        ]
    )

    assert x_batch["decoder_cat"].shape[2] == known_cat_count
    assert x_batch["decoder_cont"].shape[2] == known_cont_count
    assert y_batch.shape[0] == batch_size
    assert y_batch.shape[1] == data_module.max_prediction_length


def test_full_dataloader_iteration(data_module):
    """Test a full iteration through the train dataloader.

    Confirms batch retrieval and tensor dimensions match configuration."""
    data_module.setup()
    train_loader = data_module.train_dataloader()

    batch = next(iter(train_loader))
    x_batch, y_batch = batch

    assert x_batch["encoder_cat"].shape[0] == data_module.batch_size
    assert x_batch["encoder_cat"].shape[1] == data_module.max_encoder_length

    metadata = data_module.time_series_metadata
    known_cat_count = len(
        [
            col
            for col in metadata["cols"]["x"]
            if metadata["col_type"].get(col) == "C"
            and metadata["col_known"].get(col) == "K"
        ]
    )

    known_cont_count = len(
        [
            col
            for col in metadata["cols"]["x"]
            if metadata["col_type"].get(col) == "F"
            and metadata["col_known"].get(col) == "K"
        ]
    )

    assert x_batch["decoder_cat"].shape[0] == data_module.batch_size
    assert x_batch["decoder_cat"].shape[2] == known_cat_count
    assert x_batch["decoder_cont"].shape[0] == data_module.batch_size
    assert x_batch["decoder_cont"].shape[2] == known_cont_count
    assert y_batch.shape[0] == data_module.batch_size
    assert y_batch.shape[1] == data_module.max_prediction_length


def test_variable_encoder_lengths(sample_timeseries_data):
    """Test with variable encoder lengths.

    Ensures random length behavior is respected and functional."""
    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=sample_timeseries_data,
        max_encoder_length=24,
        min_encoder_length=12,
        max_prediction_length=12,
        batch_size=4,
        randomize_length=True,
    )

    dm.setup()
    assert dm.min_encoder_length == 12
    assert dm.max_encoder_length == 24


def test_preprocess_data(data_module, sample_timeseries_data):
    """Test the _preprocess_data method.

    Checks preprocessing output structure and alignment with raw data."""
    if not hasattr(data_module, "_split_indices"):
        data_module.setup()

    series_idx = data_module._train_indices[0]

    processed = data_module._preprocess_data(series_idx)

    assert "features" in processed
    assert "categorical" in processed["features"]
    assert "continuous" in processed["features"]
    assert "target" in processed
    assert "time_mask" in processed

    original_sample = sample_timeseries_data[series_idx.item()]
    expected_length = len(original_sample["y"])

    assert processed["features"]["categorical"].shape[0] == expected_length
    assert processed["features"]["continuous"].shape[0] == expected_length
    assert processed["target"].shape[0] == expected_length


def test_with_static_features():
    """Test with static features included.

    Validates static feature support in both metadata and sample input."""
    df = pd.DataFrame(
        {
            "group": [0, 0, 0, 1, 1, 1],
            "time": pd.date_range("2020-01-01", periods=6),
            "target": [1, 2, 3, 4, 5, 6],
            "static_cat": [0, 0, 0, 1, 1, 1],
            "static_num": [10, 10, 10, 20, 20, 20],
            "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )

    ts = TimeSeries(
        data=df,
        time="time",
        target="target",
        group=["group"],
        num=["feature1", "static_num"],
        static=["static_cat", "static_num"],
        cat=["static_cat"],
    )

    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=ts,
        max_encoder_length=2,
        max_prediction_length=1,
        batch_size=2,
    )

    dm.setup()

    metadata = dm.metadata
    assert metadata["static_categorical_features"] == 1
    assert metadata["static_continuous_features"] == 1

    x, y = dm.train_dataset[0]
    assert "static_categorical_features" in x
    assert "static_continuous_features" in x
    assert (
        x["static_categorical_features"].shape[1]
        == metadata["static_categorical_features"]
    )
    assert (
        x["static_continuous_features"].shape[1]
        == metadata["static_continuous_features"]
    )


def test_different_train_val_test_split(sample_timeseries_data):
    """Test with different train/val/test split ratios."""
    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=sample_timeseries_data,
        max_encoder_length=24,
        max_prediction_length=12,
        batch_size=4,
        train_val_test_split=(0.8, 0.1, 0.1),
    )

    dm.setup()

    total_series = len(sample_timeseries_data)
    expected_train = int(0.8 * total_series)
    expected_val = int(0.1 * total_series)

    assert len(dm._train_indices) == expected_train
    assert len(dm._val_indices) == expected_val
    assert len(dm._test_indices) == total_series - expected_train - expected_val


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

    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=ts,
        max_encoder_length=10,
        max_prediction_length=5,
        batch_size=4,
    )

    dm.setup()

    x, y = dm.train_dataset[0]
    assert y.shape[-1] == 2
