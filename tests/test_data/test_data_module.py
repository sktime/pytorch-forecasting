import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from pytorch_forecasting.data.data_modules import EncoderDecoderTimeSeriesDataModule
from pytorch_forecasting.data.timeseries import TimeSeries


@pytest.fixture
def sample_timeseries_data():
    """Generate a sample time series dataset for testing."""
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
    n_series = 100

    data = []
    for i in range(n_series):
        group_id = i
        static_feat = i % 2

        series = pd.DataFrame(
            {
                "time": (dates - dates[0]).days,
                "group_id": group_id,
                "category_1": np.random.randint(0, 3, len(dates), dtype=np.int32),
                "category_2": np.random.randint(0, 5, len(dates), dtype=np.int32),
                "value_1": np.random.randn(len(dates)).astype(np.float32),
                "value_2": np.random.randn(len(dates)).astype(np.float32),
                "known_future_1": np.random.randn(len(dates)).astype(np.float32),
                "known_future_2": np.random.randint(0, 3, len(dates), dtype=np.int32),
                "unknown_future_1": np.random.randn(len(dates)).astype(np.float32),
                "target": np.sin(np.linspace(0, 8 * np.pi, len(dates))).astype(
                    np.float32
                )
                + np.random.randn(len(dates)).astype(np.float32) * 0.1,
                "static_feat": np.full(len(dates), static_feat, dtype=np.int32),
            }
        )
        data.append(series)

    df = pd.concat(data, ignore_index=True)

    df = df.astype(
        {
            "time": np.int32,
            "group_id": np.int32,
            "category_1": np.int32,
            "category_2": np.int32,
            "value_1": np.float32,
            "value_2": np.float32,
            "known_future_1": np.float32,
            "known_future_2": np.int32,
            "unknown_future_1": np.float32,
            "target": np.float32,
            "static_feat": np.int32,
        }
    )

    future_dates = pd.date_range(start="2023-02-20", periods=20, freq="D")
    future_data = []
    for i in range(n_series):
        group_id = i
        future_series = pd.DataFrame(
            {
                "time": (future_dates - dates[0]).days,
                "group_id": group_id,
                "known_future_1": np.random.randn(len(future_dates)).astype(np.float32),
                "known_future_2": np.random.randint(
                    0, 3, len(future_dates), dtype=np.int32
                ),
            }
        )
        future_data.append(future_series)

    future_df = pd.concat(future_data, ignore_index=True)

    future_df = future_df.astype(
        {
            "time": np.int32,
            "group_id": np.int32,
            "known_future_1": np.float32,
            "known_future_2": np.int32,
        }
    )

    ts = TimeSeries(
        data=df,
        data_future=future_df,
        time="time",
        target="target",
        group=["group_id"],
        static=["static_feat"],
        cat=["category_1", "category_2", "known_future_2"],
        num=["value_1", "value_2", "known_future_1", "unknown_future_1"],
        known=["known_future_1", "known_future_2"],
        unknown=["unknown_future_1"],
    )

    return ts


def test_known_unknown_features(sample_timeseries_data):
    """Test handling of known and unknown future features.

    This test checks:

    - Whether metadata correctly identifies known and unknown future features.
    - Whether future data is correctly included in the dataset.
    - The structure and presence of known future feature tensors in a sample.
    """
    datamodule = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=sample_timeseries_data,
        max_encoder_length=20,
        max_prediction_length=5,
    )

    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    x_batch, _ = batch

    # Verify metadata contains known/unknown information
    metadata = sample_timeseries_data.get_metadata()
    assert "col_known" in metadata
    assert metadata["col_known"]["known_future_1"] == "K"
    assert metadata["col_known"]["known_future_2"] == "K"
    assert metadata["col_known"]["unknown_future_1"] == "U"

    # Verify future data handling
    sample = sample_timeseries_data[0]
    assert "x_f" in sample
    assert sample["x_f"].shape[1] == 2  # known_future_1 and known_future_2


def test_initialization(sample_timeseries_data):
    """Test the initialization of the EncoderDecoderTimeSeriesDataModule.

    This test verifies:

    - The correct assignment of encoder and prediction lengths.
    - The default batch size is set correctly.
    - Categorical and continuous features are correctly identified.
    - Metadata correctly maps categorical and continuous features.
    """
    datamodule = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=sample_timeseries_data,
        max_encoder_length=30,
        max_prediction_length=10,
    )

    assert datamodule.max_encoder_length == 30
    assert datamodule.max_prediction_length == 10
    assert datamodule.batch_size == 32

    # Check correct identification of categorical and continuous features
    assert len(datamodule.categorical_indices) == 3  # category_1, category_2
    assert len(datamodule.continuous_indices) == 5  # value_1, value_2, static_feat

    # Verify the actual indices are correct
    metadata = sample_timeseries_data.get_metadata()
    feature_cols = metadata["cols"]["x"]

    # Verify categorical indices point to the right columns
    for idx in datamodule.categorical_indices:
        assert metadata["col_type"][feature_cols[idx]] == "C"

    # Verify continuous indices point to the right columns
    for idx in datamodule.continuous_indices:
        assert metadata["col_type"][feature_cols[idx]] == "F"


def test_setup_train_val_split(sample_timeseries_data):
    """Test dataset splitting into train and validation sets.

    This test ensures:

    - The `setup` method properly splits the dataset.
    - The train and validation datasets are correctly created.
    - The size of the train dataset matches expectations based on the split ratio.
    """
    datamodule = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=sample_timeseries_data,
        max_encoder_length=20,
        max_prediction_length=5,
        train_val_test_split=(0.7, 0.15, 0.15),
    )

    datamodule.setup(stage="fit")

    # Verify dataset creation
    assert hasattr(datamodule, "train_dataset")
    assert hasattr(datamodule, "val_dataset")

    # Check split sizes
    expected_train_size = int(0.7 * len(sample_timeseries_data))
    assert len(datamodule._train_indices) == expected_train_size


def test_data_loading(sample_timeseries_data):
    """Test data loading and batch structure.

    This test checks:

    - The train dataloader is correctly instantiated.
    - The batch contains all necessary components.
    - The categorical and continuous features have the correct dimensions.
    - The target tensor has the expected shape.
    """
    datamodule = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=sample_timeseries_data,
        max_encoder_length=20,
        max_prediction_length=5,
        batch_size=16,
    )

    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()

    # Verify DataLoader
    assert isinstance(train_loader, DataLoader)
    assert train_loader.batch_size == 16

    # Check batch structure
    batch = next(iter(train_loader))
    x_batch, y_batch = batch

    # Verify all required components are present
    expected_keys = {
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
    }
    assert all(key in x_batch for key in expected_keys)

    # Check shapes
    batch_size = 16
    assert x_batch["encoder_cat"].shape == (
        batch_size,
        20,
        3,
    )  # (batch, time, n_cat_features)
    assert x_batch["encoder_cont"].shape == (
        batch_size,
        20,
        5,
    )  # (batch, time, n_cont_features)
    assert x_batch["decoder_cat"].shape == (
        batch_size,
        5,
        3,
    )  # (batch, pred_length, n_cat_features)
    assert x_batch["decoder_cont"].shape == (
        batch_size,
        5,
        5,
    )  # (batch, pred_length, n_cont_features)
    assert y_batch.shape == (batch_size, 5, 1)  # (batch, pred_length, n_targets)


def test_different_settings(sample_timeseries_data):
    """Test different configuration settings.

    This test verifies:

    - The model handles different encoder and prediction lengths correctly.
    - Relative time indices and target scales are properly included.
    """
    datamodule = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=sample_timeseries_data,
        max_encoder_length=15,
        min_encoder_length=10,
        max_prediction_length=3,
        min_prediction_length=2,
        batch_size=8,
        add_relative_time_idx=True,
        add_target_scales=True,
    )

    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    x_batch, y_batch = batch

    assert x_batch["encoder_cat"].shape[1] == 15  # max_encoder_length
    assert x_batch["decoder_cat"].shape[1] == 3  # max_prediction_length
    assert x_batch["encoder_time_idx"].shape[1] == 15
    assert "target_scale" in x_batch  # verify target scales are included


def test_static_features(sample_timeseries_data):
    """Test that static features are correctly included.

    This test ensures:

    - Static categorical features are present in the batch.
    - Static feature tensor dimensions are as expected.
    """
    datamodule = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=sample_timeseries_data,
        max_encoder_length=20,
        max_prediction_length=5,
    )

    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    x_batch, _ = batch

    # Verify static features are present
    assert "static_categorical_features" in x_batch
    assert (
        x_batch["static_categorical_features"].dim() == 3
    )  # (batch_size, 1, n_static_features)


def test_group_handling(sample_timeseries_data):
    """Test that group information is correctly processed.

    This test verifies:

    - The presence of group identifiers in the batch.
    - Group tensor dimensions are as expected.
    """
    datamodule = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=sample_timeseries_data,
        max_encoder_length=20,
        max_prediction_length=5,
    )

    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    x_batch, _ = batch

    # Verify group information
    assert "groups" in x_batch
    assert x_batch["groups"].dim() == 2  # (batch_size, 1)


def test_window_creation(sample_timeseries_data):
    """Test window creation for encoder-decoder time series.

    This test ensures:

    - Windows are correctly generated for each time series.
    - Encoder and decoder window sizes match the expected values.
    - Window indices reference valid series in the dataset.
    """
    datamodule = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=sample_timeseries_data,
        max_encoder_length=20,
        max_prediction_length=5,
    )

    datamodule.setup(stage="fit")

    # Check that windows are created for each series in training set
    processed_data = datamodule.train_processed
    windows = datamodule.train_windows

    # Verify window parameters
    for window in windows:
        series_idx, start_idx, enc_length, pred_length = window
        assert enc_length == 20  # max_encoder_length
        assert pred_length == 5  # max_prediction_length
        assert series_idx < len(processed_data)


def test_prediction_mode(sample_timeseries_data):
    """Test the behavior of the datamodule in prediction mode.

    This test checks:

    - Whether the prediction dataset is properly created.
    - The structure of the prediction batch.
    - The presence of target scale information.
    """
    datamodule = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=sample_timeseries_data,
        max_encoder_length=20,
        max_prediction_length=5,
    )

    datamodule.setup(stage="predict")
    predict_loader = datamodule.predict_dataloader()

    # Check prediction dataset
    assert hasattr(datamodule, "predict_dataset")

    # Verify prediction batch structure
    batch = next(iter(predict_loader))
    x_batch, y_batch = batch

    assert x_batch["encoder_cat"].shape[1] == 20
    assert x_batch["decoder_cat"].shape[1] == 5
    assert "target_scale" in x_batch


@pytest.mark.parametrize(
    "train_val_test_split", [(0.6, 0.2, 0.2), (0.8, 0.1, 0.1), (0.7, 0.15, 0.15)]
)
def test_different_splits(sample_timeseries_data, train_val_test_split):
    """Test different train-validation-test splits.

    This test verifies:

    - The dataset is correctly split according to different ratios.
    - The sizes of train, validation, and test sets match expected values.
    """
    datamodule = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=sample_timeseries_data,
        max_encoder_length=20,
        max_prediction_length=5,
        train_val_test_split=train_val_test_split,
    )

    datamodule.setup(stage="fit")
    total_size = len(sample_timeseries_data)
    expected_train_size = int(train_val_test_split[0] * total_size)
    expected_val_size = int(train_val_test_split[1] * total_size)
    expected_test_size = int(train_val_test_split[2] * total_size)

    assert len(datamodule._train_indices) == expected_train_size
    assert len(datamodule._val_indices) == expected_val_size
    assert len(datamodule._test_indices) == expected_test_size
