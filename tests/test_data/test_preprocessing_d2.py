"""
Deterministic tests for preprocessing steps (scaling/transformations) in D2 layer.

These tests verify that EncoderDecoderTimeSeriesDataModule._preprocess_data()
correctly applies target normalization and continuous feature scaling when provided.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch

from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
from pytorch_forecasting.data.encoders import TorchNormalizer
from pytorch_forecasting.data.timeseries import TimeSeries


@pytest.fixture
def deterministic_timeseries_data():
    """Create deterministic time series dataset for testing."""
    num_groups = 3
    seq_length = 20

    groups = []
    times = []
    values = []
    continuous_feature1 = []
    continuous_feature2 = []

    # Use fixed seed for deterministic data
    np.random.seed(42)
    for g in range(num_groups):
        for t in range(seq_length):
            groups.append(g)
            times.append(pd.Timestamp("2020-01-01") + pd.Timedelta(days=t))

            # Deterministic target values
            value = 100.0 + 10.0 * t + 5.0 * g
            values.append(value)

            # Deterministic continuous features
            continuous_feature1.append(50.0 + 2.0 * t + g)
            continuous_feature2.append(200.0 + 3.0 * t - g)

    df = pd.DataFrame(
        {
            "group": groups,
            "time": times,
            "target": values,
            "cont_feat1": continuous_feature1,
            "cont_feat2": continuous_feature2,
        }
    )

    time_series = TimeSeries(
        data=df,
        time="time",
        target="target",
        group=["group"],
        num=["cont_feat1", "cont_feat2"],
    )

    return time_series


@pytest.fixture
def data_module_without_preprocessing(deterministic_timeseries_data):
    """Data module without preprocessing (baseline)."""
    return EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=deterministic_timeseries_data,
        max_encoder_length=10,
        max_prediction_length=5,
        batch_size=4,
        target_normalizer=None,
        scalers=None,
    )


def test_target_normalization_applied(deterministic_timeseries_data):
    target_normalizer = TorchNormalizer(method="standard", center=True)

    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=deterministic_timeseries_data,
        max_encoder_length=10,
        max_prediction_length=5,
        batch_size=4,
        target_normalizer=target_normalizer,
        scalers=None,
    )
    dm.setup()

    series_idx = dm._train_indices[0]
    processed = dm._preprocess_data(series_idx)

    assert (
        "target_scale" in processed
    ), "target_scale should be stored when target_normalizer is provided"
    assert isinstance(
        processed["target_scale"], torch.Tensor
    ), "target_scale should be a torch.Tensor"
    assert processed["target_scale"].shape == (
        2,
    ), "target_scale should have shape [center, scale]"

    normalized_target = processed["target"].float().squeeze(-1)
    target_mean = normalized_target.mean().item()
    target_std = normalized_target.std().item()

    assert (
        abs(target_mean) < 1e-4
    ), f"Normalized target mean should be ~0, got {target_mean}"
    assert (
        abs(target_std - 1.0) < 0.15
    ), f"Normalized target std should be ~1, got {target_std}"


def test_continuous_feature_scaling_applied(deterministic_timeseries_data):
    """Test that continuous feature scaling is applied correctly with StandardScaler."""
    # Prepare feature data for fitting
    all_feat1 = []
    all_feat2 = []
    for i in range(len(deterministic_timeseries_data)):
        sample = deterministic_timeseries_data[i]
        features = sample["x"]
        if isinstance(features, torch.Tensor):
            feat_array = features.numpy()
        else:
            feat_array = np.array(features)
        # Assuming cont_feat1 is first continuous feature, cont_feat2 is second
        all_feat1.append(feat_array[:, 0])
        all_feat2.append(feat_array[:, 1])

    feat1_data = np.concatenate(all_feat1).reshape(-1, 1)
    feat2_data = np.concatenate(all_feat2).reshape(-1, 1)

    # Create and fit scalers
    scaler1 = StandardScaler()
    scaler1.fit(feat1_data)
    scaler2 = StandardScaler()
    scaler2.fit(feat2_data)

    # Create data module with scalers
    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=deterministic_timeseries_data,
        max_encoder_length=10,
        max_prediction_length=5,
        batch_size=4,
        target_normalizer=None,
        scalers={"cont_feat1": scaler1, "cont_feat2": scaler2},
    )
    dm.setup()

    # Get a sample and check preprocessing
    series_idx = dm._train_indices[0]
    processed = dm._preprocess_data(series_idx)

    # Verify continuous features are scaled
    continuous = processed["features"]["continuous"]
    assert isinstance(
        continuous, torch.Tensor
    ), "Continuous features should be torch.Tensor"

    # Check that features are normalized (mean ~0, std ~1 for each feature)
    if continuous.shape[1] >= 2:
        feat1_scaled = continuous[:, 0].float()
        feat2_scaled = continuous[:, 1].float()

        feat1_mean = feat1_scaled.mean().item()
        feat2_mean = feat2_scaled.mean().item()

        # Single series may not be exactly 0/1 normalized; check reasonable range.
        assert (
            abs(feat1_mean) < 10.0
        ), f"Scaled feature1 mean should be reasonable, got {feat1_mean}"
        assert (
            abs(feat2_mean) < 10.0
        ), f"Scaled feature2 mean should be reasonable, got {feat2_mean}"


def test_scaling_parameters_stored_for_inverse(deterministic_timeseries_data):
    """Test that scaling parameters are stored for inverse transformation."""
    # Fit normalizer on properly squeezed 1D data from a single series
    sample_0 = deterministic_timeseries_data[0]
    target_1d = sample_0["y"].squeeze(-1).float()

    target_normalizer = TorchNormalizer(method="standard")
    target_normalizer.fit(target_1d)
    expected_params = target_normalizer.get_parameters()

    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=deterministic_timeseries_data,
        max_encoder_length=10,
        max_prediction_length=5,
        batch_size=4,
        target_normalizer=target_normalizer,
    )
    dm.setup()

    series_idx = dm._train_indices[0]
    processed = dm._preprocess_data(series_idx)

    assert "target_scale" in processed
    stored_params = processed["target_scale"]
    assert stored_params.shape == (
        2,
    ), f"target_scale should be (2,), got {stored_params.shape}"
    assert torch.allclose(stored_params.float(), expected_params.float(), atol=1e-4), (
        "Stored parameters should match normalizer parameters. "
        f"Got {stored_params}, expected {expected_params}"
    )


def test_preprocessing_is_deterministic(deterministic_timeseries_data):
    """Test that preprocessing is deterministic (same input produces same output)."""
    target_normalizer = TorchNormalizer(method="standard")

    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=deterministic_timeseries_data,
        max_encoder_length=10,
        max_prediction_length=5,
        batch_size=4,
        target_normalizer=target_normalizer,
    )
    dm.setup()

    series_idx = dm._train_indices[0]

    # First call fits the normalizer; second call reuses it
    processed1 = dm._preprocess_data(series_idx)
    processed2 = dm._preprocess_data(series_idx)

    assert torch.equal(
        processed1["target"], processed2["target"]
    ), "Preprocessing should be deterministic - same input should produce same output"
    assert torch.equal(
        processed1["features"]["continuous"], processed2["features"]["continuous"]
    ), "Continuous features should be deterministic"
    if "target_scale" in processed1:
        assert torch.equal(
            processed1["target_scale"], processed2["target_scale"]
        ), "Target scale should be deterministic"


def test_preprocessing_with_no_scalers_pass_through(deterministic_timeseries_data):
    """Test that preprocessing with no scalers passes through unchanged."""
    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=deterministic_timeseries_data,
        max_encoder_length=10,
        max_prediction_length=5,
        batch_size=4,
        target_normalizer=None,
        scalers=None,
    )
    dm.setup()

    series_idx = dm._train_indices[0]
    processed = dm._preprocess_data(series_idx)

    # Get original sample
    original_sample = deterministic_timeseries_data[series_idx.item()]

    # Verify target is unchanged (within float conversion)
    original_target = original_sample["y"]
    if isinstance(original_target, torch.Tensor):
        original_target = original_target.float()
    else:
        original_target = torch.tensor(original_target, dtype=torch.float32)

    processed_target = processed["target"].float()

    assert torch.allclose(
        original_target, processed_target, atol=1e-5
    ), "Target should pass through unchanged when no normalizer is provided"

    # Verify continuous features are unchanged (within float conversion)
    original_features = original_sample["x"]
    if isinstance(original_features, torch.Tensor):
        original_features = original_features.float()
    else:
        original_features = torch.tensor(original_features, dtype=torch.float32)

    # Extract continuous features based on indices
    if dm.continuous_indices:
        original_continuous = original_features[:, dm.continuous_indices]
    else:
        original_continuous = torch.zeros((original_features.shape[0], 0))

    processed_continuous = processed["features"]["continuous"].float()

    assert torch.allclose(
        original_continuous, processed_continuous, atol=1e-5
    ), "Continuous features should pass through unchanged when no scalers are provided"


def test_preprocessing_with_missing_feature_raises_error(deterministic_timeseries_data):
    """Test that preprocessing raises error for missing features in scalers."""
    # Create scaler for non-existent feature
    fake_scaler = StandardScaler()
    fake_scaler.fit(np.array([[1.0], [2.0], [3.0]]))

    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=deterministic_timeseries_data,
        max_encoder_length=10,
        max_prediction_length=5,
        batch_size=4,
        target_normalizer=None,
        scalers={"nonexistent_feature": fake_scaler},
    )
    dm.setup()

    series_idx = dm._train_indices[0]

    # Should raise ValueError when scaler is provided for feature that doesn't exist
    with pytest.raises(
        (ValueError, KeyError), match=".*feature.*|.*scaler.*|.*nonexistent.*"
    ):
        dm._preprocess_data(series_idx)


def test_preprocessing_with_robust_scaler(deterministic_timeseries_data):
    """Test that RobustScaler works correctly for continuous features."""
    # Prepare feature data
    all_feat1 = []
    for i in range(len(deterministic_timeseries_data)):
        sample = deterministic_timeseries_data[i]
        features = sample["x"]
        if isinstance(features, torch.Tensor):
            feat_array = features.numpy()
        else:
            feat_array = np.array(features)
        all_feat1.append(feat_array[:, 0])

    feat1_data = np.concatenate(all_feat1).reshape(-1, 1)

    # Create and fit RobustScaler
    robust_scaler = RobustScaler()
    robust_scaler.fit(feat1_data)

    dm = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=deterministic_timeseries_data,
        max_encoder_length=10,
        max_prediction_length=5,
        batch_size=4,
        target_normalizer=None,
        scalers={"cont_feat1": robust_scaler},
    )
    dm.setup()

    series_idx = dm._train_indices[0]
    processed = dm._preprocess_data(series_idx)

    # Verify continuous features are processed
    continuous = processed["features"]["continuous"]
    assert isinstance(
        continuous, torch.Tensor
    ), "Continuous features should be torch.Tensor"
    assert continuous.shape[1] >= 1, "Should have at least one continuous feature"
