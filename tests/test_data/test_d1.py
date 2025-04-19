import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data.timeseries import TimeSeries


@pytest.fixture
def sample_data():
    """Create time series data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "target_value": np.sin(np.arange(10)) + 10,
            "feature1": np.random.randn(10),
            "feature2": np.random.randn(10),
            "feature3": np.random.randn(10),
            "group_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "weight": np.abs(np.random.randn(10)) + 0.1,
            "static_feat": [10, 10, 10, 10, 10, 20, 20, 20, 20, 20],
        }
    )
    return data


@pytest.fixture
def future_data():
    """Create future time series data."""
    dates = pd.date_range(start="2023-01-11", periods=5, freq="D")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "feature1": np.random.randn(5),
            "feature2": np.random.randn(5),
            "feature3": np.random.randn(5),
            "group_id": [1, 1, 1, 2, 2],
            "weight": np.abs(np.random.randn(5)) + 0.1,
            "static_feat": [10, 10, 10, 20, 20],
        }
    )
    return data


def test_init_basic(sample_data):
    """Test basic initialization of TimeSeries class.

    Ensures that the class stores time, target, and correctly detects feature columns
    when no group, known/unknown features, or static/weight features are specified."""
    ts = TimeSeries(data=sample_data, time="timestamp", target="target_value")

    assert ts.time == "timestamp"
    assert ts.target == ["target_value"]
    assert len(ts.feature_cols) == 6  # All columns except timestamp, target_value
    assert len(ts) == 1  # Single group by default


def test_init_with_groups(sample_data):
    """Test initialization with group parameter.

    Verifies that data is grouped correctly and each group is handled as a
    separate time series.
    """
    ts = TimeSeries(
        data=sample_data, time="timestamp", target="target_value", group=["group_id"]
    )

    assert ts.group == ["group_id"]
    assert len(ts) == 2  # Two groups (1 and 2)
    assert set(ts._group_ids) == {1, 2}


def test_init_with_features_categorization(sample_data):
    """Test feature categorization.

    Ensures that numeric, categorical, and static features are categorized and
    stored correctly in metadata."""
    ts = TimeSeries(
        data=sample_data,
        time="timestamp",
        target="target_value",
        num=["feature1", "feature2", "feature3"],
        cat=[],
        static=["static_feat"],
    )

    assert ts.num == ["feature1", "feature2", "feature3"]
    assert ts.cat == []
    assert ts.static == ["static_feat"]
    assert ts.metadata["col_type"]["feature1"] == "F"
    assert ts.metadata["col_type"]["feature2"] == "F"


def test_init_with_known_unknown(sample_data):
    """Test known and unknown features classification.

    Checks if the known and unknown feature categorization is correctly set
    and stored in metadata."""
    ts = TimeSeries(
        data=sample_data,
        time="timestamp",
        target="target_value",
        known=["feature1"],
        unknown=["feature2", "feature3"],
    )

    assert ts.known == ["feature1"]
    assert ts.unknown == ["feature2", "feature3"]
    assert ts.metadata["col_known"]["feature1"] == "K"
    assert ts.metadata["col_known"]["feature2"] == "U"


def test_init_with_weight(sample_data):
    """Test initialization with weight parameter.

    Verifies that the weight column is stored correctly and excluded
    from the feature columns."""
    ts = TimeSeries(
        data=sample_data, time="timestamp", target="target_value", weight="weight"
    )

    assert ts.weight == "weight"
    assert "weight" not in ts.feature_cols


def test_getitem_basic(sample_data):
    """Test __getitem__ with basic configuration.

    Checks the output structure of a single time series without grouping,
    ensuring x, y are tensors of correct shapes."""
    ts = TimeSeries(data=sample_data, time="timestamp", target="target_value")

    result = ts[0]
    assert torch.is_tensor(result["y"])
    assert torch.is_tensor(result["x"])
    assert "t" in result
    assert "cutoff_time" in result
    assert len(result["y"]) == 10  # 10 data points
    assert result["y"].shape == (10, 1)  # One target variable
    assert result["x"].shape[1] == 6  # Six feature columns


def test_getitem_with_groups(sample_data):
    """Test __getitem__ with groups parameter.

    Verifies the per-group access using index and checks that each group
    has the correct number of time steps."""
    ts = TimeSeries(
        data=sample_data, time="timestamp", target="target_value", group=["group_id"]
    )

    # group (1)
    result_g1 = ts[0]
    assert len(result_g1["t"]) == 5  # 5 data points in group 1

    # group (2)
    result_g2 = ts[1]
    assert len(result_g2["t"]) == 5  # 5 data points in group 2


def test_getitem_with_static(sample_data):
    """Test __getitem__ with static features.

    Ensures static features are included in the output and correctly
    mapped per group."""
    ts = TimeSeries(
        data=sample_data,
        time="timestamp",
        target="target_value",
        group=["group_id"],
        static=["static_feat"],
    )

    result_g1 = ts[0]
    result_g2 = ts[1]

    assert torch.is_tensor(result_g1["st"])
    assert result_g1["st"].item() == 10  # Static feature for group 1
    assert result_g2["st"].item() == 20  # Static feature for group 2


def test_getitem_with_weight(sample_data):
    """Test __getitem__ with weight parameter.

    Validates that weights are correctly returned in the output and have the
    expected length and type."""
    ts = TimeSeries(
        data=sample_data, time="timestamp", target="target_value", weight="weight"
    )

    result = ts[0]
    assert "weights" in result
    assert torch.is_tensor(result["weights"])
    assert len(result["weights"]) == 10


def test_with_future_data(sample_data, future_data):
    """Test with future data provided.

    Verifies that future time steps are appended to the end of each group,
    especially for known features."""
    ts = TimeSeries(
        data=sample_data,
        data_future=future_data,
        time="timestamp",
        target="target_value",
        group=["group_id"],
        known=["feature1"],
    )

    result_g1 = ts[0]  # Group 1

    assert len(result_g1["t"]) == 8  # 5 original + 3 future for group 1

    feature1_idx = ts.feature_cols.index("feature1")
    assert not torch.isnan(
        result_g1["x"][-1, feature1_idx]
    )  # feature1 is not NaN in last row


def test_future_data_with_weights(sample_data, future_data):
    """Test handling of weights with future data.

    Ensures that weights from future data are combined properly and match the
    time indices."""
    ts = TimeSeries(
        data=sample_data,
        data_future=future_data,
        time="timestamp",
        target="target_value",
        group=["group_id"],
        weight="weight",
    )

    result = ts[0]  # Group 1
    assert "weights" in result
    assert torch.is_tensor(result["weights"])
    assert len(result["weights"]) == len(result["t"])


def test_future_data_missing_columns(sample_data):
    """Test handling when future data is missing some columns.

    Verifies the handling of missing feature columns in future data by
    checking NaN padding."""
    dates = pd.date_range(start="2023-01-11", periods=5, freq="D")
    incomplete_future = pd.DataFrame(
        {
            "timestamp": dates,
            "feature1": np.random.randn(5),
            # Missing feature2, feature3
            "group_id": [1, 1, 1, 2, 2],
            "weight": np.abs(np.random.randn(5)) + 0.1,
        }
    )

    ts = TimeSeries(
        data=sample_data,
        data_future=incomplete_future,
        time="timestamp",
        target="target_value",
        group=["group_id"],
        known=["feature1"],
    )

    result = ts[0]
    # Check that missing features are NaN in future timepoints
    future_indices = np.where(result["t"] >= np.datetime64("2023-01-11"))[0]
    feature2_idx = ts.feature_cols.index("feature2")
    feature3_idx = ts.feature_cols.index("feature3")
    assert torch.isnan(result["x"][future_indices[0], feature2_idx])
    assert torch.isnan(result["x"][future_indices[0], feature3_idx])


def test_different_future_groups(sample_data):
    """Test with future data that has different groups than original data.

    Ensures that groups present only in future data are ignored if not
    in the original dataset."""
    dates = pd.date_range(start="2023-01-11", periods=5, freq="D")
    future_with_new_group = pd.DataFrame(
        {
            "timestamp": dates,
            "feature1": np.random.randn(5),
            "feature2": np.random.randn(5),
            "feature3": np.random.randn(5),
            "group_id": [1, 1, 3, 3, 3],  # Group 3 is new
            "weight": np.abs(np.random.randn(5)) + 0.1,
            "static_feat": [10, 10, 30, 30, 30],
        }
    )

    ts = TimeSeries(
        data=sample_data,
        data_future=future_with_new_group,
        time="timestamp",
        target="target_value",
        group=["group_id"],
    )

    # Original data has groups 1 and 2, but not 3
    assert len(ts) == 2
    assert 3 not in ts._group_ids


def test_multiple_targets(sample_data):
    """Test handling of multiple target variables.

    Verifies that multiple target columns are handled and returned
    as the correct shape in the output."""
    sample_data["target_value2"] = np.cos(np.arange(10)) + 5

    ts = TimeSeries(
        data=sample_data, time="timestamp", target=["target_value", "target_value2"]
    )

    result = ts[0]
    assert result["y"].shape == (10, 2)  # Two target variables


def test_empty_groups():
    """Test handling of empty groups.

    Confirms that the class handles datasets with a single group and
    no empty group errors occur."""
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="D"),
            "target_value": np.random.randn(5),
            "group_id": [1, 1, 1, 1, 1],  # Only one group
        }
    )

    ts = TimeSeries(
        data=data, time="timestamp", target="target_value", group=["group_id"]
    )

    assert len(ts) == 1  # Only one group


def test_metadata_structure(sample_data):
    """Test the structure of metadata.

    Ensures the metadata dictionary includes the expected keys and
    correct mappings of feature roles."""
    ts = TimeSeries(
        data=sample_data,
        time="timestamp",
        target="target_value",
        num=["feature1", "feature2", "feature3"],
        cat=[],  # No categorical features
        static=["static_feat"],
        known=["feature1"],
        unknown=["feature2", "feature3"],
    )

    metadata = ts.get_metadata()

    assert "cols" in metadata
    assert "col_type" in metadata
    assert "col_known" in metadata

    assert metadata["cols"]["y"] == ["target_value"]
    assert set(metadata["cols"]["x"]) == {
        "feature1",
        "feature2",
        "feature3",
        "group_id",
        "weight",
        "static_feat",
    }
    assert metadata["cols"]["st"] == ["static_feat"]

    assert metadata["col_type"]["feature1"] == "F"
    assert metadata["col_type"]["feature2"] == "F"

    assert metadata["col_known"]["feature1"] == "K"
    assert metadata["col_known"]["feature2"] == "U"
