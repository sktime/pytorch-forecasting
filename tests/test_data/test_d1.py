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
    """Basic init stores time/target and detects feature columns correctly."""
    ts = TimeSeries(data=sample_data, time="timestamp", target="target_value")

    assert ts.time == "timestamp"
    assert ts.target == ["target_value"]
    assert len(ts.feature_cols) == 6  # All columns except timestamp, target_value
    assert len(ts) == 1  # Single group by default


def test_init_with_groups(sample_data):
    """Groups are stored and each group is treated as a separate time series."""
    ts = TimeSeries(
        data=sample_data, time="timestamp", target="target_value", group=["group_id"]
    )

    assert ts.group == ["group_id"]
    assert len(ts) == 2  # Two groups (1 and 2)
    assert set(ts._group_ids) == {1, 2}


def test_init_with_features_categorization(sample_data):
    """Numeric, categorical, and static features are categorized correctly."""
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
    """Known and unknown feature classification is stored correctly in metadata."""
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
    """Weight column is stored and excluded from feature columns."""
    ts = TimeSeries(
        data=sample_data, time="timestamp", target="target_value", weight="weight"
    )

    assert ts.weight == "weight"
    assert "weight" not in ts.feature_cols


def test_getitem_basic(sample_data):
    """__getitem__ returns x and y tensors with correct shapes."""
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
    """Each group index returns only that group's time steps."""
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
    """Static features are returned per group with correct values."""
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
    """Weights are returned as a tensor with the correct length."""
    ts = TimeSeries(
        data=sample_data, time="timestamp", target="target_value", weight="weight"
    )

    result = ts[0]
    assert "weights" in result
    assert torch.is_tensor(result["weights"])
    assert len(result["weights"]) == 10


def test_with_future_data(sample_data, future_data):
    """Future time steps are appended and known features are filled in."""
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
    """Weights from future data are merged and aligned with time indices."""
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
    """Missing columns in future data are NaN-padded in the output tensor."""
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
    """Groups only in future data are ignored; dataset length stays the same."""
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
    """Multiple target columns produce a y tensor with the right shape."""
    sample_data["target_value2"] = np.cos(np.arange(10)) + 5

    ts = TimeSeries(
        data=sample_data, time="timestamp", target=["target_value", "target_value2"]
    )

    result = ts[0]
    assert result["y"].shape == (10, 2)  # Two target variables


def test_empty_groups():
    """Dataset with a single group works without errors."""
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
    """Metadata has the right keys and correct col_type/col_known mappings."""
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


# --- label encoder tests ---


@pytest.fixture
def cat_data():
    """Create time series data with string categorical features."""
    data = pd.DataFrame(
        {
            "time": list(range(10)) * 2,
            "group": ["A"] * 10 + ["B"] * 10,
            "target": np.random.randn(20),
            "cat_color": ["red", "blue", "green", "red", "blue"] * 4,
            "cat_size": ["small", "medium", "large", "small", "medium"] * 4,
            "num_feat": np.random.randn(20),
        }
    )
    return data


def test_label_encoders_fitted_on_init(cat_data):
    """Label encoders should be auto-fitted for all cat columns during __init__."""
    ts = TimeSeries(
        data=cat_data,
        time="time",
        target="target",
        group=["group"],
        cat=["cat_color", "cat_size"],
        num=["num_feat"],
    )

    assert hasattr(ts, "label_encoders")
    assert "cat_color" in ts.label_encoders
    assert "cat_size" in ts.label_encoders
    assert hasattr(ts.label_encoders["cat_color"], "classes_")
    assert hasattr(ts.label_encoders["cat_size"], "classes_")


def test_label_encoders_cover_all_cat_values(cat_data):
    """Encoder classes_ should contain all unique values from the column."""
    ts = TimeSeries(
        data=cat_data,
        time="time",
        target="target",
        group=["group"],
        cat=["cat_color"],
        num=["num_feat"],
    )

    enc = ts.label_encoders["cat_color"]
    for val in cat_data["cat_color"].unique():
        assert val in enc.classes_, f"'{val}' missing from encoder classes_"


def test_getitem_cat_columns_are_numeric(cat_data):
    """After encoding, x tensor must be fully numeric (no strings)."""
    ts = TimeSeries(
        data=cat_data,
        time="time",
        target="target",
        group=["group"],
        cat=["cat_color", "cat_size"],
        num=["num_feat"],
    )

    result = ts[0]
    assert torch.is_tensor(result["x"]), "x should be a tensor"
    assert result["x"].dtype in (
        torch.float32,
        torch.float64,
    ), "x should be float after encoding"
    # no NaNs from failed conversion
    assert not torch.isnan(result["x"]).all()


def test_getitem_encoded_values_are_integers(cat_data):
    """Encoded categorical columns should contain integer-like values."""
    ts = TimeSeries(
        data=cat_data,
        time="time",
        target="target",
        group=["group"],
        cat=["cat_color"],
        num=["num_feat"],
    )

    cat_col_idx = ts.feature_cols.index("cat_color")
    result = ts[0]
    encoded_vals = result["x"][:, cat_col_idx]
    # all values should be whole numbers (integers stored as float)
    assert torch.all(encoded_vals == encoded_vals.floor()), (
        "Encoded categorical values should be integer-valued"
    )


def test_no_label_encoders_for_numeric_only(sample_data):
    """No label encoders should be created when there are no cat columns."""
    ts = TimeSeries(
        data=sample_data,
        time="timestamp",
        target="target_value",
        num=["feature1", "feature2", "feature3"],
        cat=[],
    )

    assert ts.label_encoders == {}


def test_custom_label_encoder_is_used(cat_data):
    """A pre-fitted encoder passed via label_encoders should be used as-is."""
    from pytorch_forecasting.data.encoders import NaNLabelEncoder

    custom_enc = NaNLabelEncoder(add_nan=False)
    custom_enc.fit(cat_data["cat_color"])

    ts = TimeSeries(
        data=cat_data,
        time="time",
        target="target",
        group=["group"],
        cat=["cat_color"],
        num=["num_feat"],
        label_encoders={"cat_color": custom_enc},
    )

    assert ts.label_encoders["cat_color"] is custom_enc


def test_label_encoder_inverse_transform(cat_data):
    """Inverse transform should recover original category values."""
    import numpy as np

    ts = TimeSeries(
        data=cat_data,
        time="time",
        target="target",
        group=["group"],
        cat=["cat_color"],
        num=["num_feat"],
    )

    enc = ts.label_encoders["cat_color"]
    original = cat_data["cat_color"].iloc[:5].values
    encoded = enc.transform(original)
    decoded = enc.inverse_transform(encoded)
    np.testing.assert_array_equal(decoded, original)
