import numpy as np
import pandas as pd
import pytest

from pytorch_forecasting.data._encoders_v2 import D1CategoricalEncoder


@pytest.fixture
def sample_data():
    """Provides a fresh, abstract dataframe for each test."""
    return pd.DataFrame(
        {
            "cat1": ["a", "b", "c", "a", np.nan],
            "cat2": ["x", "y", "z", "y", "x"],
            "num1": [1.1, 2.2, 3.3, 4.4, 5.5],
        }
    )


def test_encoder_fit_transform(sample_data):
    """Validates encoding of categorical columns and preservation of numeric columns."""
    encoder = D1CategoricalEncoder(columns=["cat1", "cat2"])
    encoded_df = encoder.fit(sample_data).transform(sample_data)

    assert pd.api.types.is_integer_dtype(encoded_df["cat1"])
    assert pd.api.types.is_integer_dtype(encoded_df["cat2"])

    assert encoded_df["num1"].equals(sample_data["num1"])

    assert not encoded_df["cat1"].isna().any()


def test_encoder_inverse_transform(sample_data):
    """Ensures inverse transformation restores original values including NaNs."""
    encoder = D1CategoricalEncoder(columns=["cat1"])
    encoded_df = encoder.fit(sample_data).transform(sample_data)

    decoded_df = encoder.inverse_transform(encoded_df)

    pd.testing.assert_series_equal(decoded_df["cat1"], sample_data["cat1"])


def test_unseen_variables_warning(sample_data):
    """Checks unseen category handling and warning behavior."""
    encoder = D1CategoricalEncoder(columns=["cat1"], handle_unknown="assign_new")
    encoder.fit(sample_data)

    new_data = pd.DataFrame({"cat1": ["q", "a"]})

    with pytest.warns(UserWarning, match="Unseen categories found in column 'cat1'"):
        encoded_new = encoder.transform(new_data)

    assert encoded_new.loc[0, "cat1"] == 0


def test_only_categorical_columns_selected(sample_data):
    """Ensures only categorical columns are encoded when columns=None."""
    encoder = D1CategoricalEncoder()
    encoder.fit(sample_data)

    assert "num1" not in encoder.mapping_
