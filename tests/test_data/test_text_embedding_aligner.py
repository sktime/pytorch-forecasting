"""Comprehensive tests for :class:`TextEmbeddingAligner`.

Tests cover:
- Correct concatenation when both numerical and text data exist at a timestamp.
- Correct zero-padding when text data is missing for a timestamp.
- Correct handling of different ``embedding_dim`` sizes.
- Edge cases: all timestamps present, all timestamps missing, not-fitted error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pytorch_forecasting.data.text_embedding_aligner import TextEmbeddingAligner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hourly_index() -> pd.DatetimeIndex:
    """Four hourly timestamps starting at 2024-01-01 00:00."""
    return pd.date_range("2024-01-01", periods=4, freq="h")


@pytest.fixture()
def numerical_df(hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Simple 2-feature numerical DataFrame on an hourly index."""
    return pd.DataFrame(
        {"feat_a": [1.0, 2.0, 3.0, 4.0], "feat_b": [10.0, 20.0, 30.0, 40.0]},
        index=hourly_index,
    )


# ---------------------------------------------------------------------------
# Core alignment tests
# ---------------------------------------------------------------------------


class TestCorrectConcatenation:
    """When both numerical and text data exist at a timestamp."""

    def test_matching_timestamps_are_concatenated(
        self, numerical_df: pd.DataFrame, hourly_index: pd.DatetimeIndex
    ) -> None:
        """Rows with a matching timestamp should contain the real embedding."""
        embeddings = {
            hourly_index[0]: np.array([0.1, 0.2, 0.3]),
            hourly_index[2]: np.array([0.4, 0.5, 0.6]),
        }

        aligner = TextEmbeddingAligner()
        aligner.fit(text_embeddings_dict=embeddings)
        result = aligner.transform(numerical_df)

        # Row 0: numerical [1, 10] + embedding [0.1, 0.2, 0.3]
        expected_row_0 = np.array([1.0, 10.0, 0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(result[0], expected_row_0)

        # Row 2: numerical [3, 30] + embedding [0.4, 0.5, 0.6]
        expected_row_2 = np.array([3.0, 30.0, 0.4, 0.5, 0.6])
        np.testing.assert_array_almost_equal(result[2], expected_row_2)


class TestZeroPaddingMissing:
    """When text data is missing for a timestamp, embed with zeros."""

    def test_missing_timestamps_are_zero_padded(
        self, numerical_df: pd.DataFrame, hourly_index: pd.DatetimeIndex
    ) -> None:
        """Rows WITHOUT a matching timestamp should have a zero-vector."""
        embedding_dim = 3
        embeddings = {
            hourly_index[0]: np.array([0.1, 0.2, 0.3]),
        }

        aligner = TextEmbeddingAligner()
        aligner.fit(text_embeddings_dict=embeddings)
        result = aligner.transform(numerical_df)

        # Rows 1, 2, 3 have no embedding → zeros appended.
        for row_idx in [1, 2, 3]:
            np.testing.assert_array_equal(
                result[row_idx, 2:],
                np.zeros(embedding_dim),
            )

    def test_all_timestamps_missing(
        self, numerical_df: pd.DataFrame, hourly_index: pd.DatetimeIndex
    ) -> None:
        """If NO timestamp matches, every embedding slot should be zeros."""
        embedding_dim = 4
        # Use a timestamp that is NOT in the hourly index.
        unrelated_ts = pd.Timestamp("1999-12-31")
        embeddings = {unrelated_ts: np.ones(embedding_dim)}

        aligner = TextEmbeddingAligner()
        aligner.fit(text_embeddings_dict=embeddings)
        result = aligner.transform(numerical_df)

        # All embedding columns should be zero.
        np.testing.assert_array_equal(
            result[:, 2:],
            np.zeros((len(numerical_df), embedding_dim)),
        )


class TestAllTimestampsPresent:
    """When every numerical timestamp has a corresponding embedding."""

    def test_no_zero_padding_needed(
        self, numerical_df: pd.DataFrame, hourly_index: pd.DatetimeIndex
    ) -> None:
        embedding_dim = 2
        embeddings = {
            ts: np.full(embedding_dim, i + 1.0) for i, ts in enumerate(hourly_index)
        }

        aligner = TextEmbeddingAligner()
        aligner.fit(text_embeddings_dict=embeddings)
        result = aligner.transform(numerical_df)

        for i in range(len(numerical_df)):
            np.testing.assert_array_almost_equal(
                result[i, 2:],
                np.full(embedding_dim, i + 1.0),
            )


# ---------------------------------------------------------------------------
# Different embedding dimensions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("embedding_dim", [8, 64, 128, 768])
def test_different_embedding_dims(
    numerical_df: pd.DataFrame,
    hourly_index: pd.DatetimeIndex,
    embedding_dim: int,
) -> None:
    """Output shape should adapt correctly to varying embedding dimensions."""
    embeddings = {hourly_index[0]: np.random.randn(embedding_dim)}

    aligner = TextEmbeddingAligner()
    aligner.fit(text_embeddings_dict=embeddings)
    result = aligner.transform(numerical_df)

    n_numerical_features = numerical_df.shape[1]
    assert result.shape == (len(numerical_df), n_numerical_features + embedding_dim)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


def test_output_shape(
    numerical_df: pd.DataFrame, hourly_index: pd.DatetimeIndex
) -> None:
    """General output shape contract: (n_rows, n_features + embedding_dim)."""
    embedding_dim = 5
    embeddings = {hourly_index[1]: np.ones(embedding_dim)}

    aligner = TextEmbeddingAligner()
    aligner.fit(text_embeddings_dict=embeddings)
    result = aligner.transform(numerical_df)

    assert result.shape == (4, 2 + embedding_dim)
    assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Guard-rail tests for bad usage patterns."""

    def test_not_fitted_raises(self, numerical_df: pd.DataFrame) -> None:
        """Calling transform() before fit() must raise RuntimeError."""
        aligner = TextEmbeddingAligner()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            aligner.transform(numerical_df)

    def test_fit_with_no_embeddings_raises(self) -> None:
        """fit() without text_embeddings_dict must raise ValueError."""
        aligner = TextEmbeddingAligner()
        with pytest.raises(ValueError, match="text_embeddings_dict must be provided"):
            aligner.fit()

    def test_fit_with_empty_dict_raises(self) -> None:
        """fit() with an empty dict must raise ValueError."""
        aligner = TextEmbeddingAligner()
        with pytest.raises(ValueError, match="empty"):
            aligner.fit(text_embeddings_dict={})

    def test_non_datetime_index_raises(self, hourly_index: pd.DatetimeIndex) -> None:
        """transform() with a non-datetime index must raise TypeError."""
        df = pd.DataFrame({"a": [1, 2]}, index=[0, 1])
        embeddings = {hourly_index[0]: np.array([1.0])}

        aligner = TextEmbeddingAligner()
        aligner.fit(text_embeddings_dict=embeddings)

        with pytest.raises(TypeError, match="DatetimeIndex"):
            aligner.transform(df)

    def test_invalid_embeddings_type_raises(self) -> None:
        """Passing a list instead of dict/DataFrame must raise TypeError."""
        aligner = TextEmbeddingAligner()
        with pytest.raises(TypeError, match="dict or pd.DataFrame"):
            aligner.fit(text_embeddings_dict=[[1, 2, 3]])


# ---------------------------------------------------------------------------
# DataFrame-based embeddings
# ---------------------------------------------------------------------------


def test_dataframe_embeddings(
    numerical_df: pd.DataFrame, hourly_index: pd.DatetimeIndex
) -> None:
    """Embeddings supplied as a DataFrame should work identically to a dict."""
    embedding_dim = 3
    emb_df = pd.DataFrame(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        index=[hourly_index[0], hourly_index[2]],
        columns=["e0", "e1", "e2"],
    )

    aligner = TextEmbeddingAligner()
    aligner.fit(text_embeddings_dict=emb_df)
    result = aligner.transform(numerical_df)

    expected_row_0 = np.array([1.0, 10.0, 0.1, 0.2, 0.3])
    np.testing.assert_array_almost_equal(result[0], expected_row_0)

    # Missing row 1 → zeros
    np.testing.assert_array_equal(result[1, 2:], np.zeros(embedding_dim))


# ---------------------------------------------------------------------------
# fit_transform shortcut
# ---------------------------------------------------------------------------


def test_fit_transform(
    numerical_df: pd.DataFrame, hourly_index: pd.DatetimeIndex
) -> None:
    """fit_transform should produce identical results to fit() + transform()."""
    embeddings = {hourly_index[0]: np.array([1.0, 2.0])}

    aligner_a = TextEmbeddingAligner()
    aligner_a.fit(text_embeddings_dict=embeddings)
    result_a = aligner_a.transform(numerical_df)

    aligner_b = TextEmbeddingAligner()
    result_b = aligner_b.fit_transform(numerical_df, text_embeddings_dict=embeddings)

    np.testing.assert_array_equal(result_a, result_b)
