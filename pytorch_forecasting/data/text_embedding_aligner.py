"""Aligner for merging sparse text embeddings with dense numerical time-series.

This module provides :class:`TextEmbeddingAligner`, a scikit-learn–compatible
transformer that joins pre-computed text embedding vectors onto a numerical
time-series DataFrame indexed by datetime.  Timestamps without a corresponding
text embedding are zero-padded to maintain a fixed feature width.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TextEmbeddingAligner(BaseEstimator, TransformerMixin):
    """Align sparse text embeddings with dense numerical time-series data.

    Given a numerical DataFrame with a datetime index and a dictionary (or
    DataFrame) mapping timestamps to pre-computed text embedding vectors, this
    transformer concatenates the embedding onto each numerical row.  When no
    embedding exists for a given timestamp, a zero-vector of the learned
    ``embedding_dim`` is used instead.

    The class follows the scikit-learn estimator contract
    (:class:`~sklearn.base.BaseEstimator` / :class:`~sklearn.base.TransformerMixin`)
    so it can participate in ``Pipeline`` objects.

    Parameters
    ----------
    None – all configuration is inferred from the data during ``fit()``.

    Attributes
    ----------
    embedding_dim_ : int
        Dimensionality of the text embedding vectors, learned during ``fit()``.
    is_fitted_ : bool
        Whether the transformer has been fitted.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.date_range("2024-01-01", periods=4, freq="h")
    >>> numerical_df = pd.DataFrame({"a": [1, 2, 3, 4]}, index=idx)
    >>> embeddings = {idx[0]: np.array([0.1, 0.2]), idx[2]: np.array([0.3, 0.4])}
    >>> aligner = TextEmbeddingAligner()
    >>> aligner.fit(text_embeddings_dict=embeddings)
    TextEmbeddingAligner()
    >>> out = aligner.transform(numerical_df)
    >>> out.shape
    (4, 3)
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_embeddings(
        text_embeddings: dict[pd.Timestamp, np.ndarray] | pd.DataFrame,
    ) -> dict[pd.Timestamp, np.ndarray]:
        """Normalise embeddings input to a ``{Timestamp: ndarray}`` dict.

        Parameters
        ----------
        text_embeddings : Union[dict, pd.DataFrame]
            Either a dictionary mapping timestamps to 1-D numpy arrays or a
            DataFrame whose index is datetime and each row is an embedding.

        Returns
        -------
        dict[pd.Timestamp, np.ndarray]
            Canonical mapping from timestamp to embedding vector.

        Raises
        ------
        TypeError
            If *text_embeddings* is neither a dict nor a DataFrame.
        """
        if isinstance(text_embeddings, pd.DataFrame):
            return {ts: row.values for ts, row in text_embeddings.iterrows()}
        if isinstance(text_embeddings, dict):
            return text_embeddings
        raise TypeError(
            f"text_embeddings must be a dict or pd.DataFrame, "
            f"got {type(text_embeddings).__name__}"
        )

    @staticmethod
    def _infer_embedding_dim(
        embeddings: dict[pd.Timestamp, np.ndarray],
    ) -> int:
        """Return the embedding dimensionality from the first entry.

        Parameters
        ----------
        embeddings : dict[pd.Timestamp, np.ndarray]
            Non-empty mapping of timestamps to embedding vectors.

        Returns
        -------
        int
            Length of the first embedding vector.

        Raises
        ------
        ValueError
            If *embeddings* is empty.
        """
        if not embeddings:
            raise ValueError(
                "text_embeddings_dict is empty — cannot infer embedding_dim. "
                "Provide at least one embedding vector."
            )
        first_vec = next(iter(embeddings.values()))
        return int(np.asarray(first_vec).shape[0])

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame | None = None,
        y: None = None,
        *,
        text_embeddings_dict: (
            dict[pd.Timestamp, np.ndarray] | pd.DataFrame | None
        ) = None,
    ) -> TextEmbeddingAligner:
        """Learn the embedding dimensionality from supplied text embeddings.

        Parameters
        ----------
        X : pd.DataFrame or None, optional
            Ignored.  Present only for scikit-learn pipeline compatibility.
        y : None
            Ignored.
        text_embeddings_dict : Union[dict, pd.DataFrame]
            Mapping of timestamps to pre-computed embedding vectors.  At least
            one entry is required so the embedding dimension can be inferred.

        Returns
        -------
        TextEmbeddingAligner
            Fitted instance (``self``).

        Raises
        ------
        ValueError
            If *text_embeddings_dict* is ``None`` or empty.
        """
        if text_embeddings_dict is None:
            raise ValueError(
                "text_embeddings_dict must be provided to fit(). "
                "Pass a dict mapping timestamps to embedding vectors."
            )

        embeddings = self._resolve_embeddings(text_embeddings_dict)
        self.embedding_dim_ = self._infer_embedding_dim(embeddings)
        self.text_embeddings_dict_ = embeddings
        self.is_fitted_ = True
        return self

    def transform(
        self,
        numerical_df: pd.DataFrame,
        y: None = None,
        *,
        text_embeddings_dict: (
            dict[pd.Timestamp, np.ndarray] | pd.DataFrame | None
        ) = None,
    ) -> np.ndarray:
        """Concatenate numerical features with aligned text embeddings.

        For each row in *numerical_df*, the corresponding text embedding is
        looked up by exact timestamp match.  If no embedding exists, a
        zero-vector of length ``embedding_dim_`` is used.

        Parameters
        ----------
        numerical_df : pd.DataFrame
            Dense numerical time-series with a datetime index.
        y : None
            Ignored.
        text_embeddings_dict : Union[dict, pd.DataFrame], optional
            Override embedding source.  When ``None`` (default), the embeddings
            stored during ``fit()`` are used.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_timestamps, n_numerical_features + embedding_dim_)``
            with dtype ``np.float64``.

        Raises
        ------
        RuntimeError
            If the transformer has not been fitted.
        TypeError
            If *numerical_df* does not have a datetime index.
        """
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(
                "TextEmbeddingAligner has not been fitted yet. "
                "Call fit() before transform()."
            )

        if not isinstance(numerical_df.index, pd.DatetimeIndex):
            raise TypeError(
                "numerical_df must have a pd.DatetimeIndex. "
                f"Got index of type {type(numerical_df.index).__name__}."
            )

        # Resolve which embeddings to use.
        if text_embeddings_dict is not None:
            embeddings = self._resolve_embeddings(text_embeddings_dict)
        else:
            embeddings = self.text_embeddings_dict_

        embedding_dim = self.embedding_dim_

        numerical_values = numerical_df.values.astype(np.float64)
        n_rows = len(numerical_df)

        # Pre-allocate the embedding block.
        embedding_block = np.zeros((n_rows, embedding_dim), dtype=np.float64)

        for i, ts in enumerate(numerical_df.index):
            vec = embeddings.get(ts)
            if vec is not None:
                embedding_block[i] = np.asarray(vec, dtype=np.float64)
            # else: row already zero-filled

        return np.concatenate([numerical_values, embedding_block], axis=1)

    def fit_transform(
        self,
        numerical_df: pd.DataFrame,
        y: None = None,
        *,
        text_embeddings_dict: (
            dict[pd.Timestamp, np.ndarray] | pd.DataFrame | None
        ) = None,
    ) -> np.ndarray:
        """Fit and transform in a single step.

        Convenience wrapper equivalent to calling ``fit()`` then ``transform()``.

        Parameters
        ----------
        numerical_df : pd.DataFrame
            Dense numerical time-series with a datetime index.
        y : None
            Ignored.
        text_embeddings_dict : Union[dict, pd.DataFrame]
            Mapping of timestamps to pre-computed embedding vectors.

        Returns
        -------
        np.ndarray
            Concatenated array of shape
            ``(n_timestamps, n_numerical_features + embedding_dim_)``.
        """
        self.fit(text_embeddings_dict=text_embeddings_dict)
        return self.transform(numerical_df, text_embeddings_dict=text_embeddings_dict)
