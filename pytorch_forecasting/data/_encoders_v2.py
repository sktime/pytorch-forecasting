import warnings

import numpy as np
import pandas as pd


class D1CategoricalEncoder:
    """
    Categorical Label Encoder for the v2 D1 Layer.
    Scans specified categorical columns and maps string/text values to integers.
    """

    def __init__(
        self,
        columns: str | list[str] = None,
        handle_unknown: str = "assign_new",
    ):
        """
        Args:
            columns: List of column names to encode. If None, it will encode
                all object/category columns.
            handle_unknown: How to handle unseen categories during transform.
                'assign_new' gives them a new integer (like 0).
        """
        self.columns = [columns] if isinstance(columns, str) else columns
        self.handle_unknown = handle_unknown

        self.mapping_: dict[str, dict[str, int]] = {}
        self.inverse_mapping_: dict[str, dict[int, str]] = {}
        self._is_fitted = False
        self._warned_cols = set()

    def fit(self, df: pd.DataFrame):
        """Learns the vocabulary from the dataframe."""
        if self.columns is None:
            self.columns = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        for col in self.columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")

            series = df[col].fillna("NaN_CATEGORY")

            _, uniques = pd.factorize(series, sort=True)

            self.mapping_[col] = {val: idx + 1 for idx, val in enumerate(uniques)}

            self.inverse_mapping_[col] = {
                idx: val for val, idx in self.mapping_[col].items()
            }

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the integer translation to the dataframe."""
        if not self._is_fitted:
            raise RuntimeError("You must call fit() before transform().")

        df_encoded = df.copy()

        for col in self.columns:
            if col not in df_encoded.columns:
                continue

            series = df_encoded[col].fillna("NaN_CATEGORY")

            encoded_col = series.map(self.mapping_[col])

            if encoded_col.isna().any():
                if self.handle_unknown == "assign_new":
                    encoded_col = encoded_col.fillna(0)
                    if col not in self._warned_cols:
                        warnings.warn(
                            f"Unseen categories found in column '{col}'. "
                            "Assigned to index 0."
                        )
                        self._warned_cols.add(col)
                else:
                    raise ValueError(
                        f"Unseen categories found in column '{col}' "
                        "and handle_unknown!='assign_new'"
                    )

            df_encoded[col] = encoded_col.astype(int)

        return df_encoded

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Translates the integers back into original text/categories."""
        if not self._is_fitted:
            raise RuntimeError("You must call fit() before inverse_transform().")

        df_decoded = df.copy()

        for col in self.columns:
            if col not in df_decoded.columns:
                continue

            df_decoded[col] = df_decoded[col].map(self.inverse_mapping_[col])
            df_decoded[col] = df_decoded[col].replace("NaN_CATEGORY", np.nan)

        return df_decoded
