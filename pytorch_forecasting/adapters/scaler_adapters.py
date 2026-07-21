import pandas as pd
import torch

from pytorch_forecasting._registry import all_objects
from pytorch_forecasting.adapters.scaler_strategy import (
    ScalerStrategy,
)
from pytorch_forecasting.adapters.utils import (
    ArrayLike,
    _to_numpy,
    _to_tensor,
)
from pytorch_forecasting.data.encoders import (
    MultiNormalizer,
)


def get_scaler_strategy(scaler) -> ScalerStrategy:
    """Single dispatch point: the only place that inspects scaler type."""
    discovered = all_objects(
        object_types="scaler_strategy",
        return_names=False,
    )
    for strategy_cls in discovered:
        if strategy_cls._is_applicable(scaler):
            return strategy_cls()
    return ScalerStrategy()


class ScalerAdapter:
    """
    Unified array-in / tensor-out interface for single and multi-target scalers.

    Accepts torch.Tensor, np.ndarray, or pd.Series as input. Output is always
    a torch.Tensor.  Type-specific behavior (sklearn scalers, GroupNormalizer,
    NaNLabelEncoder, EncoderNormalizer, ...) is delegated to a strategy chosen
    once at construction time (see ``adapters/scaler_strategy.py``).


    Parameters
    ----------
    scaler : object
        The underlying scaling/encoding instance. Accepted types, their expected
            origins, and assumed API contracts are:

            * scikit-learn scalers (from ``sklearn.preprocessing``):
                Implements ``.fit(X)` and ``.transform(X)``. Expects 2D
                numpy arrays of shape ``(n_samples, 1)``. Outputs numpy arrays.

            * ``TorchNormalizer``  (from ``pytorch_forecasting.data.encoders``):
                Implements `.fit(data)` and `.transform(data)`. Expects 1D
                tensors or numpy arrays. Output can be tensor or array.

            *``EncoderNormalizer`` (from ``pytorch_forecasting.data.encoders``):
                Implements `.fit(data)` and `.transform(data)`. Expects 1D
                tensors or numpy arrays. Output can be tensor or array.
                `EncoderNormalizer` signals that it must be fit per-sequence.

            * ``NaNLabelEncoder`` (from `pytorch_forecasting.data.encoders`):
                Implements ``.fit(data)`` and ``.transform(data)``. Expects a
                1D ``pd.Series`` (or 1D array). Used for categorical encoding.

            * ``GroupNormalizer`` (from ``pytorch_forecasting.data.encoders``):
                Implements ``.fit(data, X)`` and ``.transform(data, X)``. Expects
                `data` as a 1D ``pd.Series`` and `X` as a ``pd.DataFrame`` containing
                required group columns to compute grouped statistics.

            * ``MultiNormalizer`` (from ``pytorch_forecasting.data.encoders``):
                Implements ``.fit(data, X)`` and ``.transform(data.T, X)``.
                Expects 2D array-like inputs of shape ``(n_samples, n_targets)``.
                Must expose a ``.normalizers`` attribute (iterable) containing the
                individual sub-normalizers for each target.
    """

    def __init__(self, scaler):
        self._scaler = scaler
        self.is_multi = isinstance(scaler, MultiNormalizer)

        if self.is_multi:
            self._sub_adapters = [ScalerAdapter(norm) for norm in scaler.normalizers]
            self._strategy = None
            self.is_label_encoder = False
            self.fit_per_sequence = any(a.fit_per_sequence for a in self._sub_adapters)
        else:
            self._strategy = get_scaler_strategy(scaler) if scaler is not None else None
            self.is_label_encoder = (
                self._strategy.get_tag("is_label_encoder", None)
                if self._strategy
                else False
            )
            self.fit_per_sequence = (
                self._strategy.get_tag("fit_per_sequence", None)
                if self._strategy
                else False
            )

    @property
    def label_encoder_mask(self) -> list[bool]:
        """Per-target bool list indicating which sub-normalizers are label encoders."""
        if self.is_multi:
            return [sub.is_label_encoder for sub in self._sub_adapters]
        return [self.is_label_encoder]

    def _prepare_input(self, data: ArrayLike) -> ArrayLike:
        """Coerce data to the type the underlying scaler expects."""
        if self.is_multi:
            arr = _to_numpy(data)
            return arr if arr.ndim == 2 else arr[:, None]
        return self._strategy.prepare_input(data)

    def fit(self, data: ArrayLike, X: pd.DataFrame = None) -> "ScalerAdapter":
        """Fit the scaler.

        Parameters
        ----------
        data : tensor, ndarray, or Series
            Shape ``(n_samples,)`` for single-target or
            ``(n_samples, n_targets)`` for multi-target.
        X : pd.DataFrame, optional
            Group columns. Required when scaler is GroupNormalizer or
            when MultiNormalizer contains GroupNormalizer sub-normalizers.
        """
        if self._scaler is None:
            return self

        prepared = self._prepare_input(data)
        if self.is_multi:
            self._scaler.fit(prepared, X)
            return self

        self._strategy.fit(self._scaler, prepared, X)
        return self

    def transform(self, data: ArrayLike, X: pd.DataFrame = None) -> torch.Tensor:
        """Transform data, always returning a torch.Tensor.

        Parameters
        ----------
        data : tensor, ndarray, or Series
            Shape ``(n_samples,)`` for single-target or
            ``(n_samples, n_targets)`` for multi-target.
        X : pd.DataFrame, optional
            Group columns. Required when scaler is GroupNormalizer or
            when MultiNormalizer contains GroupNormalizer sub-normalizers.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        if self._scaler is None:
            return _to_tensor(data)
        prepared = self._prepare_input(data)

        if self.is_multi:
            results = self._scaler.transform(prepared.T, X)
            return torch.stack([_to_tensor(r) for r in results], dim=-1)

        return self._strategy.transform(self._scaler, prepared, data, X)

    def fit_transform(self, data: ArrayLike, X: pd.DataFrame = None) -> torch.Tensor:
        return self.fit(data, X).transform(data, X)

    def fit_transform_sequence(
        self, data: ArrayLike, X: pd.DataFrame = None
    ) -> torch.Tensor:
        """Fit-and-transform only per-sequence sub-normalizers; transform the rest.

        Used at ``__getitem__`` time for encoder windows. Non-per-sequence
        normalizers use their already-fitted global state.

        For single-target adapters this collapses to fit_transform
        (EncoderNormalizer) or transform (everything else).

        Parameters
        ----------
        data : tensor, ndarray, or Series
            Shape ``(enc_length,)`` or ``(enc_length, n_targets)``.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        if not self.is_multi:
            return (
                self.fit_transform(data, X)
                if self.fit_per_sequence
                else _to_tensor(data)
            )

        t = _to_tensor(data)
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        columns = []
        for idx, sub in enumerate(self._sub_adapters):
            col = t[:, idx]
            col = sub.fit_transform(col, X) if sub.fit_per_sequence else col
            columns.append(col.unsqueeze(-1))
        return torch.cat(columns, dim=-1)
