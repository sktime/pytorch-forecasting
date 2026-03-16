"""
Foundation model data modules for pytorch-forecasting.

Provides ``TTMDataModule`` for IBM's TinyTimeMixer (TTM) and the backing
``_TTMDataset``.  The pattern — subclassing ``TslibDataModule`` and setting
``_dataset_class`` — is designed to be reused for other foundation models
(Chronos, Moirai, etc.) with their own ``__getitem__`` overrides.
"""

from typing import Any

import torch

from pytorch_forecasting.data._tslib_data_module import (
    TslibDataModule,
    _TslibDataset,
)


class _TTMDataset(_TslibDataset):
    """
    Dataset for TinyTimeMixer (TTM) foundation model.

    Overrides ``__getitem__`` to produce ``past_values``,
    ``past_observed_mask``, and ``future_values`` in the channel layout
    expected by TTM::

        past_values: [targets | past-only covariates | known-future covariates]

    Parameters
    ----------
    dataset : TimeSeries
        The underlying time series dataset.
    data_module : TTMDataModule
        Parent data module — provides channel index metadata.
    windows : list[tuple[int, int, int, int]]
        Pre-computed (series_idx, start_idx, context_len, pred_len) windows.
    add_relative_time_idx : bool, default False
        Passed through to base class (unused by TTM).
    """

    def __init__(self, dataset, data_module, windows, add_relative_time_idx=False):
        super().__init__(dataset, data_module, windows, add_relative_time_idx)
        # Capture stage at construction time so that a subsequent setup() call
        # on the data module does not retroactively change behaviour for
        # already-instantiated datasets.
        self._stage = getattr(data_module, "_stage", None)

    def __getitem__(self, idx: int) -> tuple[dict[str, Any], Any]:
        """
        Return one TTM-compatible sample.

        Channel ordering in ``past_values``::

            [targets | past-only covariates | known-future covariates]

        Both groups follow the ordering in ``metadata["feature_names"]["continuous"]``.
        This ordering must match the index derivation in ``TTMDataModule.setup()``.
        """
        series_idx, start_idx, context_length, prediction_length = self.windows[idx]

        # TODO: cache _preprocess_data output per index (Finding 5 from issue #2184)
        # Plan: populate self._cache: dict[int, ...] lazily on first __getitem__ access
        processed_data = self.data_module._preprocess_data(series_idx)

        history_indices = slice(start_idx, start_idx + context_length)
        future_indices = slice(
            start_idx + context_length,
            start_idx + context_length + prediction_length,
        )

        # Target: (context_length, n_targets) and (prediction_length, n_targets)
        # Note: target is always 2D (total_len, n_targets) because TimeSeries in
        # _timeseries_v2.py stores y as data[_target] where _target is always a list.
        # Even for a single target, data[["target"]].to_numpy() yields shape (T, 1).
        history_target = processed_data["target"][history_indices]
        future_target = processed_data["target"][future_indices]

        # Continuous features: (total_len, n_cont) — targets already excluded by base
        cont_features = processed_data["features"]["continuous"]
        cont_names = self.data_module.metadata["feature_names"]["continuous"]
        cont_name_to_idx = {name: i for i, name in enumerate(cont_names)}

        past_only_names = self.data_module._past_only_cont_names
        known_future_names = self.data_module._known_future_cont_names

        past_only_idx = [cont_name_to_idx[n] for n in past_only_names]
        known_future_idx = [cont_name_to_idx[n] for n in known_future_names]

        # Slice history window
        history_cont = cont_features[history_indices]  # (ctx, n_cont)

        # Build past_values: targets | past-only | known-future
        parts = [history_target]
        if past_only_idx:
            parts.append(history_cont[:, past_only_idx])
        if known_future_idx:
            parts.append(history_cont[:, known_future_idx])

        past_values = torch.cat(parts, dim=-1)  # (context_length, n_channels)

        if past_values.ndim != 2:
            raise ValueError(
                f"past_values must be 2D (seq, channels), got shape {past_values.shape}"
            )

        # Per-channel observed mask: 1.0 observed, 0.0 NaN
        past_observed_mask = (~torch.isnan(past_values)).float()
        past_values = torch.nan_to_num(past_values, nan=0.0)

        # future_values: continuous known-future covariates only (no targets)
        future_cont = cont_features[future_indices]  # (pred, n_cont)
        future_known = (
            future_cont[:, known_future_idx]
            if known_future_idx
            else torch.zeros(prediction_length, 0)
        )

        x: dict[str, Any] = {
            "past_values": past_values,
            "past_observed_mask": past_observed_mask,
            "prediction_channel_indices": self.data_module._prediction_channel_indices,
        }

        # Stage-aware: omit future_values during predict; include for all other stages
        # (None is treated identically to "fit" — include future_values)
        if self._stage != "predict":
            x["future_values"] = future_known

        # y — squeeze to (prediction_length,) for single target
        if self.data_module.n_targets > 1:
            y = [future_target[:, i] for i in range(self.data_module.n_targets)]
        else:
            y = future_target.squeeze(-1)

        return x, y


class TTMDataModule(TslibDataModule):
    """
    Data module for TinyTimeMixer (TTM) foundation model.

    Subclasses ``TslibDataModule``, swapping in ``_TTMDataset`` via the
    ``_dataset_class`` hook and providing a generic ``collate_fn`` that does
    not hardcode key names.

    Parameters
    ----------
    Same as ``TslibDataModule``.
    """

    _dataset_class = _TTMDataset

    def setup(self, stage: str | None = None) -> None:
        """
        Set up datasets for fit / test / predict stages.

        Stores ``_stage`` on the instance before calling ``super().setup()``
        so that ``_TTMDataset.__init__`` can capture it at construction time.
        Also derives ``_prediction_channel_indices`` and
        ``_exogenous_channel_indices`` from dataset metadata.

        Notes
        -----
        ``self.metadata`` is a lazy property backed by
        ``self.time_series_metadata``, which is populated in ``__init__``.
        It is therefore safe to access here before ``super().setup()`` runs.

        **Repeated calls:** The ``hasattr`` idempotency guards in the base
        ``setup()`` check ``_train_dataset`` / ``_test_dataset`` (leading
        underscore) but assign ``train_dataset`` / ``test_dataset`` (no
        underscore) — so the guards are permanently inactive and datasets are
        re-created on every call.  If Lightning calls ``setup("fit")`` then
        ``setup("predict")`` on the same instance, ``train_dataset`` will be
        replaced by a ``_TTMDataset`` with ``_stage="predict"``, which will
        omit ``future_values`` from training batches.  Do not iterate
        ``train_dataloader()`` after calling ``setup("predict")`` on the same
        instance.
        """
        self._stage = stage

        # Derive channel indices -----------------------------------------------
        feature_names = self.metadata["feature_names"]
        cont_names = feature_names["continuous"]  # already excludes targets
        known_names = set(feature_names["known"])
        n_targets = self.metadata["n_features"]["target"]

        # Ordering must match _TTMDataset.__getitem__ concatenation:
        #   targets → past-only (cont_names order) → known-future (cont_names order)
        past_only_cont = [n for n in cont_names if n not in known_names]
        known_future_cont = [n for n in cont_names if n in known_names]

        # Indices into past_values (NOT into the raw feature array)
        self._prediction_channel_indices = list(range(n_targets))
        self._exogenous_channel_indices = list(
            range(n_targets, n_targets + len(past_only_cont) + len(known_future_cont))
        )
        # Stash name lists so __getitem__ can build the concat in the same order
        self._past_only_cont_names = past_only_cont
        self._known_future_cont_names = known_future_cont

        super().setup(stage)

    @staticmethod
    def collate_fn(batch):
        """
        Generic collate function for TTM batches.

        All ``torch.Tensor``-valued keys are stacked into ``(batch, ...)``
        tensors.  Non-Tensor keys (e.g. ``prediction_channel_indices``) are
        deduplicated to a single value — they are dataset-level constants
        identical across every sample in a batch.

        .. warning::
            The non-Tensor deduplication (taking ``batch[0][0][k]``) is
            correct **only** for dataset-level constants.  Do NOT add
            per-sample non-Tensor values to the ``x`` dict without revisiting
            this logic — they would be silently dropped for all but the first
            sample.
        """
        x_keys = batch[0][0].keys()
        x_batch = {}
        for k in x_keys:
            sample_val = batch[0][0][k]
            if isinstance(sample_val, torch.Tensor):
                x_batch[k] = torch.stack([x[k] for x, _ in batch])
            else:
                x_batch[k] = sample_val

        if isinstance(batch[0][1], list | tuple):
            num_targets = len(batch[0][1])
            y_batch = [
                torch.stack([s[i] for _, s in batch]) for i in range(num_targets)
            ]
        else:
            y_batch = torch.stack([y for _, y in batch])

        return x_batch, y_batch
