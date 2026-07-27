#######################################################################################
# Disclaimer: This data-module is still work in progress and experimental, please
# use with care. This data-module is a basic skeleton of how the data-handling pipeline
# may look like in the future.
# This is D2 layer that will handle the preprocessing and data loaders.
# For now, this pipeline handles the simplest situation: The whole data can be loaded
# into the memory.
#######################################################################################
from typing import Any
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
from torch.utils.data import Dataset

from pytorch_forecasting.adapters import ScalerAdapter
from pytorch_forecasting.data.data_module.base._base_data_module import (
    NORMALIZER,
    BaseTimeSeriesDataModule,
)
from pytorch_forecasting.data.data_module.encoder_decoder._encoder_decoder_dataset import (  # noqa: E501
    _ProcessedEncoderDecoderDataset,
)
from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    MultiNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)
from pytorch_forecasting.data.timeseries import TimeSeries
from pytorch_forecasting.utils._coerce import _coerce_to_dict


class EncoderDecoderTimeSeriesDataModule(BaseTimeSeriesDataModule):
    """
    Lightning DataModule for processing time series data in an encoder-decoder format.

    This module handles preprocessing, splitting, and batching of time series data
    for use in deep learning models. It supports categorical and continuous features,
    various scalers, and automatic target normalization.

    Parameters
    ----------
    time_series_dataset : TimeSeries
        The dataset containing time series data.
    max_encoder_length : int, default=30
        Maximum length of the encoder input sequence.
    min_encoder_length : Optional[int], default=None
        Minimum length of the encoder input sequence.
        Defaults to `max_encoder_length` if not specified.
    max_prediction_length : int, default=1
        Maximum length of the decoder output sequence.
    min_prediction_length : Optional[int], default=None
        Minimum length of the decoder output sequence.
        Defaults to `max_prediction_length` if not specified.
    min_prediction_idx : Optional[int], default=None
        Minimum index from which predictions start.
    allow_missing_timesteps : bool, default=False
        Whether to allow missing timesteps in the dataset.
    add_relative_time_idx : bool, default=False
        Whether to add a relative time index feature.
    add_target_scales : bool, default=False
        Whether to add target scaling information.
    add_encoder_length : Union[bool, str], default="auto"
        Whether to include encoder length information.
    target_normalizer : torch transformer, str, list, tuple, optional, default=None
        Transformer that takes group_ids, target and time_idx to normalize targets.
        You can choose from
        :py:class:`~pytorch_forecasting.data.encoders.TorchNormalizer`,
        :py:class:`~pytorch_forecasting.data.encoders.GroupNormalizer`,
        :py:class:`~pytorch_forecasting.data.encoders.NaNLabelEncoder`,
        :py:class:`~pytorch_forecasting.data.encoders.EncoderNormalizer`
        (on which overfitting tests will fail)
        or ``None`` for using no normalizer. For multiple targets, use a
        :py:class`~pytorch_forecasting.data.encoders.MultiNormalizer`.
        By default an appropriate normalizer is chosen automatically.

    categorical_encoders : Optional[Dict[str, NaNLabelEncoder]], default=None
        Dictionary of categorical encoders.

    scalers : optional, default=None
        Mapping of continuous feature names to their designated scaling instances.

        Defaults to ``None`` - an Identity pass-through, leaving the raw
                    feature values untouched.
        Supported scaler options for individual feature keys include:

        * **PyTorch Forecasting Normalizers**:

          * :py:class:`~pytorch_forecasting.data.encoders.TorchNormalizer`
          * :py:class:`~pytorch_forecasting.data.encoders.GroupNormalizer`
          * :py:class:`~pytorch_forecasting.data.encoders.EncoderNormalizer`

        * **Scikit-Learn Scalers**:

          * ``StandardScaler``
          * ``RobustScaler``
          * ``MinMaxScaler``
          * ``MaxAbsScaler``

    randomize_length : Union[None, Tuple[float, float], bool], default=False
        Whether to randomize input sequence length.
    batch_size : int, default=32
        Batch size for DataLoader.
    num_workers : int, default=0
        Number of workers for DataLoader.
    train_val_test_split : tuple, default=(0.7, 0.15, 0.15)
        Proportions for train, validation, and test dataset splits.
    """

    def __init__(
        self,
        time_series_dataset: TimeSeries,
        max_encoder_length: int = 30,
        min_encoder_length: int | None = None,
        max_prediction_length: int = 1,
        min_prediction_length: int | None = None,
        min_prediction_idx: int | None = None,
        allow_missing_timesteps: bool = False,
        add_relative_time_idx: bool = False,
        add_target_scales: bool = False,
        add_encoder_length: bool | str = "auto",
        target_normalizer: NORMALIZER
        | str
        | list[NORMALIZER]
        | tuple[NORMALIZER]
        | None = None,
        categorical_encoders: dict[str, NaNLabelEncoder] | None = None,
        scalers: dict[
            str, StandardScaler | RobustScaler | TorchNormalizer | EncoderNormalizer
        ]
        | None = None,
        randomize_length: None | tuple[float, float] | bool = False,
        batch_size: int = 32,
        num_workers: int = 0,
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
    ):
        super().__init__(
            time_series_dataset=time_series_dataset,
            target_normalizer=target_normalizer,
            batch_size=batch_size,
            num_workers=num_workers,
            train_val_test_split=train_val_test_split,
            add_relative_time_idx=add_relative_time_idx,
        )

        self.max_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        self.max_prediction_length = max_prediction_length
        self.min_prediction_length = min_prediction_length
        self.min_prediction_idx = min_prediction_idx
        self.allow_missing_timesteps = allow_missing_timesteps
        self.add_target_scales = add_target_scales
        self.add_encoder_length = add_encoder_length
        self.randomize_length = randomize_length
        self.categorical_encoders = categorical_encoders
        self.scalers = scalers

        self._min_prediction_length = min_prediction_length or max_prediction_length
        self._min_encoder_length = min_encoder_length or max_encoder_length
        self._categorical_encoders = _coerce_to_dict(categorical_encoders)
        self._scalers = _coerce_to_dict(scalers)
        self._target_normalizer_fitted = False
        self._feature_scalers_fitted = False

        self._scalers = {
            k: ScalerAdapter(v) for k, v in _coerce_to_dict(scalers).items()
        }
        self._build_cont_scalers()

    def _context_length(self) -> int:
        return self.max_encoder_length

    def _prediction_length(self) -> int:
        return self.max_prediction_length

    def _build_dataset(self, indices: torch.Tensor) -> Dataset:
        """Preprocess series, create windows, and wrap them in a Dataset.

        Parameters
        ----------
        indices : torch.Tensor
            Series indices for this split.

        Returns
        -------
        Dataset
            ``_ProcessedEncoderDecoderDataset`` over the split windows.
        """
        preprocessed = {
            idx.item(): self._preprocess_data(idx.item()) for idx in indices
        }
        windows = self._create_windows(indices)
        return _ProcessedEncoderDecoderDataset(
            self,
            windows,
            preprocessed,
            self.add_relative_time_idx,
        )

    def _prepare_metadata(self):
        """Prepare metadata for model initialisation.

        Returns
        -------
        dict
            dictionary containing the following keys:

            * ``encoder_cat``: Number of categorical variables in the encoder.
                Computed as ``len(self.categorical_indices)``, which counts the
                categorical feature indices.
            * ``encoder_cont``: Number of continuous variables in the encoder.
                Computed as ``len(self.continuous_indices)``, which counts the
                continuous feature indices.
            * ``decoder_cat``: Number of categorical variables in the decoder that
                are known in advance.
                Computed by filtering ``self.time_series_metadata["cols"]["x"]``
                where col_type == "C"(categorical) and col_known == "K" (known)
            * ``decoder_cont``:  Number of continuous variables in the decoder that
                are known in advance.
                Computed by filtering ``self.time_series_metadata["cols"]["x"]``
                where col_type == "F"(continuous) and col_known == "K"(known)
            * ``target``: Number of target variables.
                Computed as ``len(self.time_series_metadata["cols"]["y"])``, which
                gives the number of output target columns..
            * ``static_categorical_features``: Number of static categorical features
                Computed by filtering ``self.time_series_metadata["cols"]["st"]``
                (static features) where col_type == "C" (categorical).
            * ``static_continuous_features``: Number of static continuous features
                Computed as difference of
                ``len(self.time_series_metadata["cols"]["st"])`` (static features)
                and static_categorical_features that gives static continuous feature
            * ``max_encoder_length``: maximum encoder length
                Taken directly from `self.max_encoder_length`.
            * ``max_prediction_length``: maximum prediction length
                Taken directly from `self.max_prediction_length`.
            * ``min_encoder_length``: minimum encoder length
                Taken directly from `self.min_encoder_length`.
            * ``min_prediction_length``: minimum prediction length
                Taken directly from `self.min_prediction_length`.
        """
        encoder_cat_count = len(self.categorical_indices)
        encoder_cont_count = len(self.continuous_indices)

        decoder_cat_count = len(
            [
                col
                for col in self.time_series_metadata["cols"]["x"]
                if self.time_series_metadata["col_type"].get(col) == "C"
                and self.time_series_metadata["col_known"].get(col) == "K"
            ]
        )
        decoder_cont_count = len(
            [
                col
                for col in self.time_series_metadata["cols"]["x"]
                if self.time_series_metadata["col_type"].get(col) == "F"
                and self.time_series_metadata["col_known"].get(col) == "K"
            ]
        )

        target_count = len(self.time_series_metadata["cols"]["y"])
        metadata = {
            "encoder_cat": encoder_cat_count,
            "encoder_cont": encoder_cont_count,
            "decoder_cat": decoder_cat_count,
            "decoder_cont": decoder_cont_count,
            "target": target_count,
        }
        if self.time_series_metadata["cols"]["st"]:
            static_cat_count = len(
                [
                    col
                    for col in self.time_series_metadata["cols"]["st"]
                    if self.time_series_metadata["col_type"].get(col) == "C"
                ]
            )
            static_cont_count = (
                len(self.time_series_metadata["cols"]["st"]) - static_cat_count
            )

            metadata["static_categorical_features"] = static_cat_count
            metadata["static_continuous_features"] = static_cont_count
        else:
            metadata["static_categorical_features"] = 0
            metadata["static_continuous_features"] = 0

        metadata.update(
            {
                "max_encoder_length": self.max_encoder_length,
                "max_prediction_length": self.max_prediction_length,
                "min_encoder_length": self._min_encoder_length,
                "min_prediction_length": self._min_prediction_length,
            }
        )

        return metadata

    def _create_windows(self, indices: torch.Tensor) -> list[tuple[int, int, int, int]]:
        """Generate sliding windows for training, validation, and testing.

        Returns
        -------
        List[Tuple[int, int, int, int]]
            A list of tuples, where each tuple consists of:
            - ``series_idx`` : int
              Index of the time series in `time_series_dataset`.
            - ``start_idx`` : int
              Start index of the encoder window.
            - ``enc_length`` : int
              Length of the encoder input sequence.
            - ``pred_length`` : int
              Length of the decoder output sequence.
        """
        windows = []

        for idx in indices:
            series_idx = idx.item()
            sample = self.time_series_dataset[series_idx]
            sequence_length = len(sample["y"])

            if sequence_length < self.max_encoder_length + self.max_prediction_length:
                continue

            effective_min_prediction_idx = (
                self.min_prediction_idx
                if self.min_prediction_idx is not None
                else self.max_encoder_length
            )

            max_prediction_idx = sequence_length - self.max_prediction_length + 1

            if max_prediction_idx <= effective_min_prediction_idx:
                continue

            for start_idx in range(
                0, max_prediction_idx - effective_min_prediction_idx
            ):
                if (
                    start_idx + self.max_encoder_length + self.max_prediction_length
                    <= sequence_length
                ):
                    windows.append(
                        (
                            series_idx,
                            start_idx,
                            self.max_encoder_length,
                            self.max_prediction_length,
                        )
                    )

        return windows

    def _ensure_split(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split data indices into train, val, and test sets based on the
        train_val_test_split ratio once and cache them.
        """
        if hasattr(self, "_split_indices"):
            return
        # this is a very rudimentary way to handle the splits when
        # the dataset is of size equal to 1 or 2.
        total_series = len(self.time_series_dataset)
        self._split_indices = torch.randperm(total_series)

        self._train_size = int(self.train_val_test_split[0] * total_series)
        self._val_size = int(self.train_val_test_split[1] * total_series)

        self._train_indices = self._split_indices[: self._train_size]
        self._val_indices = self._split_indices[
            self._train_size : self._train_size + self._val_size
        ]
        self._test_indices = self._split_indices[self._train_size + self._val_size :]

    # region Preprocessing

    # TODO: once TSLib preprocessing is done,
    # we need to work on refactoring of preprocessing logic.
    # https://github.com/sktime/pytorch-forecasting/issues/2330

    def _build_cont_scalers(self):
        """Pre-resolve continuous feature scalers to (position, adapter) pairs."""
        self._cont_scalers = [
            (i, self._scalers[name])
            for i, name in enumerate(
                self.time_series_metadata["cols"]["x"][idx]
                for idx in self.continuous_indices
            )
            if name in self._scalers
        ]

    def _coerce_target_normalizer(
        self,
        target_normalizer: NORMALIZER
        | str
        | list[NORMALIZER]
        | tuple[NORMALIZER]
        | None,
    ):
        _target_normalizer = None
        if isinstance(target_normalizer, str) and target_normalizer.lower() == "auto":
            self._auto_normalizer = True
        elif isinstance(target_normalizer, (tuple, list)):
            _target_normalizer = ScalerAdapter(MultiNormalizer(list(target_normalizer)))
            self._auto_normalizer = False
        else:
            _target_normalizer = ScalerAdapter(self.target_normalizer)
            self._auto_normalizer = False
        return _target_normalizer

    def _get_group_dataframe(
        self, series_idx: int, n_timesteps: int
    ) -> pd.DataFrame | None:
        """Build a DataFrame with group columns for a given series.

        Parameters
        ----------
        series_idx : int
            Index of the time series in the dataset.
        n_timesteps : int
            Number of timesteps to repeat group values for.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with group columns repeated for each timestep,
            or None if no group columns are defined.
        """
        ts = self.time_series_dataset
        if not ts._group:
            return None

        group_id = ts._group_ids[series_idx]
        if not isinstance(group_id, tuple):
            group_id = (group_id,)

        group_data = {
            col: np.repeat(val, n_timesteps) for col, val in zip(ts._group, group_id)
        }
        return pd.DataFrame(group_data)

    def _coerce_sample(
        self, sample: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert raw sample arrays to float tensors and compute time mask."""
        target = sample["y"]
        features = sample["x"]
        times = sample["t"]
        cutoff_time = sample["cutoff_time"]

        target = target.float()
        features = features.float()

        if target.ndim == 1:
            target = target.unsqueeze(-1)

        time_mask = torch.tensor(times <= cutoff_time, dtype=torch.bool)
        return target, features, times, time_mask

    def _split_features(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split feature tensor into categorical and continuous subsets."""
        n_timesteps = features.shape[0]
        categorical = (
            features[:, self.categorical_indices]
            if self.categorical_indices
            else torch.zeros((n_timesteps, 0))
        )
        continuous = (
            features[:, self.continuous_indices]
            if self.continuous_indices
            else torch.zeros((n_timesteps, 0))
        )
        return {"categorical": categorical, "continuous": continuous}

    def _normalize_target(
        self, target: torch.Tensor, series_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply global target normalization.

        Returns
        -------
        target : normalized tensor
        target_original : pre-normalization clone (for scale computation)
        """
        target_original = target.clone()

        if self._target_normalizer is None or not self._target_normalizer_fitted:
            return target, target_original

        if not self._target_normalizer.fit_per_sequence:
            X = self._get_group_dataframe(series_idx, target.shape[0])
            target = self._target_normalizer.transform(target, X)

        for i, is_enc in enumerate(self._target_normalizer.label_encoder_mask):
            if is_enc:
                target_original[:, i] = target[:, i]

        return target, target_original

    def _normalize_features(
        self, continuous: torch.Tensor, series_idx: int
    ) -> torch.Tensor:
        """Apply global continuous feature scalers."""
        if not self._feature_scalers_fitted or not self.continuous_indices:
            return continuous

        continuous = continuous.clone()
        X = self._get_group_dataframe(series_idx, continuous.shape[0])
        feature_names = [
            self.time_series_metadata["cols"]["x"][idx]
            for idx in self.continuous_indices
        ]

        for feat_idx, feat_name in enumerate(feature_names):
            if feat_name in self._scalers:
                adapter = self._scalers[feat_name]
                if not adapter.fit_per_sequence:
                    continuous[:, feat_idx] = adapter.transform(
                        continuous[:, feat_idx], X
                    )
        return continuous

    def _preprocess_data(self, series_idx: int) -> dict[str, Any]:
        """Preprocess one series into a cache dict.

        Composes coercion, feature splitting, and global normalization.
        Sequence-local normalization (EncoderNormalizer) is deferred to
        __getitem__.
        """
        sample = self.time_series_dataset[series_idx]
        target, features, times, time_mask = self._coerce_sample(sample)
        split = self._split_features(features)
        target, target_original = self._normalize_target(target, series_idx)
        continuous = self._normalize_features(split["continuous"], series_idx)

        return {
            "features": {"categorical": split["categorical"], "continuous": continuous},
            "target": target,
            "target_original": target_original,
            "static": sample.get("st", None),
            "group": sample.get("group", torch.tensor([0])),
            "length": len(target),
            "time_mask": time_mask,
            "times": times,
            "timestep": times,
            "cutoff_time": sample["cutoff_time"],
        }

    def _fit_target_normalizer(self, train_indices):
        """Fit target normalizer on the target variable's training data."""

        if self._target_normalizer is None:
            return

        if (
            not self._target_normalizer.is_multi
            and self._target_normalizer.fit_per_sequence
        ):
            return

        all_targets = []
        all_groups = []
        for idx in train_indices:
            series_idx = idx.item()
            sample = self.time_series_dataset[idx]
            target = sample["y"]
            all_targets.append(target)
            n_timesteps = len(target)
            all_groups.append(self._get_group_dataframe(series_idx, n_timesteps))

        if not all_targets:
            return

        all_targets = torch.cat(all_targets, dim=0)
        X = (
            pd.concat(all_groups, ignore_index=True)
            if all_groups[0] is not None
            else None
        )

        self._target_normalizer.fit(all_targets, X)
        self._target_normalizer_fitted = True

    def _fit_scalers(self, train_indices):
        """Fit scalers on continuous features in the training data."""

        if not self._scalers or not self.continuous_indices:
            return

        features_to_scale = {
            self.time_series_metadata["cols"]["x"][idx]: pos
            for pos, idx in enumerate(self.continuous_indices)
        }

        for feat_name, adapter in self._scalers.items():
            if feat_name not in features_to_scale:
                continue
            feat_idx = features_to_scale[feat_name]
            feat_data = []
            all_groups = []

            for idx in train_indices:
                series_idx = idx.item()
                sample = self.time_series_dataset[idx]
                feature_data = sample["x"][:, feat_idx]
                feat_data.append(feature_data)
                all_groups.append(
                    self._get_group_dataframe(series_idx, len(feature_data))
                )

            feat_data = torch.cat(feat_data, dim=0)
            X = (
                pd.concat(all_groups, ignore_index=True)
                if all_groups[0] is not None
                else None
            )
            adapter.fit(feat_data, X)

        self._feature_scalers_fitted = True

    def _compute_data_properties(self, train_indices: torch.Tensor) -> dict:
        """Scan training targets to determine per-target type, positivity, skewness.

        Returns
        -------
        dict with keys:
            - ``target_type``: dict[str, str]
                if target is``"categorical"`` or ``"real"``
            - ``target_positive``: dict[str, bool]
                True if all values strictly positive
            - ``target_skew``: dict[str, float]
                Pearson moment skewness
        """
        target_names = self.time_series_metadata["cols"]["y"]
        col_type = self.time_series_metadata["col_type"]
        per_target = {name: [] for name in target_names}

        for idx in train_indices:
            sample = self.time_series_dataset[idx.item()]
            target = sample["y"]
            for i, name in enumerate(target_names):
                per_target[name].append(target[..., i] if target.ndim > 1 else target)

        target_type, target_positive, target_skew = {}, {}, {}

        for name in target_names:
            target_type[name] = "categorical" if col_type.get(name) == "C" else "real"

            valid_vals = torch.cat(per_target[name]).float()
            valid_vals = valid_vals[~torch.isnan(valid_vals)]

            if target_type[name] == "categorical" or valid_vals.numel() == 0:
                target_positive[name] = False
                target_skew[name] = 0.0
                continue

            target_positive[name] = bool((valid_vals > 0).all())
            mean = valid_vals.mean()
            std = valid_vals.std()
            target_skew[name] = (
                0.0 if std == 0 else float(((valid_vals - mean) ** 3).mean() / (std**3))
            )

        return {
            "target_type": target_type,
            "target_positive": target_positive,
            "target_skew": target_skew,
        }

    def _get_auto_normalizer(self, data_properties: dict) -> NORMALIZER:
        """Select normalizer based on data properties and current module config.

        Parameters
        ----------
        data_properties : dict
            As returned by ``_compute_data_properties``.
        """
        target_names = self.time_series_metadata["cols"]["y"]
        has_groups = bool(self.time_series_dataset._group)
        use_encoder_normalizer = (
            self.max_encoder_length > 20 and self._min_encoder_length > 1
        )

        normalizers = []
        for target in target_names:
            if data_properties["target_type"][target] == "categorical":
                if self.add_target_scales:
                    warn(
                        "Target scales will be only added for continuous targets",
                        UserWarning,
                    )
                normalizers.append(NaNLabelEncoder())
                continue

            if data_properties["target_positive"][target]:
                transformer = (
                    "log" if data_properties["target_skew"][target] > 2.5 else "relu"
                )
            else:
                transformer = None

            if use_encoder_normalizer:
                normalizers.append(EncoderNormalizer(transformation=transformer))
            elif has_groups:
                from pytorch_forecasting.data.encoders import GroupNormalizer

                normalizers.append(GroupNormalizer(transformation=transformer))
            else:
                normalizers.append(TorchNormalizer(transformation=transformer))

        return MultiNormalizer(normalizers) if self.n_targets > 1 else normalizers[0]

    def _resolve_target_normalizer(self, train_indices: torch.Tensor) -> None:
        """Resolve target normalizer"""
        if not self._auto_normalizer:
            return
        data_properties = self._compute_data_properties(train_indices)
        normalizer = self._get_auto_normalizer(data_properties)
        self._target_normalizer = ScalerAdapter(normalizer)

    def setup(self, stage: str | None = None):
        """Prepare the datasets for training, validation, testing, or prediction.

        Parameters
        ----------
        stage : Optional[str], default=None
            Specifies the stage of setup. Can be one of:
            - ``"fit"`` : Prepares training and validation datasets.
            - ``"test"`` : Prepares the test dataset.
            - ``"predict"`` : Prepares the dataset for inference.
            - ``None`` : Prepares ``fit`` datasets.
        """
        self._ensure_split()

        if stage is None or stage == "fit":
            self._resolve_target_normalizer(self._train_indices)
            if not self._target_normalizer_fitted:
                self._fit_target_normalizer(self._train_indices)
            if not self._feature_scalers_fitted:
                self._fit_scalers(self._train_indices)
            if self.train_dataset is None or self.val_dataset is None:
                self.train_dataset = self._build_dataset(self._train_indices)
                self.val_dataset = self._build_dataset(self._val_indices)
                self.train_windows = self.train_dataset.windows
                self.val_windows = self.val_dataset.windows

        elif stage == "test":
            if self.test_dataset is None:
                self.test_dataset = self._build_dataset(self._test_indices)
                self.test_windows = self.test_dataset.windows
        elif stage == "predict":
            predict_indices = torch.arange(len(self.time_series_dataset))
            self.predict_dataset = self._build_dataset(predict_indices)
            self.predict_windows = self.predict_dataset.windows

    # endregion

    @staticmethod
    def collate_fn(batch):
        """Stack encoder-decoder samples into batched ``x`` and ``y``.

        Parameters
        ----------
        batch : list of tuple[dict, target]
            Samples from ``_ProcessedEncoderDecoderDataset`` dataset.

        Returns
        -------
        tuple[dict, torch.Tensor or list[torch.Tensor]]
            Collated inputs and targets. Multivariate targets become a list of tensors.
        """
        x_batch = {
            "encoder_cat": torch.stack([x["encoder_cat"] for x, _ in batch]),
            "encoder_cont": torch.stack([x["encoder_cont"] for x, _ in batch]),
            "decoder_cat": torch.stack([x["decoder_cat"] for x, _ in batch]),
            "decoder_cont": torch.stack([x["decoder_cont"] for x, _ in batch]),
            "encoder_lengths": torch.stack([x["encoder_lengths"] for x, _ in batch]),
            "decoder_lengths": torch.stack([x["decoder_lengths"] for x, _ in batch]),
            "decoder_target_lengths": torch.stack(
                [x["decoder_target_lengths"] for x, _ in batch]
            ),
            "groups": torch.stack([x["groups"] for x, _ in batch]),
            "target_past": torch.stack([x["target_past"] for x, _ in batch]),
            "encoder_time_idx": torch.stack([x["encoder_time_idx"] for x, _ in batch]),
            "decoder_time_idx": torch.stack([x["decoder_time_idx"] for x, _ in batch]),
            "encoder_mask": torch.stack([x["encoder_mask"] for x, _ in batch]),
            "decoder_mask": torch.stack([x["decoder_mask"] for x, _ in batch]),
        }

        if "static_categorical_features" in batch[0][0]:
            x_batch["static_categorical_features"] = torch.stack(
                [x["static_categorical_features"] for x, _ in batch]
            )
            x_batch["static_continuous_features"] = torch.stack(
                [x["static_continuous_features"] for x, _ in batch]
            )

        if isinstance(batch[0][0]["target_scale"], list | tuple):
            num_targets = len(batch[0][0]["target_scale"])
            target_scale = [
                torch.stack([x["target_scale"][i] for x, _ in batch])
                for i in range(num_targets)
            ]
        else:
            target_scale = torch.stack([x["target_scale"] for x, _ in batch])

        x_batch["target_scale"] = target_scale

        if isinstance(batch[0][1], list | tuple):
            num_targets = len(batch[0][1])
            y_batch = []
            for i in range(num_targets):
                target_tensors = [sample_y[i] for _, sample_y in batch]
                stacked_target = torch.stack(target_tensors)
                y_batch.append(stacked_target)
        else:
            y_batch = torch.stack([y for _, y in batch])
        return x_batch, y_batch
