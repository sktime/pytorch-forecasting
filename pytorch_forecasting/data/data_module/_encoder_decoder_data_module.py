#######################################################################################
# Disclaimer: This data-module is still work in progress and experimental, please
# use with care. This data-module is a basic skeleton of how the data-handling pipeline
# may look like in the future.
# This is D2 layer that will handle the preprocessing and data loaders.
# For now, this pipeline handles the simplest situation: The whole data can be loaded
# into the memory.
#######################################################################################

from pathlib import Path
import pickle
from typing import Any, Optional, Union
from warnings import warn

from lightning.pytorch import LightningDataModule
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_forecasting.adapters import ScalerAdapter
from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    MultiNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)
from pytorch_forecasting.data.timeseries import TimeSeries
from pytorch_forecasting.utils._coerce import _coerce_to_dict

NORMALIZER = TorchNormalizer | EncoderNormalizer | NaNLabelEncoder


class EncoderDecoderTimeSeriesDataModule(LightningDataModule):
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
        self.time_series_dataset = time_series_dataset
        self.max_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        self.max_prediction_length = max_prediction_length
        self.min_prediction_length = min_prediction_length
        self.min_prediction_idx = min_prediction_idx
        self.allow_missing_timesteps = allow_missing_timesteps
        self.add_relative_time_idx = add_relative_time_idx
        self.add_target_scales = add_target_scales
        self.add_encoder_length = add_encoder_length
        self.randomize_length = randomize_length
        self.target_normalizer = target_normalizer
        self.categorical_encoders = categorical_encoders
        self.scalers = scalers
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split

        warn(
            "EncoderDecoderTimeSeriesDataModule is part of an experimental "
            "rework of the "
            "pytorch-forecasting data layer, "
            "scheduled for release with v2.0.0. "
            "The API is not stable and may change without prior warning. "
            "For beta testing, but not for stable production use. "
            "Feedback and suggestions are very welcome in "
            "pytorch-forecasting issue 1736, "
            "https://github.com/sktime/pytorch-forecasting/issues/1736",
            UserWarning,
        )

        super().__init__()

        if isinstance(target_normalizer, str) and target_normalizer.lower() == "auto":
            self._target_normalizer = None
            self._auto_normalizer = True
        elif isinstance(target_normalizer, (tuple, list)):
            self._target_normalizer = ScalerAdapter(
                MultiNormalizer(list(target_normalizer))
            )
            self._auto_normalizer = False
        else:
            self._target_normalizer = ScalerAdapter(self.target_normalizer)
            self._auto_normalizer = False

        self.time_series_metadata = time_series_dataset.get_metadata()
        self._min_prediction_length = min_prediction_length or max_prediction_length
        self._min_encoder_length = min_encoder_length or max_encoder_length
        self._categorical_encoders = _coerce_to_dict(categorical_encoders)
        self.n_targets = len(self.time_series_metadata["cols"]["y"])

        self.categorical_indices = []
        self.continuous_indices = []
        self._metadata = None
        self._target_normalizer_fitted = False
        self._feature_scalers_fitted = False

        for idx, col in enumerate(self.time_series_metadata["cols"]["x"]):
            if self.time_series_metadata["col_type"].get(col) == "C":
                self.categorical_indices.append(idx)
            else:
                self.continuous_indices.append(idx)

        self._scalers = {
            k: ScalerAdapter(v) for k, v in _coerce_to_dict(scalers).items()
        }
        self._build_cont_scalers()

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

    @property
    def metadata(self):
        """Compute metadata for model initialization.

        This property returns a dictionary containing the shapes and key information
        related to the time series model. The metadata includes:

        * ``encoder_cat``: Number of categorical variables in the encoder.
        * ``encoder_cont``: Number of continuous variables in the encoder.
        * ``decoder_cat``: Number of categorical variables in the decoder that are
                            known in advance.
        * ``decoder_cont``:  Number of continuous variables in the decoder that are
                            known in advance.
        * ``target``: Number of target variables.

        If static features are present, the following keys are added:

        * ``static_categorical_features``: Number of static categorical features
        * ``static_continuous_features``: Number of static continuous features

        It also contains the following information:

        * ``max_encoder_length``: maximum encoder length
        * ``max_prediction_length``: maximum prediction length
        * ``min_encoder_length``: minimum encoder length
        * ``min_prediction_length``: minimum prediction length
        """
        if self._metadata is None:
            self._metadata = self._prepare_metadata()
        return self._metadata

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

    class _ProcessedEncoderDecoderDataset(Dataset):
        """PyTorch Dataset for processed encoder-decoder time series data.

        Parameters
        ----------
        data_module : EncoderDecoderTimeSeriesDataModule
            The data module handling preprocessing and metadata configuration.
        windows : List[Tuple[int, int, int, int]]
            List of window tuples containing
            (series_idx, start_idx, enc_length, pred_length).
        add_relative_time_idx : bool, default=False
            Whether to include relative time indices.
        preprocessed_data : Optional[dict[int, dict[str, Any]]], default=None
            Preprocessed data for all time series indices on input dataset.
        """

        def __init__(
            self,
            data_module: "EncoderDecoderTimeSeriesDataModule",
            windows: list[tuple[int, int, int, int]],
            preprocessed_data: dict[int, dict[str, Any]],
            add_relative_time_idx: bool = False,
        ):
            self.data_module = data_module
            self.windows = windows
            self.preprocessed_data = preprocessed_data
            self.add_relative_time_idx = add_relative_time_idx

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, idx):
            """Retrieve a processed time series window for dataloader input.

            Parameters
            ----------
            idx : int
                Index of the window to retrieve from the dataset.

            Returns
            -------
            x : dict
                Dictionary containing model inputs:

                * ``encoder_cat`` : tensor of shape (enc_length, n_cat_features)
                  Categorical features for the encoder.
                * ``encoder_cont`` : tensor of shape (enc_length, n_cont_features)
                  Continuous features for the encoder.
                * ``decoder_cat`` : tensor of shape (pred_length, n_cat_features)
                  Categorical features for the decoder.
                * ``decoder_cont`` : tensor of shape (pred_length, n_cont_features)
                  Continuous features for the decoder.
                * ``encoder_lengths`` : tensor of shape (1,)
                  Length of the encoder sequence.
                * ``decoder_lengths`` : tensor of shape (1,)
                  Length of the decoder sequence.
                * ``decoder_target_lengths`` : tensor of shape (1,)
                  Length of the decoder target sequence.
                * ``groups`` : tensor of shape (1,)
                  Group identifier for the time series instance.
                * ``encoder_time_idx`` : tensor of shape (enc_length,)
                  Time indices for the encoder sequence.
                * ``decoder_time_idx`` : tensor of shape (pred_length,)
                  Time indices for the decoder sequence.
                * ``target_past`` : torch.Tensor of shape (enc_length,)
                  Historical target values for the encoder sequence.
                * ``target_scale`` : tensor of shape (1,)
                  Scaling factor for the target values.
                * ``encoder_mask`` : tensor of shape (enc_length,)
                  Boolean mask indicating valid encoder time points.
                * ``decoder_mask`` : tensor of shape (pred_length,)
                  Boolean mask indicating valid decoder time points.

                  If static features are present, the following keys are added:

                * ``static_categorical_features`` : tensor of shape
                                                    (1, n_static_cat_features), optional
                  Static categorical features, if available.
                * ``static_continuous_features`` : tensor of shape (1, 0), optional
                  Placeholder for static continuous features (currently empty).

            y : torch.Tensor or list of torch.Tensor
                Target values for the decoder sequence.
                If ``n_targets`` > 1, a list of tensors each of shape (pred_length,)
                is returned. Otherwise, a tensor of shape (pred_length,) is returned.
            """
            series_idx, start_idx, enc_length, pred_length = self.windows[idx]
            data = self.preprocessed_data[series_idx]

            end_idx = start_idx + enc_length + pred_length
            encoder_indices = slice(start_idx, start_idx + enc_length)
            decoder_indices = slice(start_idx + enc_length, end_idx)

            target_past = data["target"][encoder_indices]

            # apply encoder normalizer on target_past.
            normalizer = self.data_module._target_normalizer
            if normalizer is not None and normalizer.fit_per_sequence:
                target_past = (
                    self.data_module._target_normalizer.fit_transform_sequence(
                        target_past
                    )
                )

            target_original_past = data["target_original"][encoder_indices]
            valid_mask = ~torch.isnan(target_original_past)
            abs_vals = target_original_past.abs().masked_fill(~valid_mask, 0.0)
            counts = valid_mask.sum(dim=0).clamp(min=1)
            target_scale_vec = abs_vals.sum(dim=0) / counts
            target_scale_vec = torch.where(
                (target_scale_vec == 0) | torch.isnan(target_scale_vec),
                torch.ones_like(target_scale_vec),
                target_scale_vec,
            )

            if self.data_module.n_targets > 1:
                target_scale = [
                    target_scale_vec[i] for i in range(self.data_module.n_targets)
                ]
            else:
                target_scale = target_scale_vec.squeeze(0)

            encoder_mask = (
                data["time_mask"][encoder_indices]
                if "time_mask" in data
                else torch.ones(enc_length, dtype=torch.bool)
            )
            decoder_mask = (
                data["time_mask"][decoder_indices]
                if "time_mask" in data
                else torch.zeros(pred_length, dtype=torch.bool)
            )

            encoder_cat = data["features"]["categorical"][encoder_indices]

            encoder_cont = data["features"]["continuous"][encoder_indices]

            # apply encoder normalizer on cont features (assuming the presence of
            # EncoderNormalizer)
            for feat_idx, adapter in self.data_module._cont_scalers:
                if adapter.fit_per_sequence:
                    encoder_cont[:, feat_idx] = adapter.fit_transform_sequence(
                        encoder_cont[:, feat_idx]
                    )

            features = data["features"]
            metadata = self.data_module.time_series_metadata

            known_cat_indices = [
                i
                for i, col in enumerate(metadata["cols"]["x"])
                if metadata["col_type"].get(col) == "C"
                and metadata["col_known"].get(col) == "K"
            ]

            known_cont_indices = [
                i
                for i, col in enumerate(metadata["cols"]["x"])
                if metadata["col_type"].get(col) == "F"
                and metadata["col_known"].get(col) == "K"
            ]

            cat_map = {
                orig_idx: i
                for i, orig_idx in enumerate(self.data_module.categorical_indices)
            }
            cont_map = {
                orig_idx: i
                for i, orig_idx in enumerate(self.data_module.continuous_indices)
            }

            mapped_known_cat_indices = [
                cat_map[idx] for idx in known_cat_indices if idx in cat_map
            ]
            mapped_known_cont_indices = [
                cont_map[idx] for idx in known_cont_indices if idx in cont_map
            ]

            decoder_cat = (
                features["categorical"][decoder_indices][:, mapped_known_cat_indices]
                if mapped_known_cat_indices
                else torch.zeros((pred_length, 0))
            )

            decoder_cont = (
                features["continuous"][decoder_indices][:, mapped_known_cont_indices]
                if mapped_known_cont_indices
                else torch.zeros((pred_length, 0))
            )

            x = {
                "encoder_cat": encoder_cat,
                "encoder_cont": encoder_cont,
                "decoder_cat": decoder_cat,
                "decoder_cont": decoder_cont,
                "encoder_lengths": torch.tensor(enc_length),
                "decoder_lengths": torch.tensor(pred_length),
                "decoder_target_lengths": torch.tensor(pred_length),
                "groups": data["group"],
                "target_past": target_past,
                "encoder_time_idx": torch.arange(enc_length),
                "decoder_time_idx": torch.arange(enc_length, enc_length + pred_length),
                "target_scale": target_scale,
                "encoder_mask": encoder_mask,
                "decoder_mask": decoder_mask,
            }
            if data["static"] is not None:
                raw_st_tensor = data.get("static")
                static_col_names = self.data_module.time_series_metadata["cols"]["st"]

                is_categorical_mask = torch.tensor(
                    [
                        self.data_module.time_series_metadata["col_type"].get(col_name)
                        == "C"
                        for col_name in static_col_names
                    ],
                    dtype=torch.bool,
                )

                is_continuous_mask = ~is_categorical_mask

                st_cat_values_for_item = raw_st_tensor[is_categorical_mask]
                st_cont_values_for_item = raw_st_tensor[is_continuous_mask]

                if st_cat_values_for_item.shape[0] > 0:
                    x["static_categorical_features"] = st_cat_values_for_item.unsqueeze(
                        0
                    )
                else:
                    x["static_categorical_features"] = torch.zeros(
                        (1, 0), dtype=torch.float32
                    )

                if st_cont_values_for_item.shape[0] > 0:
                    x["static_continuous_features"] = st_cont_values_for_item.unsqueeze(
                        0
                    )
                else:
                    x["static_continuous_features"] = torch.zeros(
                        (1, 0), dtype=torch.float32
                    )

            y = data["target"][decoder_indices]

            if y.shape[-1] > 1:
                y = [y[:, i] for i in range(y.shape[-1])]
            else:
                y = y.squeeze(-1)
            return x, y

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

    def _ensure_split(self):
        """Compute train/val/test indices once and cache them."""
        if hasattr(self, "_split_indices"):
            return

        total_series = len(self.time_series_dataset)
        self._split_indices = torch.randperm(total_series)

        self._train_size = int(self.train_val_test_split[0] * total_series)
        self._val_size = int(self.train_val_test_split[1] * total_series)

        self._train_indices = self._split_indices[: self._train_size]
        self._val_indices = self._split_indices[
            self._train_size : self._train_size + self._val_size
        ]
        self._test_indices = self._split_indices[self._train_size + self._val_size :]

    def _make_dataset(self, indices: torch.Tensor):
        """Preprocess a set of series indices into a windowed Dataset.

        Returns
        -------
        preprocessed : dict
            preprocessed dictionary of series indices.
        windows : list
            list of (series_idx, start_idx, enc_length, pred_length)
        dataset : Dataset
            dataset wrapping the windows over the preprocessed cache
        """
        preprocessed = {
            idx.item(): self._preprocess_data(idx.item()) for idx in indices
        }
        windows = self._create_windows(indices)
        dataset = self._ProcessedEncoderDecoderDataset(
            self, windows, preprocessed, self.add_relative_time_idx
        )
        return preprocessed, windows, dataset

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
            if not hasattr(self, "train_dataset") or not hasattr(self, "val_dataset"):
                self._train_preprocessed, self.train_windows, self.train_dataset = (
                    self._make_dataset(self._train_indices)
                )
                self._val_preprocessed, self.val_windows, self.val_dataset = (
                    self._make_dataset(self._val_indices)
                )

        elif stage == "test":
            if not hasattr(self, "test_dataset"):
                self._test_preprocessed, self.test_windows, self.test_dataset = (
                    self._make_dataset(self._test_indices)
                )
        elif stage == "predict":
            predict_indices = torch.arange(len(self.time_series_dataset))
            self._predict_preprocessed, self.predict_windows, self.predict_dataset = (
                self._make_dataset(predict_indices)
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
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
        if isinstance(batch[0][0]["target_scale"], list | tuple):
            num_targets = len(batch[0][0]["target_scale"])
            target_scale = [
                torch.stack([x["target_scale"][i] for x, _ in batch])
                for i in range(num_targets)
            ]
        else:
            target_scale = torch.stack([x["target_scale"] for x, _ in batch])

        x_batch["target_scale"] = target_scale

        if "static_categorical_features" in batch[0][0]:
            x_batch["static_categorical_features"] = torch.stack(
                [x["static_categorical_features"] for x, _ in batch]
            )
            x_batch["static_continuous_features"] = torch.stack(
                [x["static_continuous_features"] for x, _ in batch]
            )

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
