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
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
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
    target_normalizer :
        Union[NORMALIZER, str, List[NORMALIZER], Tuple[NORMALIZER], None],
         default="auto"
        Normalizer for the target variable. If "auto", uses `RobustScaler`.

    categorical_encoders : Optional[Dict[str, NaNLabelEncoder]], default=None
        Dictionary of categorical encoders.

    scalers :
    Optional[Dict[str, Union[StandardScaler, RobustScaler,
                        TorchNormalizer, EncoderNormalizer]]], default=None
        Dictionary of feature scalers.

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
        | None = "auto",
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

        # handle defaults and derived attributes
        if isinstance(target_normalizer, str) and target_normalizer.lower() == "auto":
            self._target_normalizer = RobustScaler()
        else:
            self._target_normalizer = target_normalizer

        self.time_series_metadata = time_series_dataset.get_metadata()
        self._min_prediction_length = min_prediction_length or max_prediction_length
        self._min_encoder_length = min_encoder_length or max_encoder_length
        self._categorical_encoders = _coerce_to_dict(categorical_encoders)
        self._scalers = _coerce_to_dict(scalers)
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

    def _preprocess_data(self, series_idx: torch.Tensor) -> list[dict[str, Any]]:
        """Preprocess the data before feeding it into _ProcessedEncoderDecoderDataset.

        Preprocessing steps
        --------------------

        * Converts target (`y`) and features (`x`) to `torch.float32`.
        * Masks time points that are at or before the cutoff time.
        * Splits features into categorical and continuous subsets based on
            predefined indices.
        * Normalizes the target variable using the specified target normalizer.
        * Normalizes continuous features using the specified feature scalers.
        """
        sample = self.time_series_dataset[series_idx]

        target = sample["y"]
        features = sample["x"]
        times = sample["t"]
        cutoff_time = sample["cutoff_time"]

        time_mask = torch.tensor(times <= cutoff_time, dtype=torch.bool)

        if isinstance(target, torch.Tensor):
            target = target.float()
        else:
            target = torch.tensor(target, dtype=torch.float32)

        if isinstance(features, torch.Tensor):
            features = features.float()
        else:
            features = torch.tensor(features, dtype=torch.float32)

        # target is always made into 2D tensor before normalizing.
        # helps in generalizing to all cases - single and multi target.
        if target.ndim == 1:
            target = target.unsqueeze(-1)

        categorical = (
            features[:, self.categorical_indices]
            if self.categorical_indices
            else torch.zeros((features.shape[0], 0))
        )
        continuous = (
            features[:, self.continuous_indices]
            if self.continuous_indices
            else torch.zeros((features.shape[0], 0))
        )

        target_original = target.clone()

        if self._target_normalizer is not None and self._target_normalizer_fitted:
            normalized_target = target.clone()
            if isinstance(self._target_normalizer, list):
                for i, normalizer in enumerate(self._target_normalizer):
                    normalized_target[:, i] = normalizer.transform(target[:, i])
            elif isinstance(self._target_normalizer, TorchNormalizer):
                # single target with n_targets = 1 as the second dimension.
                target = target.squeeze(-1)
                normalized_target = self._target_normalizer.transform(target).unsqueeze(
                    -1
                )  # noqa: E501
            elif isinstance(self._target_normalizer, (StandardScaler, RobustScaler)):
                target_np = target.detach().numpy()
                target_np = self._target_normalizer.transform(target_np)
                normalized_target = torch.tensor(target_np, dtype=torch.float32)
            target = normalized_target

        # applying feature scalers.
        if self._feature_scalers_fitted and self.continuous_indices:
            normalized_cont = continuous.clone()
            feature_names = [
                self.time_series_metadata["cols"]["x"][idx]
                for idx in self.continuous_indices
            ]

            for feat_idx, feat_name in enumerate(feature_names):
                if feat_name in self._scalers:
                    scaler = self._scalers[feat_name]
                    feature_data = continuous[:, feat_idx]

                    if isinstance(scaler, (TorchNormalizer, EncoderNormalizer)):
                        normalized_cont[:, feat_idx] = scaler.transform(feature_data)
                    elif isinstance(scaler, (StandardScaler, RobustScaler)):
                        feature_np = feature_data.numpy()
                        feature_np = scaler.transform(
                            feature_np.reshape(-1, 1)
                        ).reshape(-1)  # noqa: E501
                        normalized_cont[:, feat_idx] = torch.tensor(
                            feature_np, dtype=torch.float32
                        )  # noqa: E501
            continuous = normalized_cont

        return {
            "features": {"categorical": categorical, "continuous": continuous},
            "target": target,
            "target_original": target_original,
            "static": sample.get("st", None),
            "group": sample.get("group", torch.tensor([0])),
            "length": len(target),
            "time_mask": time_mask,
            "times": times,
            "cutoff_time": cutoff_time,
        }

    def _fit_target_normalizer(self, train_indices):
        """Fit target normalizer on the target variable's training data."""

        if self._target_normalizer is None:
            return

        if isinstance(self._target_normalizer, EncoderNormalizer):
            # Encoder normalizer does not need fitting on global data
            return

        all_targets = []
        for idx in train_indices:
            sample = self.time_series_dataset[idx]
            target = sample["y"]
            if isinstance(target, torch.Tensor):
                all_targets.append(target)
            else:
                all_targets.append(torch.tensor(target, dtype=torch.float32))

        if not all_targets:
            return

        all_targets = torch.cat(all_targets, dim=0)

        if isinstance(self._target_normalizer, TorchNormalizer):
            # handle multiple targets (in case).
            if all_targets.ndim > 1 and all_targets.shape[1] > 1:
                self._target_normalizer = [
                    TorchNormalizer() for _ in range(all_targets.shape[1])
                ]
                for i, normalizer in enumerate(self._target_normalizer):
                    normalizer.fit(all_targets[:, i])
            else:
                if all_targets.ndim > 1 and all_targets.shape[1] == 1:
                    all_targets = all_targets.squeeze(-1)
                self._target_normalizer.fit(all_targets)
        elif isinstance(self._target_normalizer, (StandardScaler, RobustScaler)):
            all_targets_np = all_targets.detach().numpy()
            if all_targets_np.ndim == 1:
                all_targets_np = all_targets_np.reshape(-1, 1)
            self._target_normalizer.fit(all_targets_np)

        self._target_normalizer_fitted = True

    def _fit_scalers(self, train_indices):
        """Fit scalers on continuous features in the training data."""

        if not self._scalers or not self.continuous_indices:
            return

        features_to_scale = {
            self.time_series_metadata["cols"]["x"][idx]: pos
            for pos, idx in enumerate(self.continuous_indices)
        }

        for feat_name, scaler in self._scalers.items():
            if feat_name not in features_to_scale:
                continue
            feat_idx = features_to_scale[feat_name]
            feat_data = []

            for idx in train_indices:
                sample = self.time_series_dataset[idx]
                feature_data = sample["x"][:, feat_idx]

                if not isinstance(feature_data, torch.Tensor):
                    feature_data = torch.tensor(feature_data, dtype=torch.float32)

                feat_data.append(feature_data)
            feat_data = torch.cat(feat_data, dim=0)

            if isinstance(scaler, (TorchNormalizer)):
                scaler.fit(feat_data)
            elif isinstance(scaler, (StandardScaler, RobustScaler)):
                feat_data_np = feat_data.detach().numpy()
                scaler.fit(feat_data_np.reshape(-1, 1))
        self._feature_scalers_fitted = True

    def _apply_encoder_normalizer_on_target(self, encoder_target: torch.Tensor):
        """Apply encoder normalizer on the target variable's
        current encoding sequence.

        This function fits and transforms the current target encoding
        sequence using the EncoderNormalizer (if applicable). EncoderNormalizers
        are expected to be fitted on-the-fly for each sequence, not on the whole data.

        Parameters
        ----------
        encoder_target : torch.Tensor
            The target values for the encoder sequence.

        Returns
        -------
        torch.Tensor
            The normalized target values for the encoder sequence.
        """

        if isinstance(self._target_normalizer, EncoderNormalizer):
            encoder_normalizer = self._target_normalizer
            if encoder_target.ndim == 2 and encoder_target.shape[1] == 1:
                encoder_target = encoder_target.squeeze(-1)
                encoder_normalizer.fit(encoder_target)
                normalized_target = encoder_normalizer.transform(encoder_target)
                encoder_target = normalized_target.unsqueeze(-1)
            else:
                encoder_normalizer.fit(encoder_target)
                encoder_target = encoder_normalizer.transform(encoder_target)

        elif isinstance(self._target_normalizer, list):
            # Handle list of normalizers (one per target)
            for i, normalizer in enumerate(self._target_normalizer):
                if isinstance(normalizer, EncoderNormalizer):
                    target_col = encoder_target[:, i]
                    normalizer.fit(target_col)
                encoder_target[:, i] = normalizer.transform(target_col)

        self._target_normalizer_fitted = True
        return encoder_target

    def _apply_encoder_normalizer_on_feat(
        self,
        encoder_cont: torch.Tensor,
    ):
        """
        Apply encoder normalizers on continuous features of the current
        encoding sequence (if applicable to the feature).

        Parameters
        ----------
        encoder_cont : torch.Tensor
            Continuous features for the encoder sequence.

        Returns
        -------
        torch.Tensor
            Normalized continuous features for the encoder sequence.
        """

        feature_names = [
            self.time_series_metadata["cols"]["x"][idx]
            for idx in self.continuous_indices
        ]

        for feat_idx, feat_name in enumerate(feature_names):
            if feat_name in self._scalers:
                scaler = self._scalers[feat_name]
                if isinstance(scaler, EncoderNormalizer):
                    feature_data = encoder_cont[:, feat_idx]
                    scaler.fit(feature_data)
                    encoder_cont[:, feat_idx] = scaler.transform(feature_data)

        return encoder_cont

    def _preprocess_all_data(self, indices: torch.Tensor) -> dict[dict[str, Any]]:
        """Preprocess all data samples for given indices.

        Parameters
        ----------
        indices : torch.Tensor
            Tensor of indices specifying which samples to preprocess.

        Returns
        -------
        dict[int, dict[str, Any]]
            A dictionary mapping series indices to dictionaries containing preprocessed
            data for each sample.
        """
        preprocessed_data = {}
        for idx in indices:
            series_idx = idx.item()
            preprocessed_data[series_idx] = self._preprocess_data(series_idx)
        return preprocessed_data

    def _validate_preprocessing(self):
        """Validate preprocessing by checking if scalers and normalizers are fitted
        on training data."""

        if self._target_normalizer and not self._target_normalizer_fitted:  # noqa: E501
            raise RuntimeError(
                "Cannot setup test stage: target_normalizer is configured "
                "but not fitted. You must call setup('fit') first on this"
                " DataModule instance or use the same DataModule instance "
                "that was used for training."
            )

        if self._scalers and not self._feature_scalers_fitted:  # noqa: E501
            raise RuntimeError(
                "Cannot setup test stage: feature scalers are configured "
                "but not fitted. You must call setup('fit') first on this "
                "DataModule instance or use the same DataModule instance "
                "that was used for training."
            )

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
            if not self.data_module._target_normalizer_fitted:
                target_past = self.data_module._apply_encoder_normalizer_on_target(
                    target_past
                )

            target_original_past = data["target_original"][encoder_indices]
            target_scale = (
                target_original_past[~torch.isnan(target_original_past)].abs().mean()
            )  # noqa: E501
            if torch.isnan(target_scale) or target_scale == 0:
                target_scale = torch.tensor(1.0)

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
            encoder_cont = self.data_module._apply_encoder_normalizer_on_feat(
                encoder_cont
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

    def save_scalers(self, path: str | Path):
        """Save fitted scalers and normalizers to disk.

        Parameters
        ----------
        path: str or Path
            File path to save the scalers and normalizers.
        """

        save_state = {
            "target_normalizer": self._target_normalizer,
            "target_normalizer_fitted": self._target_normalizer_fitted,
            "feature_scalers": self._scalers,
            "feature_scalers_fitted": self._feature_scalers_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(save_state, f)

    def load_scalers(self, path: str | Path):
        """Load fitted scalers and normalizers from disk.

        Parameters
        ----------
        path: str or Path
            File path to load the scalers and normalizers from.
        """

        with open(path, "rb") as f:
            load_state = pickle.load(f)  # noqa: S301

        loaded_target_normalizer = load_state["target_normalizer"]
        # check if target normalizer matches
        if self._target_normalizer is None and loaded_target_normalizer is not None:
            raise ValueError(
                "Loaded target normalizer is not None, but no target normalizer "
                "is configured in this DataModule."
            )

        if self._target_normalizer is not None and loaded_target_normalizer is None:
            raise ValueError(
                "No target normalizer found in loaded state, but this DataModule "
                "expects a fitted target normalizer when loading from a saved state."
            )

        # filter unexpected features for scaling
        loaded_feature_scalers = load_state["feature_scalers"]
        unexpected_keys = set(loaded_feature_scalers.keys()) - set(self._scalers.keys())  # noqa: E501
        if unexpected_keys:
            raise ValueError(
                f"Loaded scalers contain unexpected feature keys: {unexpected_keys}. "
                f"Expected keys: {set(self._scalers.keys())}"
            )

        # missing keys check
        missing_keys = set(self._scalers.keys()) - set(loaded_feature_scalers.keys())
        if missing_keys:
            raise ValueError(
                f"Loaded scalers are missing expected feature keys: {missing_keys}. "
                f"Loaded keys: {set(loaded_feature_scalers.keys())}"
            )

        for key, loaded_scaler in loaded_feature_scalers.items():
            if key in self._scalers and not isinstance(
                loaded_scaler, self._scalers[key].__class__
            ):
                raise TypeError(
                    f"Loaded scaler for '{key}' has type {type(loaded_scaler).__name__} "  # noqa: E501
                    f"but configured scaler has type {type(self._scalers[key]).__name__}"  # noqa: E501
                )
        self._target_normalizer = load_state["target_normalizer"]
        self._target_normalizer_fitted = load_state["target_normalizer_fitted"]
        self._scalers = load_state["feature_scalers"]
        self._feature_scalers_fitted = load_state["feature_scalers_fitted"]

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
        total_series = len(self.time_series_dataset)
        self._split_indices = torch.randperm(total_series)

        self._train_size = int(self.train_val_test_split[0] * total_series)
        self._val_size = int(self.train_val_test_split[1] * total_series)

        self._train_indices = self._split_indices[: self._train_size]
        self._val_indices = self._split_indices[
            self._train_size : self._train_size + self._val_size
        ]
        self._test_indices = self._split_indices[self._train_size + self._val_size :]

        if stage is None or stage == "fit":
            if not self._target_normalizer_fitted:
                self._fit_target_normalizer(self._train_indices)
            if not self._feature_scalers_fitted:
                self._fit_scalers(self._train_indices)
            if not hasattr(self, "train_dataset") or not hasattr(self, "val_dataset"):
                self._train_preprocessed = self._preprocess_all_data(
                    self._train_indices
                )
                self._val_preprocessed = self._preprocess_all_data(self._val_indices)

                self.train_windows = self._create_windows(self._train_indices)
                self.val_windows = self._create_windows(self._val_indices)

                self.train_dataset = self._ProcessedEncoderDecoderDataset(
                    self,
                    self.train_windows,
                    self._train_preprocessed,
                    self.add_relative_time_idx,
                )
                self.val_dataset = self._ProcessedEncoderDecoderDataset(
                    self,
                    self.val_windows,
                    self._val_preprocessed,
                    self.add_relative_time_idx,
                )

        elif stage == "test":
            if not hasattr(self, "test_dataset"):
                self._validate_preprocessing()
                self._test_preprocessed = self._preprocess_all_data(self._test_indices)
                self.test_windows = self._create_windows(self._test_indices)
                self.test_dataset = self._ProcessedEncoderDecoderDataset(
                    self,
                    self.test_windows,
                    self._test_preprocessed,
                    self.add_relative_time_idx,
                )
        elif stage == "predict":
            self._validate_preprocessing()
            predict_indices = torch.arange(len(self.time_series_dataset))
            self._predict_preprocessed = self._preprocess_all_data(predict_indices)
            self.predict_windows = self._create_windows(predict_indices)
            self.predict_dataset = self._ProcessedEncoderDecoderDataset(
                self,
                self.predict_windows,
                self._predict_preprocessed,
                self.add_relative_time_idx,
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
            "target_scale": torch.stack([x["target_scale"] for x, _ in batch]),
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
