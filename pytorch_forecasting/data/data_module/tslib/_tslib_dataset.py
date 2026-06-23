"""Processed tslib-format dataset for v2 time series datamodules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

from pytorch_forecasting.data.timeseries._timeseries_v2 import TimeSeries

if TYPE_CHECKING:
    from pytorch_forecasting.data.data_module.tslib._tslib_data_module import (
        TslibDataModule,
    )


class _TslibDataset(Dataset):
    """Dataset class for `tslib` time series dataset.

    Parameters
    ----------
    dataset : TimeSeries
        The time series dataset to be used for training and validation.
    data_module : TslibDataModule
        The data module that contains the metadata and other configurations for the
        dataset.
    windows: list[tuple[int, int, int, int]]
        A list of tuples where each tuple contains:
            - series_idx: Index of time series in the dataset
            - start_idx: Start index of the window
            - context_length: Length of the context/encoder window
            - prediction_length: Length of the prediction/decoder window
    add_relative_time_idx: bool
        Whether to add relative time index to the dataset.
    """

    def __init__(
        self,
        dataset: TimeSeries,
        data_module: TslibDataModule,
        windows: list[tuple[int, int, int, int]],
        add_relative_time_idx: bool = False,
    ):
        self.dataset = dataset
        self.data_module = data_module
        self.windows = windows
        self.add_relative_time_idx = add_relative_time_idx

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[dict[str, Any], torch.Tensor | list]:
        """Get the processed dataset item at the given index.

        Parameters
        ----------
        idx : int
            The index of the dataset item to be retrieved.

        Returns
        -------
        x : dict[str, torch.Tensor]
            Dict containing processed inputs for the model, with the following keys:

            * ``history_cont`` : torch.Tensor of shape
                                    (context_length, n_history_cont_features)
                Continuous features for the encoder (historical data).
            * ``history_cat`` : torch.Tensor of shape
                                    (context_length, n_history_cat_features)
                Categorical features for the encoder (historical data).
            * ``future_cont`` : torch.Tensor of shape
                                    (prediction_length, n_future_cont_features)
                Known continuous features for the decoder (future data).
            * ``future_cat`` : torch.Tensor of shape
                                    (prediction_length, n_future_cat_features)
                Known categorical features for the decoder (future data).
            * ``history_length`` : torch.Tensor of shape (1,)
                Length of the encoder sequence.
            * ``future_length`` : torch.Tensor of shape (1,)
                Length of the decoder sequence.
            * ``history_mask`` : torch.Tensor of shape (context_length,)
                Boolean mask indicating valid encoder time points.
            * ``future_mask`` : torch.Tensor of shape (prediction_length,)
                Boolean mask indicating valid decoder time points.
            * ``groups`` : torch.Tensor of shape (1,)
                Group identifier for the time series instance.
            * ``history_time_idx`` : torch.Tensor of shape (context_length,)
                Time indices for the encoder sequence.
            * ``future_time_idx`` : torch.Tensor of shape (prediction_length,)
                Time indices for the decoder sequence.
            * ``history_target`` : torch.Tensor of shape (context_length,)
                Historical target values for the encoder sequence.
            * ``future_target`` : torch.Tensor of shape (prediction_length,)
                Target values for the decoder sequence.
            * ``future_target_len`` : torch.Tensor of shape (1,)
                Length of the decoder target sequence.

            Optional fields, depending on dataset configuration:

            * ``history_relative_time_idx`` : torch.Tensor of shape (context_length,),
                                                optional
                Relative time indices for the encoder sequence, present if
                `add_relative_time_idx` is True.
            * ``future_relative_time_idx`` : torch.Tensor of shape (prediction_length,),
                                                optional
                Relative time indices for the decoder sequence, present if
                `add_relative_time_idx` is True.
            * ``static_categorical_features`` : torch.Tensor of shape
                                                (1, n_static_features), optional
                Static categorical features if available.
            * ``static_continuous_features`` : torch.Tensor of shape
                                                (1, n_static_features), optional
                Static continuous features if available.
            * ``target_scale`` : torch.Tensor of shape (1,), optional
                Scaling factor for the target values if provided by the dataset.

        y : torch.Tensor or list of torch.Tensor
            Target values for the decoder sequence.
            If ``n_targets`` > 1, a list of tensors each of shape (prediction_length,)
            is returned. Otherwise, a tensor of shape (prediction_length,) is returned.
        """
        series_idx, start_idx, context_length, prediction_length = self.windows[idx]

        processed_data = self.data_module._preprocess_data(series_idx)

        continuous_features = processed_data["features"]["continuous"]
        categorical_features = processed_data["features"]["categorical"]

        end_idx = start_idx + context_length + prediction_length
        history_indices = slice(start_idx, start_idx + context_length)
        future_indices = slice(start_idx + context_length, end_idx)

        metadata = self.data_module.metadata

        history_cont = continuous_features[history_indices]
        history_cat = categorical_features[history_indices]

        future_cont = continuous_features[future_indices]
        future_cat = categorical_features[future_indices]

        known_features = set(metadata["feature_names"]["known"])
        continuous_feature_names = metadata["feature_names"]["continuous"]
        categorical_feature_names = metadata["feature_names"]["categorical"]

        cont_known_mask = torch.tensor(
            [feat in known_features for feat in continuous_feature_names],
            dtype=torch.bool,
        )

        cat_known_mask = torch.tensor(
            [feat in known_features for feat in categorical_feature_names],
            dtype=torch.bool,
        )

        future_cont = (
            future_cont[:, cont_known_mask]
            if len(cont_known_mask) > 0
            else torch.zeros((future_cont.shape[0], 0))
        )
        future_cat = (
            future_cat[:, cat_known_mask]
            if len(cat_known_mask) > 0
            else torch.zeros((future_cat.shape[0], 0))
        )

        history_mask = (
            processed_data["time_mask"][history_indices]
            if "time_mask" in processed_data
            else torch.ones(context_length, dtype=torch.bool)
        )

        future_mask = (
            processed_data["time_mask"][future_indices]
            if "time_mask" in processed_data
            else torch.ones(prediction_length, dtype=torch.bool)
        )

        history_target = processed_data["target"][history_indices]
        future_target = processed_data["target"][future_indices]

        x = {
            "history_cont": history_cont,
            "history_cat": history_cat,
            "future_cont": future_cont,
            "future_cat": future_cat,
            "history_length": torch.tensor(context_length),
            "future_length": torch.tensor(prediction_length),
            "history_mask": history_mask,
            "future_mask": future_mask,
            "groups": processed_data["group"],
            "history_time_idx": torch.arange(context_length),
            "future_time_idx": torch.arange(
                context_length, context_length + prediction_length
            ),
            "history_target": history_target,
            "future_target": future_target,
            "future_target_len": torch.tensor(prediction_length),
        }

        if self.add_relative_time_idx:
            x["history_relative_time_idx"] = torch.arange(-context_length, 0)
            x["future_relative_time_idx"] = torch.arange(0, prediction_length)

        if processed_data["static"] is not None:
            x["static_categorical_features"] = processed_data["static"].unsqueeze(0)
            x["static_continuous_features"] = processed_data["static"].unsqueeze(0)

        if "target_scale" in processed_data:
            x["target_scale"] = processed_data["target_scale"]

        y = processed_data["target"][future_indices]
        if self.data_module.n_targets > 1:
            y = [t.squeeze(-1) for t in torch.split(y, 1, dim=1)]
        else:
            y = y.squeeze(-1)

        return x, y
