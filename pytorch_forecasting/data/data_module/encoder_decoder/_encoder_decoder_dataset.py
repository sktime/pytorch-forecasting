"""Processed encoder-decoder dataset for v2 time series datamodules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from pytorch_forecasting.data.timeseries import TimeSeries

if TYPE_CHECKING:
    from pytorch_forecasting.data.data_module.encoder_decoder._encoder_decoder_data_module import (  # noqa: E501
        EncoderDecoderTimeSeriesDataModule,
    )


class _ProcessedEncoderDecoderDataset(Dataset):
    """PyTorch Dataset for processed encoder-decoder time series data.

    Parameters
    ----------
    dataset : TimeSeries
        The base time series dataset that provides access to raw data and metadata.
    data_module : EncoderDecoderTimeSeriesDataModule
        The data module handling preprocessing and metadata configuration.
    windows : List[Tuple[int, int, int, int]]
        List of window tuples containing
        (series_idx, start_idx, enc_length, pred_length).
    add_relative_time_idx : bool, default=False
        Whether to include relative time indices.
    """

    def __init__(
        self,
        dataset: TimeSeries,
        data_module: EncoderDecoderTimeSeriesDataModule,
        windows: list[tuple[int, int, int, int]],
        add_relative_time_idx: bool = False,
    ):
        self.dataset = dataset
        self.data_module = data_module
        self.windows = windows
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
        data = self.data_module._preprocess_data(series_idx)

        end_idx = start_idx + enc_length + pred_length
        encoder_indices = slice(start_idx, start_idx + enc_length)
        decoder_indices = slice(start_idx + enc_length, end_idx)

        target_past = data["target"][encoder_indices]
        target_scale = target_past[~torch.isnan(target_past)].abs().mean()
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
                x["static_categorical_features"] = st_cat_values_for_item.unsqueeze(0)
            else:
                x["static_categorical_features"] = torch.zeros(
                    (1, 0), dtype=torch.float32
                )

            if st_cont_values_for_item.shape[0] > 0:
                x["static_continuous_features"] = st_cont_values_for_item.unsqueeze(0)
            else:
                x["static_continuous_features"] = torch.zeros(
                    (1, 0), dtype=torch.float32
                )

        y = data["target"][decoder_indices]

        if self.data_module.n_targets > 1:
            y = [t.squeeze(-1) for t in torch.split(y, 1, dim=1)]
        else:
            y = y.squeeze(-1)

        return x, y
