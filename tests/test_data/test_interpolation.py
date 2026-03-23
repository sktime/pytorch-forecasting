import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeriesDataSet


def test_interpolation_strategies():
    # Create sample data with a gap
    # time_idx: 0, 1, 2, (gap), 5, 6
    data = pd.DataFrame(
        dict(
            time_idx=[0, 1, 2, 5, 6],
            value=[0.0, 1.0, 2.0, 5.0, 6.0],
            group=["A"] * 5,
            cat=["a", "b", "c", "f", "g"],
        )
    )

    # 1. Test Forward Fill (Default)
    dataset_forward = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        max_encoder_length=6,
        max_prediction_length=1,
        min_encoder_length=6,
        allow_missing_timesteps=True,
        interpolation_strategy="forward_fill",
    )

    x, y = dataset_forward[0]
    # x["encoder_target"] is a tensor of length 6
    # y[0] is a tensor of length 1
    actual_values = x["encoder_target"].tolist() + y[0].tolist()
    expected_forward = [0.0, 1.0, 2.0, 2.0, 2.0, 5.0, 6.0]
    np.testing.assert_allclose(actual_values, expected_forward, atol=1e-5)

    # 2. Test Linear Interpolation
    dataset_linear = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        max_encoder_length=6,
        max_prediction_length=1,
        min_encoder_length=6,
        allow_missing_timesteps=True,
        interpolation_strategy="linear",
    )
    x_linear, y_linear = dataset_linear[0]
    actual_linear = x_linear["encoder_target"].tolist() + y_linear[0].tolist()
    expected_linear = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    np.testing.assert_allclose(actual_linear, expected_linear, atol=1e-5)

    # 3. Test Zero Fill
    dataset_zero = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        max_encoder_length=6,
        max_prediction_length=1,
        min_encoder_length=6,
        allow_missing_timesteps=True,
        interpolation_strategy="zero",
    )
    x_zero, y_zero = dataset_zero[0]
    actual_zero = x_zero["encoder_target"].tolist() + y_zero[0].tolist()
    expected_zero = [0.0, 1.0, 2.0, 0.0, 0.0, 5.0, 6.0]
    np.testing.assert_allclose(actual_zero, expected_zero, atol=1e-5)

    # 4. Test Mixed Strategies (Dictionary)
    dataset_mixed = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        max_encoder_length=6,
        max_prediction_length=1,
        min_encoder_length=6,
        allow_missing_timesteps=True,
        interpolation_strategy={"value": "linear", "cat": "forward_fill"},
    )
    x_mixed, y_mixed = dataset_mixed[0]
    actual_mixed = x_mixed["encoder_target"].tolist() + y_mixed[0].tolist()
    expected_mixed = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    np.testing.assert_allclose(actual_mixed, expected_mixed, atol=1e-5)

    # Verify categorical is forward filled (it should be since it's not real-valued
    # and we don't interpolate cats)
    # Cat values for indices [0, 1, 2, gap, gap, 3, 4] where gap is repetition of 2.
    # Original cats: ["a", "b", "c", "f", "g"]
    # Forward filled: ["a", "b", "c", "c", "c", "f", "g"]
    # We need to check encoder_cat and decoder_cat
    # In TimeSeriesDataSet, cats are encoded. "c" is index 2 (usually).


if __name__ == "__main__":
    test_interpolation_strategies()
