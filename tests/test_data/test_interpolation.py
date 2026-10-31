import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer


def test_interpolation_strategies():
    # Create sample data with a gap
    # time_idx: 0, 1, 2, (gap), 5, 6
    data = pd.DataFrame(
        dict(
            time_idx=[0, 1, 2, 5, 6],
            value=[0.0, 1.0, 2.0, 5.0, 6.0],
            target2=[10.0, 11.0, 12.0, 15.0, 16.0],
            real_val=[100.0, 101.0, 102.0, 105.0, 106.0],
            group=["A"] * 5,
            cat=["a", "b", "c", "f", "g"],
        )
    )

    # Use identity normalizers to compare raw values
    common_params = dict(
        time_idx="time_idx",
        target=["value", "target2"],
        time_varying_unknown_reals=["real_val"],
        group_ids=["group"],
        max_encoder_length=6,
        max_prediction_length=1,
        min_encoder_length=6,
        allow_missing_timesteps=True,
        target_normalizer=[
            EncoderNormalizer(method="identity"),
            EncoderNormalizer(method="identity"),
        ],
        scalers={"real_val": EncoderNormalizer(method="identity")},
    )

    # Expected values for time_idx 0 to 6
    expected_f_val = [0.0, 1.0, 2.0, 2.0, 2.0, 5.0, 6.0]
    expected_l_val = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    expected_z_val = [0.0, 1.0, 2.0, 0.0, 0.0, 5.0, 6.0]

    expected_f_real = [100.0, 101.0, 102.0, 102.0, 102.0, 105.0, 106.0]
    expected_l_real = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    expected_z_real = [100.0, 101.0, 102.0, 0.0, 0.0, 105.0, 106.0]

    def verify_sample(x, y, expected_val, expected_real):
        # Target 1 (value)
        actual_v1 = x["encoder_target"][0].tolist() + y[0][0].tolist()
        np.testing.assert_allclose(actual_v1, expected_val, atol=1e-5)

        # Real 1 (real_val)
        actual_r1 = x["x_cont"][:, 0].tolist()
        np.testing.assert_allclose(actual_r1, expected_real, atol=1e-5)

    # 1. Test Forward Fill (Default)
    ds_f = TimeSeriesDataSet(
        data, interpolation_strategy="forward_fill", **common_params
    )
    verify_sample(ds_f[0][0], ds_f[0][1], expected_f_val, expected_f_real)

    # 2. Test Linear Interpolation
    ds_l = TimeSeriesDataSet(data, interpolation_strategy="linear", **common_params)
    verify_sample(ds_l[0][0], ds_l[0][1], expected_l_val, expected_l_real)

    # 3. Test Zero Fill
    ds_z = TimeSeriesDataSet(data, interpolation_strategy="zero", **common_params)
    verify_sample(ds_z[0][0], ds_z[0][1], expected_z_val, expected_z_real)

    # 4. Test Mixed Strategies
    ds_m = TimeSeriesDataSet(
        data,
        interpolation_strategy={
            "value": "linear",
            "real_val": "forward_fill",
            "target2": "zero",
        },
        **common_params,
    )
    x, y = ds_m[0]
    # value: linear
    actual_v1 = x["encoder_target"][0].tolist() + y[0][0].tolist()
    np.testing.assert_allclose(actual_v1, expected_l_val, atol=1e-5)
    # target2: zero
    actual_v2 = x["encoder_target"][1].tolist() + y[0][1].tolist()
    expected_z_t2 = [10.0, 11.0, 12.0, 0.0, 0.0, 15.0, 16.0]
    np.testing.assert_allclose(actual_v2, expected_z_t2, atol=1e-5)
    # real_val: forward_fill
    actual_r1 = x["x_cont"][:, 0].tolist()
    np.testing.assert_allclose(actual_r1, expected_f_real, atol=1e-5)

    print("All interpolation strategy tests passed with full coverage!")


if __name__ == "__main__":
    test_interpolation_strategies()
