import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting.data import TimeSeriesDataSet

def generate_demo_plot():
    # Create sample data with a gap
    data = pd.DataFrame(
        dict(
            time_idx=[0, 1, 2, 8, 9, 10],
            value=[0.0, 1.0, 2.0, 8.0, 9.0, 10.0],
            group=["A"] * 6,
        )
    )

    # Forward Fill
    ds_forward = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        max_encoder_length=10,
        max_prediction_length=1,
        min_encoder_length=10,
        allow_missing_timesteps=True,
        interpolation_strategy="forward_fill"
    )
    # Since we need 11 steps total (0-10), and max_encoder_length=10, max_prediction_length=1
    # the only valid sequence is index 0.
    x_f, y_f = ds_forward[0]
    vals_f = x_f["encoder_target"].tolist() + y_f[0].tolist()
    
    # Linear
    ds_linear = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        max_encoder_length=10,
        max_prediction_length=1,
        min_encoder_length=10,
        allow_missing_timesteps=True,
        interpolation_strategy="linear"
    )
    x_l, y_l = ds_linear[0]
    vals_l = x_l["encoder_target"].tolist() + y_l[0].tolist()

    # Zero
    ds_zero = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        max_encoder_length=10,
        max_prediction_length=1,
        min_encoder_length=10,
        allow_missing_timesteps=True,
        interpolation_strategy="zero"
    )
    x_z, y_z = ds_zero[0]
    vals_z = x_z["encoder_target"].tolist() + y_z[0].tolist()

    time_indices = np.arange(11)
    original_time = [0, 1, 2, 8, 9, 10]
    original_values = [0.0, 1.0, 2.0, 8.0, 9.0, 10.0]

    plt.figure(figsize=(10, 6))
    plt.plot(time_indices, vals_f, 'o--', label='Forward Fill (Default)', alpha=0.6)
    plt.plot(time_indices, vals_l, 's-', label='Linear Interpolation', linewidth=2)
    plt.plot(time_indices, vals_z, 'x:', label='Zero Fill', alpha=0.6)
    plt.scatter(original_time, original_values, color='red', zorder=5, label='Original Data')
    
    plt.title('Comparison of Missing Timestep Filling Strategies')
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_path = "/Users/sujanyd/.gemini/antigravity/brain/9f06a2bf-f2db-4d15-8525-6e44b2dcf310/interpolation_demo.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    generate_demo_plot()
