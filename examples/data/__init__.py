from pathlib import Path
import pandas as pd
import numpy as np


def get_stallion_data():
    return pd.read_parquet(Path(__file__).joinpath("../stallion.parquet").resolve())


def generate_ar_data(
    n_series: int = 10,
    timesteps: int = 400,
    seasonality: float = 3.0,
    noise: float = 0.0,
    level: float = 1.0,
    exp: bool = False,
):
    # sample parameters
    linear_trends = np.random.normal(size=n_series)[:, None] / timesteps
    quadratic_trends = np.random.normal(size=n_series)[:, None] / timesteps ** 2
    seasonalities = np.random.normal(size=n_series)[:, None]
    levels = level * np.random.normal(size=n_series)[:, None]

    # generate series
    x = np.arange(timesteps)[None, :]
    series = (x * linear_trends + x ** 2 * quadratic_trends) + seasonalities * np.sin(
        2 * np.pi * seasonality * x / timesteps
    )
    # add noise
    series = levels * series * (1 + noise * np.random.normal(size=series.shape))
    if exp:
        series = np.exp(series)

    # insert into dataframe
    data = (
        pd.DataFrame(series)
        .stack()
        .reset_index()
        .rename(columns={"level_0": "series", "level_1": "time_idx", 0: "value"})
    )

    return data
