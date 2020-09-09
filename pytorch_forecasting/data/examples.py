"""
Example datasets for tutorials and testing.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://raw.github.com/jdb78/pytorch-forecasting/master/examples/data/"

DATA_PATH = Path(__file__).parent


def _get_data_by_filename(fname: str) -> Path:
    """
    Download file or used cached version.

    Args:
        fname (str): name of file to download

    Returns:
        Path: path at which file lives
    """
    full_fname = DATA_PATH.joinpath(fname)

    # check if file exists - download if necessary
    if not full_fname.exists():
        url = BASE_URL + fname
        download = requests.get(url, allow_redirects=True)
        with open(full_fname, "wb") as file:
            file.write(download.content)

    return full_fname


def get_stallion_data() -> pd.DataFrame:
    """
    Demand data with covariates.

    ~20k samples of 350 timeseries. Important columns

    * Timeseries can be identified by ``agency`` and ``sku``.
    * ``volume`` is the demand
    * ``date`` is the month of the demand.

    Returns:
        pd.DataFrame: data
    """
    fname = _get_data_by_filename("stallion.parquet")
    return pd.read_parquet(fname)


def generate_ar_data(
    n_series: int = 10,
    timesteps: int = 400,
    seasonality: float = 3.0,
    trend: float = 3.0,
    noise: float = 0.1,
    level: float = 1.0,
    exp: bool = False,
    seed: int = 213,
) -> pd.DataFrame:
    """
    Generate multivariate data without covariates.

    Eeach timeseries is generated from seasonality and trend. Important columns:

    * ``series``: series ID
    * ``time_idx``: time index
    * ``value``: target value

    Args:
        n_series (int, optional): Number of series. Defaults to 10.
        timesteps (int, optional): Number of timesteps. Defaults to 400.
        seasonality (float, optional): Normalized frequency, i.e. frequency is ``seasonality / timesteps``.
            Defaults to 3.0.
        trend (float, optional): Trend multiplier (seasonality is multiplied with 1.0). Defaults to 3.0.
        noise (float, optional): Level of gaussian noise. Defaults to 0.1.
        level (float, optional): Level multiplier (level is a constant to be aded to timeseries). Defaults to 1.0.
        exp (bool, optional): If to return exponential of timeseries values. Defaults to False.
        seed (int, optional): Random seed. Defaults to 213.

    Returns:
        pd.DataFrame: data
    """
    # sample parameters
    np.random.seed(seed)
    linear_trends = np.random.normal(size=n_series)[:, None] / timesteps
    quadratic_trends = np.random.normal(size=n_series)[:, None] / timesteps ** 2
    seasonalities = np.random.normal(size=n_series)[:, None]
    levels = level * np.random.normal(size=n_series)[:, None]

    # generate series
    x = np.arange(timesteps)[None, :]
    series = (x * linear_trends + x ** 2 * quadratic_trends) * trend + seasonalities * np.sin(
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
