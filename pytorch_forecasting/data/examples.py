"""
Example datasets for tutorials and testing.
"""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

BASE_URL = "https://github.com/sktime/pytorch-forecasting/raw/main/examples/data/"

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

    if not full_fname.exists():
        url = BASE_URL + fname
        urlretrieve(url, full_fname)  # noqa: S310

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


def get_stallion_dummy_data(seed: int | None = 0) -> pd.DataFrame:
    """
    Small dummy dataset for testing.

    Returns:
        pd.DataFrame: data with same structure as stallion.parquet
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-01-01", periods=48, freq="ME")

    agency_list = [f"Agency_{i:02d}" for i in range(1, 61)]
    sku_list = [f"SKU_{j:02d}" for j in range(1, 36)]
    selected_pairs = [(a, s) for a in agency_list for s in sku_list]
    rng.shuffle(selected_pairs)

    first_pair_batch = [p for p in selected_pairs if p[0] == "Agency_01"]
    other_pairs = [p for p in selected_pairs if p[0] != "Agency_01"]
    selected_pairs = first_pair_batch[:2] + other_pairs
    rng.shuffle(selected_pairs)

    agencies = np.array([p[0] for p in selected_pairs])
    skus = np.array([p[1] for p in selected_pairs])
    n_rows = len(selected_pairs) * len(dates)

    volume = rng.exponential(scale=0.5, size=n_rows)
    zero_mask = rng.random(n_rows) < 0.12
    volume[zero_mask] = 0.0

    df = pd.DataFrame(
        {
            "agency": np.repeat(agencies, len(dates)),
            "sku": np.repeat(skus, len(dates)),
            "volume": volume,
            "date": np.tile(dates, len(selected_pairs)),
            "industry_volume": np.clip(
                rng.normal(5.4e8, 6.3e7, n_rows), 4.0e8, None
            ).astype(np.int64),
            "soda_volume": np.clip(
                rng.normal(8.5e8, 8.0e7, n_rows), 6.5e8, None
            ).astype(np.int64),
            "avg_max_temp": rng.normal(28.5, 4.0, n_rows),
            "price_regular": np.clip(rng.normal(1500.0, 450.0, n_rows), 100.0, 2000.0),
            "discount": rng.gamma(2.0, 5.0, n_rows),
            "avg_population_2017": np.clip(
                rng.normal(60000, 8000, n_rows), 20000, None
            ).astype(np.int64),
            "avg_yearly_household_income_2017": np.clip(
                rng.normal(35000, 5000, n_rows), 15000, None
            ).astype(np.int64),
            "timeseries": np.repeat(np.arange(len(selected_pairs)), len(dates)),
        }
    )

    df["price_actual"] = np.maximum(df["price_regular"] - df["discount"], 0.0)
    df["discount_in_percent"] = (
        df["discount"] / np.maximum(df["price_regular"], 1.0)
    ) * 100.0

    holiday_cols = [
        "easter_day",
        "good_friday",
        "new_year",
        "christmas",
        "labor_day",
        "independence_day",
        "revolution_day_memorial",
        "regional_games",
        "fifa_u_17_world_cup",
        "football_gold_cup",
        "beer_capital",
        "music_fest",
    ]
    for col in holiday_cols:
        df[col] = rng.binomial(1, 0.08, n_rows).astype(np.int64)

    df["agency"] = df["agency"].astype("category")
    df["sku"] = df["sku"].astype("category")
    df["volume"] = df["volume"].astype(np.float64)
    df["avg_max_temp"] = df["avg_max_temp"].astype(np.float64)
    df["price_regular"] = df["price_regular"].astype(np.float64)
    df["price_actual"] = df["price_actual"].astype(np.float64)
    df["discount"] = df["discount"].astype(np.float64)
    df["discount_in_percent"] = df["discount_in_percent"].astype(np.float64)

    return df


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
        level (float, optional): Level multiplier (level is a constant to be added to timeseries). Defaults to 1.0.
        exp (bool, optional): If to return exponential of timeseries values. Defaults to False.
        seed (int, optional): Random seed. Defaults to 213.

    Returns:
        pd.DataFrame: data
    """  # noqa: E501
    # sample parameters
    np.random.seed(seed)
    linear_trends = np.random.normal(size=n_series)[:, None] / timesteps
    quadratic_trends = np.random.normal(size=n_series)[:, None] / timesteps**2
    seasonalities = np.random.normal(size=n_series)[:, None]
    levels = level * np.random.normal(size=n_series)[:, None]

    # generate series
    x = np.arange(timesteps)[None, :]
    series = (
        x * linear_trends + x**2 * quadratic_trends
    ) * trend + seasonalities * np.sin(2 * np.pi * seasonality * x / timesteps)
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


def load_toydata(num_series, seq_length):
    data_list = []
    for i in range(num_series):
        x = np.arange(seq_length)
        y = np.sin(x / 5.0) + np.random.normal(scale=0.1, size=seq_length)
        category = i % 5
        static_value = np.random.rand()
        for t in range(seq_length - 1):
            data_list.append(
                {
                    "series_id": i,
                    "time_idx": t,
                    "x": y[t],
                    "y": y[t + 1],
                    "category": category,
                    "future_known_feature": np.cos(t / 10),
                    "static_feature": static_value,
                    "static_feature_cat": i % 3,
                }
            )
    data_df = pd.DataFrame(data_list)
    return data_df
