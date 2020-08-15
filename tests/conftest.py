import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))
sys.path.insert(0, "examples")

from examples.data import get_stallion_data
from pytorch_forecasting import TimeSeriesDataSet


@pytest.fixture
def test_data():
    data = get_stallion_data()
    data["month"] = data.date.dt.month.astype(str)
    data["log_volume"] = np.log1p(data.volume)
    data["weight"] = 1 + np.sqrt(data.volume)

    data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
    data["time_idx"] -= data["time_idx"].min()

    special_days = [
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
    data[special_days] = data[special_days].apply(lambda x: x.map({0: "", 1: x.name})).astype("category")

    data = data[lambda x: x.time_idx < 10]  # downsample
    return data


@pytest.fixture
def test_dataset(test_data):
    training = TimeSeriesDataSet(
        test_data,
        time_idx="time_idx",
        target="volume",
        time_varying_known_reals=["price_regular"],
        group_ids=["agency", "sku"],
        static_categoricals=["agency"],
        max_encoder_length=5,
        max_prediction_length=2,
        randomize_length=None,
    )
    return training
