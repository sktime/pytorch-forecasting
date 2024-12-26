import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))  # isort:skip


from pytorch_forecasting import TimeSeriesDataSet  # isort:skip
from pytorch_forecasting.data.examples import get_stallion_data  # isort:skip


# for vscode debugging: https://stackoverflow.com/a/62563106/14121677
if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


@pytest.fixture(scope="session")
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
    data[special_days] = (
        data[special_days].apply(lambda x: x.map({0: "", 1: x.name})).astype("category")
    )

    data = data[lambda x: x.time_idx < 10]  # downsample
    return data


@pytest.fixture(scope="session")
def test_dataset(test_data):
    training = TimeSeriesDataSet(
        test_data.copy(),
        time_idx="time_idx",
        target="volume",
        time_varying_known_reals=["price_regular", "time_idx"],
        group_ids=["agency", "sku"],
        static_categoricals=["agency"],
        max_encoder_length=5,
        max_prediction_length=2,
        min_prediction_length=1,
        min_encoder_length=0,
        randomize_length=None,
    )
    return training


@pytest.fixture(autouse=True)
def disable_mps(monkeypatch):
    """Disable MPS for all tests"""
    monkeypatch.setattr("torch._C._mps_is_available", lambda: False)
