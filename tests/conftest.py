import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))
sys.path.insert(0, "examples")
from data import get_stallion_data
from pytorch_forecasting import TimeSeriesDataSet


@pytest.fixture
def data_with_covariates():
    data = get_stallion_data()
    data["month"] = data.date.dt.month
    data["log_volume"] = np.log1p(data.volume)
    data["weight"] = 1 + np.sqrt(data.volume)

    data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
    data["time_idx"] -= data["time_idx"].min()

    return data


@pytest.fixture
def dataloaders_with_coveratiates(data_with_covariates):
    training_cutoff = "2016-09-01"
    max_encoder_length = 36
    max_prediction_length = 6

    training = TimeSeriesDataSet(
        data_with_covariates[lambda x: x.date < training_cutoff],
        time_idx="time_idx",
        target="volume",
        # weight="weight",
        group_ids=["agency", "sku"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["agency", "sku"],
        static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
        time_varying_known_categoricals=[
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
        ],
        time_varying_known_reals=["time_idx", "price_regular", "price_actual", "discount", "discount_in_percent",],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["volume", "log_volume", "industry_volume", "soda_volume", "avg_max_temp"],
        constant_fill_strategy={"volume": 0},
        dropout_categoricals=["sku"],
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, data_with_covariates, min_prediction_idx=training.index.time.max() + 1
    )
    batch_size = 32
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    return dict(train=train_dataloader, val=val_dataloader)


@pytest.fixture
def dataloaders_fixed_window_without_coveratiates(data_with_covariates):
    training_cutoff = "2016-09-01"
    max_encoder_length = 12
    max_prediction_length = 6

    training = TimeSeriesDataSet(
        data_with_covariates[lambda x: x.date < training_cutoff],
        time_idx="time_idx",
        target="volume",
        # weight="weight",
        group_ids=["agency", "sku"],
        max_encoder_length=max_encoder_length,
        min_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=[],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[],
        randomize_length=None,
        add_relative_time_idx=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, data_with_covariates, min_prediction_idx=training.index.time.max() + 1
    )
    batch_size = 32
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    return dict(train=train_dataloader, val=val_dataloader)
