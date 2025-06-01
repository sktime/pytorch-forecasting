from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer, GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
from pytorch_forecasting.data.examples import generate_ar_data, get_stallion_data
from pytorch_forecasting.data.timeseries import TimeSeries

torch.manual_seed(23)


@pytest.fixture(scope="session")
def gpus():
    if torch.cuda.is_available():
        return [0]
    else:
        return 0


@pytest.fixture(scope="session")
def data_with_covariates():
    data = get_stallion_data()
    data["month"] = data.date.dt.month.astype(str)
    data["log_volume"] = np.log1p(data.volume)
    data["weight"] = 1 + np.sqrt(data.volume)

    data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
    data["time_idx"] -= data["time_idx"].min()

    # convert special days into strings
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
    data = data.astype(dict(industry_volume=float))

    # select data subset
    data = data[lambda x: x.sku.isin(data.sku.unique()[:2])][
        lambda x: x.agency.isin(data.agency.unique()[:2])
    ]

    # default target
    data["target"] = data["volume"].clip(1e-3, 1.0)

    return data


def make_dataloaders(data_with_covariates, **kwargs):
    training_cutoff = "2016-09-01"
    max_encoder_length = 4
    max_prediction_length = 3

    kwargs.setdefault("target", "volume")
    kwargs.setdefault("group_ids", ["agency", "sku"])
    kwargs.setdefault("add_relative_time_idx", True)
    kwargs.setdefault("time_varying_unknown_reals", ["volume"])

    training = TimeSeriesDataSet(
        data_with_covariates[lambda x: x.date < training_cutoff].copy(),
        time_idx="time_idx",
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        **kwargs,  # fixture parametrization
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        data_with_covariates.copy(),
        min_prediction_idx=training.index.time.max() + 1,
    )
    train_dataloader = training.to_dataloader(train=True, batch_size=2, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=2, num_workers=0)
    test_dataloader = validation.to_dataloader(train=False, batch_size=1, num_workers=0)

    return dict(train=train_dataloader, val=val_dataloader, test=test_dataloader)


@pytest.fixture(scope="session")
def data_with_covariates_v2():
    """Create synthetic time series data with all numerical features."""

    start_date = datetime(2015, 1, 1)
    end_date = datetime(2017, 12, 31)
    dates = pd.date_range(start_date, end_date, freq="M")

    agencies = [0, 1]
    skus = [0, 1]
    data_list = []

    for agency in agencies:
        for sku in skus:
            for date in dates:
                time_idx = (date.year - 2015) * 12 + date.month - 1

                volume = (
                    np.random.exponential(2)
                    + 0.1 * time_idx
                    + 0.5 * np.sin(date.month * np.pi / 6)
                )
                volume = max(0.001, volume)
                month = date.month
                year = date.year
                quarter = (date.month - 1) // 3 + 1

                seasonal_1 = np.sin(2 * np.pi * date.month / 12)
                seasonal_2 = np.cos(2 * np.pi * date.month / 12)

                agency_feature_1 = agency * 10 + np.random.normal(0, 0.1)
                agency_feature_2 = agency * 5 + np.random.normal(0, 0.1)

                sku_feature_1 = sku * 8 + np.random.normal(0, 0.1)
                sku_feature_2 = sku * 3 + np.random.normal(0, 0.1)

                trend = time_idx * 0.1
                noise = np.random.normal(0, 0.1)

                special_event_1 = 1 if date.month in [12, 1] else 0
                special_event_2 = 1 if date.month in [6, 7, 8] else 0

                data_list.append(
                    {
                        "date": date,
                        "time_idx": time_idx,
                        "agency_encoded": agency,
                        "sku_encoded": sku,
                        "volume": volume,
                        "target": volume,
                        "weight": 1.0 + np.sqrt(volume),
                        "month": month,
                        "year": year,
                        "quarter": quarter,
                        "seasonal_1": seasonal_1,
                        "seasonal_2": seasonal_2,
                        "agency_feature_1": agency_feature_1,
                        "agency_feature_2": agency_feature_2,
                        "sku_feature_1": sku_feature_1,
                        "sku_feature_2": sku_feature_2,
                        "trend": trend,
                        "noise": noise,
                        "special_event_1": special_event_1,
                        "special_event_2": special_event_2,
                        "log_volume": np.log1p(volume),
                    }
                )

    data = pd.DataFrame(data_list)

    numeric_cols = [col for col in data.columns if col != "date"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[numeric_cols] = data[numeric_cols].fillna(0)

    return data


def make_dataloaders_v2(data_with_covariates, **kwargs):
    """Create dataloaders with consistent encoder/decoder features."""

    training_cutoff = "2016-09-01"
    max_encoder_length = kwargs.get("max_encoder_length", 4)
    max_prediction_length = kwargs.get("max_prediction_length", 3)

    target_col = kwargs.get("target", "target")
    group_cols = kwargs.get("group_ids", ["agency_encoded", "sku_encoded"])
    add_relative_time_idx = kwargs.get("add_relative_time_idx", True)

    known_features = [
        "month",
        "year",
        "quarter",
        "seasonal_1",
        "seasonal_2",
        "special_event_1",
        "special_event_2",
        "trend",
    ]
    unknown_features = [
        "agency_feature_1",
        "agency_feature_2",
        "sku_feature_1",
        "sku_feature_2",
        "noise",
        "log_volume",
    ]

    numerical_features = known_features + unknown_features
    categorical_features = []
    static_features = group_cols

    for col in numerical_features + categorical_features + group_cols + [target_col]:
        if col in data_with_covariates.columns:
            data_with_covariates[col] = pd.to_numeric(
                data_with_covariates[col], errors="coerce"
            ).fillna(0)

    for col in categorical_features + group_cols:
        if col in data_with_covariates.columns:
            data_with_covariates[col] = data_with_covariates[col].astype("int64")

    if "weight" in data_with_covariates.columns:
        data_with_covariates["weight"] = pd.to_numeric(
            data_with_covariates["weight"], errors="coerce"
        ).fillna(1.0)

    training_data = data_with_covariates[
        data_with_covariates.date < training_cutoff
    ].copy()
    validation_data = data_with_covariates.copy()

    required_columns = (
        ["time_idx", target_col, "weight", "date"]
        + group_cols
        + numerical_features
        + categorical_features
    )

    available_columns = [
        col for col in required_columns if col in data_with_covariates.columns
    ]

    training_data_clean = training_data[available_columns].copy()
    validation_data_clean = validation_data[available_columns].copy()

    if "date" in training_data_clean.columns:
        training_data_clean = training_data_clean.drop("date", axis=1)
    if "date" in validation_data_clean.columns:
        validation_data_clean = validation_data_clean.drop("date", axis=1)

    training_dataset = TimeSeries(
        data=training_data_clean,
        time="time_idx",
        target=[target_col],
        group=group_cols,
        weight="weight",
        num=numerical_features,
        cat=categorical_features if categorical_features else None,
        known=known_features,
        unknown=unknown_features,
        static=static_features,
    )

    validation_dataset = TimeSeries(
        data=validation_data_clean,
        time="time_idx",
        target=[target_col],
        group=group_cols,
        weight="weight",
        num=numerical_features,
        cat=categorical_features if categorical_features else None,
        known=known_features,
        unknown=unknown_features,
        static=static_features,
    )

    training_max_time_idx = training_data["time_idx"].max() + 1

    train_datamodule = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=training_dataset,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        add_relative_time_idx=add_relative_time_idx,
        batch_size=2,
        num_workers=0,
        train_val_test_split=(0.8, 0.2, 0.0),
    )

    val_datamodule = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=validation_dataset,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        min_prediction_idx=training_max_time_idx,
        add_relative_time_idx=add_relative_time_idx,
        batch_size=2,
        num_workers=0,
        train_val_test_split=(0.0, 1.0, 0.0),
    )

    test_datamodule = EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=validation_dataset,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        min_prediction_idx=training_max_time_idx,
        add_relative_time_idx=add_relative_time_idx,
        batch_size=1,
        num_workers=0,
        train_val_test_split=(0.0, 0.0, 1.0),
    )

    train_datamodule.setup("fit")
    val_datamodule.setup("fit")
    test_datamodule.setup("test")

    train_dataloader = train_datamodule.train_dataloader()
    val_dataloader = val_datamodule.val_dataloader()
    test_dataloader = test_datamodule.test_dataloader()

    return {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
        "data_module": train_datamodule,
    }


@pytest.fixture(
    params=[
        dict(),
        dict(
            static_categoricals=["agency", "sku"],
            static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
            time_varying_known_categoricals=["special_days", "month"],
            variable_groups=dict(
                special_days=[
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
            ),
            time_varying_known_reals=[
                "time_idx",
                "price_regular",
                "price_actual",
                "discount",
                "discount_in_percent",
            ],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                "volume",
                "log_volume",
                "industry_volume",
                "soda_volume",
                "avg_max_temp",
            ],
            constant_fill_strategy={"volume": 0},
            categorical_encoders={"sku": NaNLabelEncoder(add_nan=True)},
        ),
        dict(static_categoricals=["agency", "sku"]),
        dict(randomize_length=True, min_encoder_length=2),
        dict(target_normalizer=EncoderNormalizer(), min_encoder_length=2),
        dict(target_normalizer=GroupNormalizer(transformation="log1p")),
        dict(
            target_normalizer=GroupNormalizer(
                groups=["agency", "sku"], transformation="softplus", center=False
            )
        ),
        dict(target="agency"),
        # test multiple targets
        dict(target=["industry_volume", "volume"]),
        dict(target=["agency", "volume"]),
        dict(
            target=["agency", "volume"], min_encoder_length=1, min_prediction_length=1
        ),
        dict(target=["agency", "volume"], weight="volume"),
        # test weights
        dict(target="volume", weight="volume"),
    ],
    scope="session",
)
def multiple_dataloaders_with_covariates(data_with_covariates, request):
    return make_dataloaders(data_with_covariates, **request.param)


@pytest.fixture(scope="session")
def dataloaders_with_different_encoder_decoder_length(data_with_covariates):
    return make_dataloaders(
        data_with_covariates.copy(),
        target="target",
        time_varying_known_categoricals=["special_days", "month"],
        variable_groups=dict(
            special_days=[
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
        ),
        time_varying_known_reals=[
            "time_idx",
            "price_regular",
            "price_actual",
            "discount",
            "discount_in_percent",
        ],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "target",
            "volume",
            "log_volume",
            "industry_volume",
            "soda_volume",
            "avg_max_temp",
        ],
        static_categoricals=["agency"],
        add_relative_time_idx=False,
        target_normalizer=GroupNormalizer(groups=["agency", "sku"], center=False),
    )


@pytest.fixture(scope="session")
def dataloaders_with_covariates(data_with_covariates):
    return make_dataloaders(
        data_with_covariates.copy(),
        target="target",
        time_varying_known_reals=["discount"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["agency"],
        add_relative_time_idx=False,
        target_normalizer=GroupNormalizer(groups=["agency", "sku"], center=False),
    )


@pytest.fixture(scope="session")
def dataloaders_multi_target(data_with_covariates):
    return make_dataloaders(
        data_with_covariates.copy(),
        time_varying_unknown_reals=["target", "discount"],
        target=["target", "discount"],
        add_relative_time_idx=False,
    )


@pytest.fixture(scope="session")
def dataloaders_fixed_window_without_covariates():
    return _dataloaders_fixed_window_without_covariates()


def _dataloaders_fixed_window_without_covariates():
    data = generate_ar_data(seasonality=10.0, timesteps=50, n_series=2)
    validation = data.series.iloc[:2]

    max_encoder_length = 30
    max_prediction_length = 10

    training = TimeSeriesDataSet(
        data[lambda x: ~x.series.isin(validation)],
        time_idx="time_idx",
        target="value",
        categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
        group_ids=["series"],
        static_categoricals=[],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["value"],
        target_normalizer=EncoderNormalizer(),
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        data[lambda x: x.series.isin(validation)],
        stop_randomization=True,
    )
    batch_size = 2
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )
    test_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )

    return dict(train=train_dataloader, val=val_dataloader, test=test_dataloader)
