import numpy as np
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer, GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data, get_stallion_data

torch.manual_seed(23)


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


def dataloaders_with_different_encoder_decoder_length():
    return make_dataloaders(
        data_with_covariates(),
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


def dataloaders_with_covariates():
    return make_dataloaders(
        data_with_covariates(),
        target="target",
        time_varying_known_reals=["discount"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["agency"],
        add_relative_time_idx=False,
        target_normalizer=GroupNormalizer(groups=["agency", "sku"], center=False),
    )


def dataloaders_multi_target():
    return make_dataloaders(
        data_with_covariates(),
        time_varying_unknown_reals=["target", "discount"],
        target=["target", "discount"],
        add_relative_time_idx=False,
    )


def dataloaders_fixed_window_without_covariates():
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
