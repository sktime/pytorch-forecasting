import numpy as np
import pytest
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer, GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data, get_stallion_data

torch.manual_seed(23)


@pytest.fixture(scope="session")
def gpus():
    if torch.cuda.is_available():
        return [0]
    else:
        return 0


@pytest.fixture(scope="session")
def data_with_covariates():
    from pytorch_forecasting.tests._data_scenarios import (
        data_with_covariates as _data_with_covariates,
    )

    return _data_with_covariates()


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
    from pytorch_forecasting.tests._data_scenarios import (
        dataloaders_with_different_encoder_decoder_length as _dataloader,
    )

    return _dataloader()


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
    from pytorch_forecasting.tests._data_scenarios import (
        dataloaders_fixed_window_without_covariates as _dataloader,
    )

    return _dataloader()
