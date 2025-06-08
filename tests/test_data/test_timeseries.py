from copy import deepcopy
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data.sampler import SequentialSampler

from pytorch_forecasting.data import (
    EncoderNormalizer,
    GroupNormalizer,
    NaNLabelEncoder,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data.encoders import MultiNormalizer, TorchNormalizer
from pytorch_forecasting.data.timeseries import _find_end_indices
from pytorch_forecasting.utils import to_list


def test_find_end_indices():
    diffs = np.array([1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1])
    max_lengths = np.array(
        [4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1]
    )

    ends, missings = _find_end_indices(diffs, max_lengths, min_length=3)
    ends_test = np.array(
        [
            3,
            4,
            4,
            5,
            6,
            8,
            9,
            10,
            10,
            10,
            10,
            14,
            15,
            15,
            16,
            17,
            19,
            20,
            21,
            21,
            21,
            21,
        ]
    )
    missings_test = np.array([[0, 2], [5, 7], [11, 13], [16, 18]])
    np.testing.assert_array_equal(ends, ends_test)
    np.testing.assert_array_equal(missings, missings_test)


def test_raise_short_encoder_length(test_data):
    with pytest.warns(UserWarning):
        test_data = test_data[
            lambda x: ~(
                (x.agency == "Agency_22") & (x.sku == "SKU_01") & (x.time_idx > 3)
            )
        ]
        TimeSeriesDataSet(
            test_data,
            time_idx="time_idx",
            target="volume",
            group_ids=["agency", "sku"],
            max_encoder_length=5,
            max_prediction_length=2,
            min_prediction_length=1,
            min_encoder_length=5,
        )


def test_categorical_target(test_data):
    dataset = TimeSeriesDataSet(
        test_data,
        time_idx="time_idx",
        target="agency",
        group_ids=["agency", "sku"],
        max_encoder_length=5,
        max_prediction_length=2,
        min_prediction_length=1,
        min_encoder_length=1,
    )
    _, y = next(iter(dataset.to_dataloader()))
    assert y[0].dtype is torch.long, "target must be of type long"


def test_pickle(test_dataset):
    pickle.dumps(test_dataset)
    pickle.dumps(test_dataset.to_dataloader())


def check_dataloader_output(dataset: TimeSeriesDataSet, out: dict[str, torch.Tensor]):
    x, y = out

    assert isinstance(y, tuple), "y output should be tuple of wegith and target"

    # check for nans and finite
    for k, v in x.items():
        for vi in to_list(v):
            assert torch.isfinite(vi).all(), f"Values for {k} should be finite"
            assert not torch.isnan(vi).any(), f"Values for {k} should not be nan"

    # check weight
    assert y[1] is None or isinstance(
        y[1], torch.Tensor
    ), "weights should be none or tensor"
    if isinstance(y[1], torch.Tensor):
        assert torch.isfinite(y[1]).all(), "Values for weight should be finite"
        assert not torch.isnan(y[1]).any(), "Values for weight should not be nan"

    # check target
    for targeti in to_list(y[0]):
        assert torch.isfinite(targeti).all(), "Values for target should be finite"
        assert not torch.isnan(targeti).any(), "Values for target should not be nan"

    # check shape
    assert x["encoder_cont"].size(2) == len(dataset.reals)
    assert x["encoder_cat"].size(2) == len(dataset.flat_categoricals)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(min_encoder_length=0, max_prediction_length=2),
        dict(static_categoricals=["agency", "sku"]),
        dict(static_reals=["avg_population_2017", "avg_yearly_household_income_2017"]),
        dict(time_varying_known_categoricals=["month"]),
        dict(
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
        ),
        dict(
            time_varying_known_reals=[
                "time_idx",
                "price_regular",
                "discount_in_percent",
            ]
        ),
        dict(
            time_varying_unknown_reals=[
                "volume",
                "log_volume",
                "industry_volume",
                "soda_volume",
                "avg_max_temp",
            ]
        ),
        dict(
            target_normalizer=GroupNormalizer(
                groups=["agency", "sku"],
                transformation="log1p",
                scale_by_group=True,
            )
        ),
        dict(target_normalizer=EncoderNormalizer(), min_encoder_length=2),
        dict(randomize_length=True, min_encoder_length=2, min_prediction_length=1),
        dict(predict_mode=True),
        dict(add_target_scales=True),
        dict(add_encoder_length=True),
        dict(add_encoder_length=True),
        dict(add_relative_time_idx=True),
        dict(weight="volume"),
        dict(
            scalers=dict(time_idx=GroupNormalizer(), price_regular=StandardScaler()),
            categorical_encoders=dict(month=NaNLabelEncoder()),
            time_varying_known_categoricals=["month"],
            time_varying_known_reals=["time_idx", "price_regular"],
        ),
        dict(
            categorical_encoders={"month": NaNLabelEncoder(add_nan=True)},
            time_varying_known_categoricals=["month"],
        ),
        dict(constant_fill_strategy=dict(volume=0.0), allow_missing_timesteps=True),
        dict(target_normalizer=None),
    ],
)
def test_TimeSeriesDataSet(test_data, kwargs):
    defaults = dict(
        time_idx="time_idx",
        target="volume",
        group_ids=["agency", "sku"],
        max_encoder_length=5,
        max_prediction_length=2,
    )
    defaults.update(kwargs)
    kwargs = defaults

    if kwargs.get("allow_missing_timesteps", False):
        np.random.seed(2)
        test_data = test_data.sample(frac=0.5)
        defaults["min_encoder_length"] = 0
        defaults["min_prediction_length"] = 1

    # create dataset and sample from it
    dataset = TimeSeriesDataSet(test_data, **kwargs)
    repr(dataset)
    check_dataloader_output(dataset, next(iter(dataset.to_dataloader(num_workers=0))))


def test_from_dataset(test_dataset, test_data):
    dataset = TimeSeriesDataSet.from_dataset(test_dataset, test_data)
    check_dataloader_output(dataset, next(iter(dataset.to_dataloader(num_workers=0))))


def test_from_dataset_equivalence(test_data):
    training = TimeSeriesDataSet(
        test_data[lambda x: x.time_idx < x.time_idx.max() - 1],
        time_idx="time_idx",
        target="volume",
        time_varying_known_reals=["price_regular", "time_idx"],
        group_ids=["agency", "sku"],
        static_categoricals=["agency"],
        max_encoder_length=3,
        max_prediction_length=2,
        min_prediction_length=1,
        min_encoder_length=0,
        randomize_length=None,
        add_encoder_length=True,
        add_relative_time_idx=True,
        add_target_scales=True,
    )
    validation1 = TimeSeriesDataSet.from_dataset(training, test_data, predict=True)
    validation2 = TimeSeriesDataSet.from_dataset(
        training,
        test_data[lambda x: x.time_idx > x.time_idx.min() + 2],
        predict=True,
    )
    # ensure validation1 and validation2 datasets are exactly
    # the same despite different data inputs
    for v1, v2 in zip(
        iter(validation1.to_dataloader(train=False)),
        iter(validation2.to_dataloader(train=False)),
    ):
        for k in v1[0].keys():
            if isinstance(v1[0][k], (tuple, list)):
                assert len(v1[0][k]) == len(v2[0][k])
                for idx in range(len(v1[0][k])):
                    assert torch.isclose(v1[0][k][idx], v2[0][k][idx]).all()
            else:
                assert torch.isclose(v1[0][k], v2[0][k]).all()
        assert torch.isclose(v1[1][0], v2[1][0]).all()


def test_dataset_index(test_dataset):
    index = []
    for x, _ in iter(test_dataset.to_dataloader()):
        index.append(test_dataset.x_to_index(x))
    index = pd.concat(index, axis=0, ignore_index=True)
    assert len(index) <= len(test_dataset), "Index can only be subset of dataset"


@pytest.mark.parametrize("min_prediction_idx", [0, 1, 3, 7])
def test_min_prediction_idx(test_dataset, test_data, min_prediction_idx):
    dataset = TimeSeriesDataSet.from_dataset(
        test_dataset,
        test_data,
        min_prediction_idx=min_prediction_idx,
        min_encoder_length=1,
        max_prediction_length=10,
    )

    for x, _ in iter(dataset.to_dataloader(num_workers=0, batch_size=1000)):
        assert x["decoder_time_idx"].min() >= min_prediction_idx


@pytest.mark.parametrize(
    "value,variable,target",
    [
        (1.0, "price_regular", "encoder"),
        (1.0, "price_regular", "all"),
        (1.0, "price_regular", "decoder"),
        ("Agency_01", "agency", "all"),
        ("Agency_01", "agency", "decoder"),
    ],
)
def test_overwrite_values(test_dataset, value, variable, target):
    dataset = deepcopy(test_dataset)

    # create variables to check against
    control_outputs = next(iter(dataset.to_dataloader(num_workers=0, train=False)))
    dataset.set_overwrite_values(value, variable=variable, target=target)

    # test change
    outputs = next(iter(dataset.to_dataloader(num_workers=0, train=False)))
    check_dataloader_output(dataset, outputs)

    if variable in dataset.reals:
        output_name_suffix = "cont"
    else:
        output_name_suffix = "cat"

    if target == "all":
        output_names = [
            f"encoder_{output_name_suffix}",
            f"decoder_{output_name_suffix}",
        ]
    else:
        output_names = [f"{target}_{output_name_suffix}"]

    for name in outputs[0].keys():
        changed = torch.isclose(outputs[0][name], control_outputs[0][name]).all()
        if name in output_names or (
            "cat" in name and variable == "agency"
        ):  # exception for static categorical which should always change
            assert not changed, f"Output {name} should change"
        else:
            assert changed, f"Output {name} should not change"

    # test resetting
    dataset.reset_overwrite_values()
    outputs = next(iter(dataset.to_dataloader(num_workers=0, train=False)))
    for name in outputs[0].keys():
        changed = torch.isclose(outputs[0][name], control_outputs[0][name]).all()
        assert changed, f"Output {name} should be reset"
    assert torch.isclose(
        outputs[1][0], control_outputs[1][0]
    ).all(), "Target should be reset"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(
            target_normalizer=GroupNormalizer(
                groups=["agency", "sku"], transformation="log1p", scale_by_group=True
            ),
        ),
    ],
)
def test_new_group_ids(test_data, kwargs):
    """Test for new group ids in dataset"""
    train_agency = test_data["agency"].iloc[0]
    train_dataset = TimeSeriesDataSet(
        test_data[lambda x: x.agency == train_agency],
        time_idx="time_idx",
        target="volume",
        group_ids=["agency", "sku"],
        max_encoder_length=5,
        max_prediction_length=2,
        min_prediction_length=1,
        min_encoder_length=1,
        categorical_encoders=dict(
            agency=NaNLabelEncoder(add_nan=True), sku=NaNLabelEncoder(add_nan=True)
        ),
        **kwargs,
    )

    # test sampling from training dataset
    next(iter(train_dataset.to_dataloader()))

    # create test dataset with group ids that have not been observed before
    test_dataset = TimeSeriesDataSet.from_dataset(train_dataset, test_data)

    # check that we can iterate through dataset without error
    for _ in iter(test_dataset.to_dataloader()):
        pass


def test_timeseries_columns_naming(test_data):
    with pytest.raises(ValueError):
        TimeSeriesDataSet(
            test_data.rename(columns=dict(agency="agency.2")),
            time_idx="time_idx",
            target="volume",
            group_ids=["agency.2", "sku"],
            max_encoder_length=5,
            max_prediction_length=2,
            min_prediction_length=1,
            min_encoder_length=1,
        )


def test_encoder_normalizer_for_covariates(test_data):
    dataset = TimeSeriesDataSet(
        test_data,
        time_idx="time_idx",
        target="volume",
        group_ids=["agency", "sku"],
        max_encoder_length=5,
        max_prediction_length=2,
        min_prediction_length=1,
        min_encoder_length=1,
        time_varying_known_reals=["price_regular"],
        scalers={"price_regular": EncoderNormalizer()},
    )
    next(iter(dataset.to_dataloader()))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(
            target_normalizer=MultiNormalizer(
                normalizers=[TorchNormalizer(), EncoderNormalizer()]
            ),
        ),
        dict(add_target_scales=True),
        dict(weight="volume"),
    ],
)
def test_multitarget(test_data, kwargs):
    dataset = TimeSeriesDataSet(
        test_data.assign(volume1=lambda x: x.volume),
        time_idx="time_idx",
        target=["volume", "volume1"],
        group_ids=["agency", "sku"],
        max_encoder_length=5,
        max_prediction_length=2,
        min_prediction_length=1,
        min_encoder_length=1,
        time_varying_known_reals=["price_regular"],
        scalers={"price_regular": EncoderNormalizer()},
        **kwargs,
    )
    next(iter(dataset.to_dataloader()))


def test_check_nas(test_data):
    data = test_data.copy()
    data.loc[0, "volume"] = np.nan
    with pytest.raises(ValueError, match=r"1 \(.*infinite"):
        TimeSeriesDataSet(
            data,
            time_idx="time_idx",
            target=["volume"],
            group_ids=["agency", "sku"],
            max_encoder_length=5,
            max_prediction_length=2,
            min_prediction_length=1,
            min_encoder_length=1,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(target="volume"),
        dict(target="agency", scalers={"volume": EncoderNormalizer()}),
        dict(target="volume", target_normalizer=EncoderNormalizer()),
        dict(target=["volume", "agency"]),
    ],
)
def test_lagged_variables(test_data, kwargs):
    dataset = TimeSeriesDataSet(
        test_data.copy(),
        time_idx="time_idx",
        group_ids=["agency", "sku"],
        max_encoder_length=5,
        max_prediction_length=2,
        min_prediction_length=1,
        min_encoder_length=3,  # one more than max lag for validation
        time_varying_unknown_reals=["volume"],
        time_varying_unknown_categoricals=["agency"],
        lags={"volume": [1, 2], "agency": [1, 2]},
        add_encoder_length=False,
        **kwargs,
    )

    x_all, _ = next(iter(dataset.to_dataloader()))

    for name in ["volume", "agency"]:
        if name in dataset.reals:
            vars = dataset.reals
            x = x_all["encoder_cont"]
        else:
            vars = dataset.flat_categoricals
            x = x_all["encoder_cat"]
        target_idx = vars.index(name)
        for lag in [1, 2]:
            lag_idx = vars.index(f"{name}_lagged_by_{lag}")
            target = x[..., target_idx][:, 0]
            lagged_target = torch.roll(x[..., lag_idx], -lag, dims=1)[:, 0]
            assert torch.isclose(
                target, lagged_target
            ).all(), "lagged target must be the same as non-lagged target"


@pytest.mark.parametrize(
    "agency,first_prediction_idx,should_raise",
    [
        ("Agency_01", 0, False),
        ("xxxxx", 0, True),
        ("Agency_01", 100, True),
        ("Agency_01", 4, False),
    ],
)
def test_filter_data(test_dataset, agency, first_prediction_idx, should_raise):
    func = lambda x: (x.agency == agency) & (
        x.time_idx_first_prediction >= first_prediction_idx
    )
    if should_raise:
        with pytest.raises(ValueError):
            test_dataset.filter(func)
    else:
        filtered_dataset = test_dataset.filter(func)
        assert len(test_dataset.index) > len(
            filtered_dataset.index
        ), "filtered dataset should have less entries than original dataset"
        for x, _ in iter(filtered_dataset.to_dataloader()):
            index = test_dataset.x_to_index(x)
            assert (index["agency"] == agency).all(), "Agency filter has failed"
            assert (
                index["time_idx"].min() == first_prediction_idx
            ), "First prediction filter has failed"


def test_graph_sampler(test_dataset):
    from pytorch_forecasting.data.samplers import TimeSynchronizedBatchSampler

    class NeighborhoodSampler(TimeSynchronizedBatchSampler):
        def construct_batch_groups(self, groups):
            batch_size = self.batch_size
            self.batch_size = 1
            super().construct_batch_groups(groups)
            self.batch_size = batch_size

        def __iter__(self):
            if self.shuffle:
                batch_samples = np.random.permutation(len(self))
            else:
                batch_samples = np.arange(len(self))

            def distance_to_weights(dist):
                return 1 / (1e-2 + np.power(dist * 5, 2))

            # for each point, sample the neighborhood
            # get groups associated with chosen sample
            data_groups = self.sampler.data_source.data["groups"].float()
            n_groups = data_groups.size(1)  # number time series ids
            for idx in batch_samples:
                name = self._group_index[idx]  # time-synchronized group name
                sub_group_idx = self._sub_group_index[idx]
                selected_index = self._groups[name][sub_group_idx]
                # select all other indices in same time group
                indices = self.sampler.data_source.index.iloc[self._groups[name]]
                selected_pos = indices["index_start"].iloc[sub_group_idx]
                # remove selected sample
                indices = indices[
                    lambda x: x["sequence_id"]
                    != indices["sequence_id"].iloc[sub_group_idx]
                ]
                # filter duplicate timeseries
                # indices = indices.sort_values("sequence_length").drop_duplicates("sequence_id", keep="last") # noqa : E501

                # calculate distances for corresponding groups
                group_distances = torch.cdist(
                    data_groups[[selected_pos]],
                    data_groups[indices["index_start"].to_numpy()],
                    p=0,
                )[0].numpy()
                # filter out all samples without group-link but not itself
                connected_samples = group_distances < n_groups
                relevant_indices = indices.index[connected_samples]
                sample_weights = distance_to_weights(
                    group_distances[connected_samples]
                )  # calculate weights for sampling neighborhood

                # sample random subset of neighborhood
                batch_size = min(len(relevant_indices), self.batch_size - 1)
                batch_indices = [selected_index] + np.random.choice(
                    relevant_indices,
                    p=sample_weights / sample_weights.sum(),
                    replace=False,
                    size=batch_size,
                ).tolist()
                yield batch_indices

    dl = test_dataset.to_dataloader(
        batch_sampler=NeighborhoodSampler(
            SequentialSampler(test_dataset), batch_size=200, shuffle=True
        )
    )
    for idx, a in enumerate(dl):
        print(a[0]["groups"].shape)
        if idx > 100:
            break
    print(a)
