"""Automated tests for all v2 D2 datamodules."""

import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting.data.timeseries import TimeSeries
from pytorch_forecasting.tests._base._fixture_generator import BaseFixtureGenerator
from pytorch_forecasting.tests._config import resolve_batch_key
from pytorch_forecasting.tests._data_scenarios import make_datamodule_test_timeseries
from pytorch_forecasting.tests._datamodule_config import (
    EXCLUDE_DATA_MODULES,
    EXCLUDED_TESTS,
)


def _known_feature_counts(time_series_metadata):
    """Count known categorical and continuous x-features from D1 metadata."""
    known_cat_count = len(
        [
            col
            for col in time_series_metadata["cols"]["x"]
            if time_series_metadata["col_type"].get(col) == "C"
            and time_series_metadata["col_known"].get(col) == "K"
        ]
    )
    known_cont_count = len(
        [
            col
            for col in time_series_metadata["cols"]["x"]
            if time_series_metadata["col_type"].get(col) == "F"
            and time_series_metadata["col_known"].get(col) == "K"
        ]
    )
    return known_cat_count, known_cont_count


def _series_length(sample):
    """Return the number of timesteps in a D1 series sample."""
    if "t" in sample:
        return len(sample["t"])
    if "y" in sample:
        return len(sample["y"])
    return len(sample)


class DataModulePackageConfig:
    """Configuration for datamodule package tests."""

    package_name = "pytorch_forecasting.data.data_module"

    exclude_objects = EXCLUDE_DATA_MODULES
    excluded_tests = EXCLUDED_TESTS


class DataModuleFixtureGenerator(BaseFixtureGenerator):
    """Fixture generator for D2 datamodule contract tests."""

    fixture_sequence = ["object_pkg", "object_instance"]
    indirect_fixtures = []

    @staticmethod
    def is_excluded(test_name, obj_meta, param_name=None):
        """Shorthand to check whether test test_name is excluded for datamodule obj."""

        # global excluded tests
        if test_name in EXCLUDED_TESTS.get(obj_meta.__name__, []):
            return True

        # package level excluded tests
        if obj_meta.__name__.endswith("_pkg") or obj_meta.__name__.endswith("_pkg_v2"):
            excl_tag = obj_meta.get_class_tag("tests:skip_by_name", [])
        else:
            excl_tag = obj_meta.pkg.get_class_tag("tests:skip_by_name", [])
        if excl_tag is None:
            excl_tag = []
        cond = test_name in excl_tag

        # indivisual parameterized test exclusion
        if param_name is not None:
            full_test_name = f"{test_name}[{obj_meta.__name__}-{param_name}]"
            if full_test_name in excl_tag:
                return True
        return cond

    def _generate_object_instance(self, test_name, **kwargs):
        if "object_pkg" not in kwargs:
            return [], []

        obj_meta = kwargs["object_pkg"]
        dm_class = obj_meta.get_cls()
        all_params = obj_meta.get_datamodule_test_params()

        if self.is_excluded(test_name, obj_meta):
            return [], []

        if not all_params:
            ts = make_datamodule_test_timeseries()
            return [dm_class(time_series_dataset=ts)], ["default"]

        instances = []
        names = []
        for i, params in enumerate(all_params):
            params_copy = dict(params)
            ts_kwargs = params_copy.pop("timeseries_kwargs", {})
            ts = make_datamodule_test_timeseries(**ts_kwargs)
            if self.is_excluded(test_name, obj_meta, str(i)):
                continue
            instances.append(dm_class(time_series_dataset=ts, **params_copy))
            names.append(str(i))

        return instances, names


class TestAllDataModules(DataModulePackageConfig, DataModuleFixtureGenerator):
    """Generic contract tests for all v2 D2 datamodules."""

    object_type_filter = "datamodule_v2"

    def test_metadata_property(self, object_pkg, object_instance):
        """Metadata property is cached and reports correct feature counts.

        Categorical and continuous feature counts match the expected counts.
        """
        metadata = object_instance.metadata
        assert object_instance.metadata is metadata  # same object on repeat access

        known_cat, known_cont = _known_feature_counts(
            object_instance.time_series_metadata
        )
        expected_counts = {
            "history_cat": len(object_instance.categorical_indices),
            "history_cont": len(object_instance.continuous_indices),
            "future_cat": known_cat,
            "future_cont": known_cont,
        }
        for role, expected in expected_counts.items():
            key = resolve_batch_key(metadata, role)
            if key is not None:
                assert metadata[key] == expected

    def test_setup_fit(self, object_pkg, object_instance):
        """setup('fit') creates non-empty train/val datasets aligned with windows."""
        object_instance.setup(stage="fit")
        assert object_instance.train_dataset is not None
        assert object_instance.val_dataset is not None
        assert len(object_instance.train_windows) > 0
        assert len(object_instance.val_windows) > 0
        assert len(object_instance.train_dataset) == len(object_instance.train_windows)
        assert len(object_instance.val_dataset) == len(object_instance.val_windows)

    def test_setup_test_predict(self, object_pkg, object_instance):
        """setup for 'test' and 'predict' create their non-empty datasets & windows."""
        object_instance.setup(stage="fit")
        object_instance.setup(stage="test")
        object_instance.setup(stage="predict")

        assert object_instance.test_dataset is not None
        assert object_instance.predict_dataset is not None
        assert len(object_instance.test_windows) > 0
        assert len(object_instance.predict_windows) > 0

    @pytest.mark.parametrize(
        "split",
        [(0.7, 0.15, 0.15), (0.8, 0.1, 0.1), (0.6, 0.2, 0.2)],
    )
    def test_different_train_val_test_split(self, object_pkg, split):
        """Train/val/test index sizes follow the configured split ratios.

        Parametrized over lengths of three tuples (train, val, test).
        """
        dm_class = object_pkg.get_cls()
        ts = make_datamodule_test_timeseries()
        params = dict(object_pkg.get_datamodule_test_params()[0])
        params["train_val_test_split"] = split
        dm = dm_class(time_series_dataset=ts, **params)
        dm.setup(stage="fit")

        total_series = len(ts)
        expected_train = int(split[0] * total_series)
        expected_val = int(split[1] * total_series)
        expected_test = total_series - expected_train - expected_val

        assert len(dm._train_indices) == expected_train
        assert len(dm._val_indices) == expected_val
        assert len(dm._test_indices) == expected_test
        assert dm.train_val_test_split == split
        # train + val + test must partition all series exactly once
        assert (
            len(dm._train_indices) + len(dm._val_indices) + len(dm._test_indices)
            == total_series
        )

    def test_create_windows(self, object_pkg, object_instance):
        """_create_windows returns valid (series_idx, start, context, prediction)
        tuples.

        Each window must fit inside the source series. Also covers the all-series
        and empty-index edge cases.
        """
        object_instance.setup(stage="fit")
        windows = object_instance._create_windows(object_instance._train_indices)

        assert len(windows) > 0
        context_length = object_instance._context_length()
        prediction_length = object_instance._prediction_length()

        for window in windows:
            assert len(window) == 4
            series_idx, start_idx, window_context, window_prediction = window
            assert window_context == context_length
            assert window_prediction == prediction_length
            assert isinstance(series_idx, int)
            assert isinstance(start_idx, int)
            assert 0 <= series_idx < len(object_instance.time_series_dataset)

            sample = object_instance.time_series_dataset[series_idx]
            min_required_length = context_length + prediction_length
            assert start_idx + min_required_length <= _series_length(sample)

        # all series should produce at least as many windows as the train split
        all_indices = torch.arange(len(object_instance.time_series_dataset))
        all_windows = object_instance._create_windows(all_indices)
        assert len(all_windows) >= len(windows)

        empty_windows = object_instance._create_windows(torch.tensor([]))
        assert len(empty_windows) == 0

    def test_dataloader_creation(self, object_pkg, object_instance):
        """Dataloaders for fit/test/predict honour batch_size and num_workers."""
        object_instance.setup(stage="fit")
        train_loader = object_instance.train_dataloader()
        val_loader = object_instance.val_dataloader()

        assert train_loader.batch_size == object_instance.batch_size
        assert train_loader.num_workers == object_instance.num_workers
        assert val_loader.batch_size == object_instance.batch_size

        object_instance.setup(stage="test")
        test_loader = object_instance.test_dataloader()
        assert test_loader.batch_size == object_instance.batch_size
        assert test_loader.num_workers == object_instance.num_workers

        object_instance.setup(stage="predict")
        predict_loader = object_instance.predict_dataloader()
        assert predict_loader.batch_size == object_instance.batch_size
        assert predict_loader.num_workers == object_instance.num_workers

    def test_processed_dataset(self, object_pkg, object_instance):
        """A single train sample has expected keys, tensor shapes, and dtypes.

        History/future features should match context/prediction length and
        known-feature counts; multivariate targets return a list of tensors.
        """
        object_instance.setup(stage="fit")
        assert len(object_instance.train_dataset) == len(object_instance.train_windows)
        assert len(object_instance.val_dataset) == len(object_instance.val_windows)

        x, y = object_instance.train_dataset[0]
        for key in object_pkg.get_sample_item_keys():
            assert key in x

        context_length = object_instance._context_length()
        prediction_length = object_instance._prediction_length()
        known_cat_count, known_cont_count = _known_feature_counts(
            object_instance.time_series_metadata
        )

        x_history_cat_key = resolve_batch_key(x, "history_cat")
        if x_history_cat_key is not None:
            assert x[x_history_cat_key].shape[0] == context_length

        x_future_cat_key = resolve_batch_key(x, "future_cat")
        if x_future_cat_key is not None:
            assert x[x_future_cat_key].shape[0] == prediction_length
            assert x[x_future_cat_key].shape[1] == known_cat_count

        x_future_cont_key = resolve_batch_key(x, "future_cont")
        if x_future_cont_key is not None:
            assert x[x_future_cont_key].shape[1] == known_cont_count

        x_history_target_key = resolve_batch_key(x, "history_target")
        if x_history_target_key is not None:
            assert x[x_history_target_key].shape[0] == context_length

        assert y.shape[0] == prediction_length

        x_history_cont_key = resolve_batch_key(x, "history_cont")
        if x_history_cont_key is not None:
            assert x[x_history_cont_key].dtype == torch.float32

        if x_future_cont_key is not None:
            assert x[x_future_cont_key].dtype == torch.float32

        if x_history_target_key is not None:
            assert x[x_history_target_key].dtype == torch.float32
        assert y.dtype == torch.float32

        if object_instance.n_targets > 1:
            assert isinstance(y, list)
        else:
            assert isinstance(y, torch.Tensor)

    def test_collate_fn(self, object_pkg, object_instance):
        """collate_fn stacks manual samples into a batch with correct dimensions."""
        object_instance.setup(stage="fit")
        batch_size = min(3, len(object_instance.train_dataset))
        batch = [object_instance.train_dataset[i] for i in range(batch_size)]
        x_batch, y_batch = object_instance.collate_fn(batch)

        for key in object_pkg.get_batch_keys():
            assert key in x_batch

        for value in x_batch.values():
            assert value.shape[0] == batch_size

        prediction_length = object_instance._prediction_length()
        known_cat_count, known_cont_count = _known_feature_counts(
            object_instance.time_series_metadata
        )

        x_future_cat_key = resolve_batch_key(x_batch, "future_cat")
        if x_future_cat_key is not None:
            assert x_batch[x_future_cat_key].shape[2] == known_cat_count

        x_future_cont_key = resolve_batch_key(x_batch, "future_cont")
        if x_future_cont_key is not None:
            assert x_batch[x_future_cont_key].shape[2] == known_cont_count
        assert y_batch.shape[0] == batch_size
        assert y_batch.shape[1] == prediction_length

    def test_full_dataloader_iteration(self, object_pkg, object_instance):
        """One train-loader batch has the expected batch/time/feature dimensions.

        End-to-end check through the dataloader rather than collate_fn alone.
        """
        object_instance.setup(stage="fit")
        train_loader = object_instance.train_dataloader()
        x_batch, y_batch = next(iter(train_loader))

        assert isinstance(x_batch, dict)
        batch_size = object_instance.batch_size
        context_length = object_instance._context_length()
        prediction_length = object_instance._prediction_length()

        known_cat_count, known_cont_count = _known_feature_counts(
            object_instance.time_series_metadata
        )

        x_history_cat_key = resolve_batch_key(x_batch, "history_cat")
        if x_history_cat_key is not None:
            assert x_batch[x_history_cat_key].shape[0] == batch_size
            assert x_batch[x_history_cat_key].shape[1] == context_length

        x_future_cat_key = resolve_batch_key(x_batch, "future_cat")
        if x_future_cat_key is not None:
            assert x_batch[x_future_cat_key].shape[0] == batch_size
            assert x_batch[x_future_cat_key].shape[2] == known_cat_count

        x_future_cont_key = resolve_batch_key(x_batch, "future_cont")
        if x_future_cont_key is not None:
            assert x_batch[x_future_cont_key].shape[0] == batch_size
            assert x_batch[x_future_cont_key].shape[2] == known_cont_count
        if isinstance(y_batch, list):
            assert all(t.shape[0] == batch_size for t in y_batch)
        else:
            assert y_batch.shape[0] == batch_size
            assert y_batch.shape[1] == prediction_length

    def test_prepare_metadata(self, object_pkg, object_instance):
        """Prepared metadata has all package-expected keys."""
        metadata = object_instance.metadata
        for key in object_pkg.get_expected_metadata_keys():
            assert key in metadata

    def test_multivariate_target(self, object_pkg, object_instance):
        """Two target columns are exposed correctly in a train sample.

        Encoder-decoder: returns a list of tensors;
        Tslib: may stack them.
        """
        dm_class = object_pkg.get_cls()
        df = pd.DataFrame(
            {
                "group": np.repeat([0, 1], 50),
                "time": np.tile(pd.date_range("2020-01-01", periods=50), 2),
                "target1": np.random.normal(0, 1, 100),
                "target2": np.random.normal(5, 2, 100),
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
            }
        )

        ts = TimeSeries(
            data=df,
            time="time",
            target=["target1", "target2"],
            group=["group"],
            num=["feature1", "feature2"],
        )
        params = object_pkg.get_datamodule_test_params()[0]
        params = dict(params)

        if object_pkg.get_class_tag("batch_format") == "encoder_decoder":
            params.setdefault("max_encoder_length", 10)
            params.setdefault("max_prediction_length", 5)
        else:
            params.setdefault("context_length", 8)
            params.setdefault("prediction_length", 4)

        params["batch_size"] = 2
        dm = dm_class(time_series_dataset=ts, **params)
        dm.setup(stage="fit")

        x, y = dm.train_dataset[0]
        assert x is not None
        if isinstance(y, list):
            assert len(y) == 2
        elif object_pkg.get_class_tag("batch_format") == "tslib":
            assert y.shape[-1] == 2  # stacked multivariate target
        else:
            assert len(y) == 2

    def test_preprocess_data(self, object_pkg, object_instance):
        """_preprocess_data returns the expected per-series dict and tensor lengths.

        Feature and target tensors should span the full original series length.
        """
        object_instance.setup(stage="fit")
        series_idx = object_instance._train_indices[0]
        processed = object_instance._preprocess_data(series_idx)

        assert "features" in processed
        assert "categorical" in processed["features"]
        assert "continuous" in processed["features"]
        assert "target" in processed
        assert "time_mask" in processed
        assert "static" in processed
        assert "group" in processed
        assert "length" in processed
        assert "timestep" in processed

        original_sample = object_instance.time_series_dataset[series_idx.item()]
        expected_length = _series_length(original_sample)
        assert processed["features"]["categorical"].shape[0] == expected_length
        assert processed["features"]["continuous"].shape[0] == expected_length
        assert processed["target"].shape[0] == expected_length

    def test_with_static_features(self, object_pkg, object_instance):
        """Non-empty static features appear in samples and batches when configured."""
        dm_class = object_pkg.get_cls()

        ts = make_datamodule_test_timeseries(
            include_static=True,
            include_static_categorical=True,
        )
        params = dict(object_pkg.get_datamodule_test_params()[0])
        dm = dm_class(time_series_dataset=ts, **params)
        dm.setup(stage="fit")

        x, _ = dm.train_dataset[0]
        assert "static_categorical_features" in x
        assert "static_continuous_features" in x
        n_static_cat = x["static_categorical_features"].shape[1]
        n_static_cont = x["static_continuous_features"].shape[1]
        assert n_static_cat > 0
        assert n_static_cont > 0

        train_loader = dm.train_dataloader()
        x_batch, _ = next(iter(train_loader))
        assert "static_categorical_features" in x_batch
        assert "static_continuous_features" in x_batch
        assert x_batch["static_categorical_features"].shape[-1] == n_static_cat
        assert x_batch["static_continuous_features"].shape[-1] == n_static_cont
