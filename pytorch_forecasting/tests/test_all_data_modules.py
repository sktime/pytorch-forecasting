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

    def test_init(self, object_pkg, object_instance):
        """Datamodule stores hyperparameters and reads D1 metadata."""
        assert object_instance.time_series_dataset is not None
        assert isinstance(object_instance.time_series_metadata, dict)
        assert "cols" in object_instance.time_series_metadata
        assert object_instance.batch_size > 0
        assert object_instance._context_length() > 0
        assert object_instance._prediction_length() > 0

        batch_format = object_pkg.get_class_tag("batch_format")
        if batch_format == "encoder_decoder":
            assert (
                object_instance.max_encoder_length == object_instance._context_length()
            )
            assert (
                object_instance.max_prediction_length
                == object_instance._prediction_length()
            )
            assert object_instance.train_val_test_split == (0.7, 0.15, 0.15)
        elif batch_format == "tslib":
            assert object_instance.context_length == object_instance._context_length()
            assert (
                object_instance.prediction_length
                == object_instance._prediction_length()
            )
            assert object_instance.train_val_test_split == (0.7, 0.15, 0.15)

    def test_metadata_property(self, object_pkg, object_instance):
        """Metadata property caches the prepared dict with format-specific counts."""
        metadata = object_instance.metadata
        assert object_instance.metadata is metadata

        batch_format = object_pkg.get_class_tag("batch_format")
        if batch_format == "encoder_decoder":
            assert metadata["encoder_cat"] == len(object_instance.categorical_indices)
            assert metadata["encoder_cont"] == len(object_instance.continuous_indices)
            known_cat, known_cont = _known_feature_counts(
                object_instance.time_series_metadata
            )
            assert metadata["decoder_cat"] == known_cat
            assert metadata["decoder_cont"] == known_cont
        elif batch_format == "tslib":
            for key in metadata["n_features"]:
                assert metadata["n_features"][key] == len(
                    metadata["feature_names"][key]
                )

    def test_setup_fit(self, object_pkg, object_instance):
        """Fit stage creates train and validation datasets with windows."""
        object_instance.setup(stage="fit")
        assert object_instance.train_dataset is not None
        assert object_instance.val_dataset is not None
        assert len(object_instance.train_windows) > 0
        assert len(object_instance.val_windows) > 0
        assert len(object_instance.train_dataset) == len(object_instance.train_windows)
        assert len(object_instance.val_dataset) == len(object_instance.val_windows)

    def test_setup_test_predict(self, object_pkg, object_instance):
        """Test and predict stages create their datasets."""
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
        """Train/val/test indices respect configured split ratios."""
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
        assert (
            len(dm._train_indices) + len(dm._val_indices) + len(dm._test_indices)
            == total_series
        )

    def test_create_windows(self, object_pkg, object_instance):
        """Windows are 4-tuples with valid indices and configured lengths."""
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

        all_indices = torch.arange(len(object_instance.time_series_dataset))
        all_windows = object_instance._create_windows(all_indices)
        assert len(all_windows) >= len(windows)

        empty_windows = object_instance._create_windows(torch.tensor([]))
        assert len(empty_windows) == 0

    def test_dataloader_creation(self, object_pkg, object_instance):
        """Dataloaders honour batch_size and num_workers across all stages."""
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
        """Single dataset items expose expected keys, shapes, and dtypes."""
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
        """Collated batch contains expected keys and feature dimensions."""
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
        """Train dataloader yields batches with correct tensor dimensions."""
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
        """Metadata contains format-specific keys and nested structure."""
        metadata = object_instance.metadata
        for key in object_pkg.get_expected_metadata_keys():
            assert key in metadata

        batch_format = object_pkg.get_class_tag("batch_format")
        if batch_format == "encoder_decoder":
            assert metadata["max_encoder_length"] == object_instance._context_length()
            assert (
                metadata["max_prediction_length"]
                == object_instance._prediction_length()
            )
        elif batch_format == "tslib":
            assert metadata["context_length"] == object_instance._context_length()
            assert metadata["prediction_length"] == object_instance._prediction_length()

            for group in (
                "categorical",
                "continuous",
                "static",
                "known",
                "unknown",
                "target",
                "all",
                "static_categorical",
                "static_continuous",
            ):
                assert group in metadata["feature_names"]
            for group in (
                "categorical",
                "continuous",
                "static",
                "known",
                "unknown",
                "target",
            ):
                assert group in metadata["feature_indices"]

    def test_multivariate_target(self, object_pkg, object_instance):
        """Multivariate targets are returned as a list of tensors."""
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
            assert y.shape[-1] == 2
        else:
            assert len(y) == 2

    def test_preprocess_data(self, object_pkg, object_instance):
        """Preprocessed series expose feature and target tensors."""
        object_instance.setup(stage="fit")
        series_idx = object_instance._train_indices[0]
        processed = object_instance._preprocess_data(series_idx)

        assert "features" in processed
        assert "categorical" in processed["features"]
        assert "continuous" in processed["features"]
        assert "target" in processed
        assert "time_mask" in processed

        batch_format = object_pkg.get_class_tag("batch_format")
        if batch_format == "tslib":
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
        """Datamodule exposes static features in metadata and samples when configured"""
        batch_format = object_pkg.get_class_tag("batch_format")
        dm_class = object_pkg.get_cls()

        if batch_format == "encoder_decoder":
            df = pd.DataFrame(
                {
                    "group": [0, 0, 0, 1, 1, 1],
                    "time": pd.date_range("2020-01-01", periods=6),
                    "target": [1, 2, 3, 4, 5, 6],
                    "static_cat": [0, 0, 0, 1, 1, 1],
                    "static_num": [10, 10, 10, 20, 20, 20],
                    "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                }
            )
            ts = TimeSeries(
                data=df,
                time="time",
                target="target",
                group=["group"],
                num=["feature1", "static_num"],
                static=["static_cat", "static_num"],
                cat=["static_cat"],
            )
            dm = dm_class(
                time_series_dataset=ts,
                max_encoder_length=2,
                max_prediction_length=1,
                batch_size=2,
            )
            dm.setup(stage="fit")

            metadata = dm.metadata
            assert metadata["static_categorical_features"] == 1
            assert metadata["static_continuous_features"] == 1

            x, _ = dm.train_dataset[0]
            assert "static_categorical_features" in x
            assert "static_continuous_features" in x
            assert (
                x["static_categorical_features"].shape[1]
                == metadata["static_categorical_features"]
            )
            assert (
                x["static_continuous_features"].shape[1]
                == metadata["static_continuous_features"]
            )
        elif batch_format == "tslib":
            ts = make_datamodule_test_timeseries()
            params = object_pkg.get_datamodule_test_params()[0]
            dm = dm_class(time_series_dataset=ts, **params)
            dm.setup(stage="fit")

            metadata = dm.metadata
            assert metadata["n_features"]["static_continuous"] == 1

            x, _ = dm.train_dataset[0]
            assert "static_continuous_features" in x
            assert (
                x["static_continuous_features"].shape[1]
                == metadata["n_features"]["static_continuous"]
            )

            train_loader = dm.train_dataloader()
            x_batch, _ = next(iter(train_loader))
            assert "static_categorical_features" in x_batch
            assert "static_continuous_features" in x_batch

    def test_variable_encoder_lengths(self, object_pkg, object_instance):
        """Variable encoder lengths are respected when randomize_length is enabled."""
        if object_pkg.get_class_tag("batch_format") != "encoder_decoder":
            pytest.skip(
                "Variable encoder length test only applies to encoder-decoder" "format."
            )

        dm_class = object_pkg.get_cls()
        ts = object_instance.time_series_dataset
        params = object_pkg.get_datamodule_test_params()[0]
        params = {
            **dict(params),
            "batch_size": 4,
            "min_encoder_length": 12,
            "max_encoder_length": 24,
            "randomize_length": True,
            "max_prediction_length": 12,
        }
        dm = dm_class(time_series_dataset=ts, **params)
        dm.setup(stage="fit")
        assert dm.min_encoder_length == 12
        assert dm.max_encoder_length == 24
