"""
Tests for the tuning module:
_SearchRange, _parse_param_grid, _auto_discover_ranges.
"""

import pytest

from pytorch_forecasting.tuning.search_range import _SearchRange


def _make_search_cv(**kwargs):
    """Create a ForecastingSearchCV with a DLinear_pkg_v2 instance."""
    from pytorch_forecasting.models.dlinear import DLinear_pkg_v2
    from pytorch_forecasting.tuning.forecasting_search_cv import ForecastingSearchCV

    pkg = DLinear_pkg_v2(
        datamodule_cfg={
            "context_length": 8,
            "prediction_length": 2,
            "batch_size": 2,
            "train_val_test_split": (0.5, 0.5),
        },
    )
    return ForecastingSearchCV(pkg=pkg, **kwargs)


class TestParseParamGrid:
    """Tests for ForecastingSearchCV._parse_param_grid."""

    def test_list_becomes_categorical(self):
        """A list input should be converted to a categorical _SearchRange."""
        search = _make_search_cv()
        result = search._parse_param_grid({"moving_avg": [3, 5, 7]})

        assert "moving_avg" in result
        assert isinstance(result["moving_avg"], _SearchRange)
        assert result["moving_avg"].param_type == "categorical"
        assert result["moving_avg"].choices == [3, 5, 7]

    def test_int_tuple_becomes_int_range(self):
        """A tuple of two ints should be converted to an int _SearchRange."""
        search = _make_search_cv()
        result = search._parse_param_grid({"hidden_size": (16, 512)})

        assert result["hidden_size"].param_type == "int"
        assert result["hidden_size"].low == 16
        assert result["hidden_size"].high == 512

    def test_float_tuple_becomes_float_range(self):
        """A tuple with any float should be converted to a float _SearchRange."""
        search = _make_search_cv()
        result = search._parse_param_grid({"dropout": (0.1, 0.5)})

        assert result["dropout"].param_type == "float"
        assert result["dropout"].low == 0.1
        assert result["dropout"].high == 0.5

    def test_mixed_tuple_becomes_float_range(self):
        """A tuple of (int, float) should be treated as a float _SearchRange."""
        search = _make_search_cv()
        result = search._parse_param_grid({"weight_decay": (0, 0.1)})

        assert result["weight_decay"].param_type == "float"

    def test_searchrange_passthrough(self):
        """An existing _SearchRange should pass through unchanged."""
        search = _make_search_cv()
        original = _SearchRange(param_type="int", low=1, high=10)
        result = search._parse_param_grid({"n_heads": original})

        assert result["n_heads"] is original

    def test_boolean_tuple_raises(self):
        """A tuple of booleans should raise ValueError with helpful message."""
        search = _make_search_cv()
        with pytest.raises(ValueError, match="boolean tuple"):
            search._parse_param_grid({"use_norm": (True, False)})

    def test_invalid_type_raises(self):
        """An unsupported type (e.g., a string) should raise ValueError."""
        search = _make_search_cv()
        with pytest.raises(ValueError, match="Invalid search range"):
            search._parse_param_grid({"hidden_size": "big"})

    def test_single_element_tuple_raises(self):
        """A tuple that doesn't have exactly 2 elements should raise ValueError."""
        search = _make_search_cv()
        with pytest.raises(ValueError, match="Invalid search range"):
            search._parse_param_grid({"hidden_size": (16,)})

    def test_multiple_params(self):
        """Multiple parameters should all be parsed correctly."""
        search = _make_search_cv()
        result = search._parse_param_grid(
            {
                "moving_avg": [3, 5, 7],
                "dropout": (0.1, 0.5),
                "hidden_size": (16, 512),
            }
        )

        assert len(result) == 3
        assert result["moving_avg"].param_type == "categorical"
        assert result["dropout"].param_type == "float"
        assert result["hidden_size"].param_type == "int"


class TestAutoDiscoverRanges:
    """Tests for ForecastingSearchCV._auto_discover_ranges."""

    def test_discovers_known_params(self):
        """Params in model __init__ are discovered."""
        search = _make_search_cv()
        discovered = search._auto_discover_ranges()

        assert "moving_avg" in discovered
        assert isinstance(discovered["moving_avg"], _SearchRange)

    def test_skips_unknown_params(self):
        """Params NOT in model __init__ should not appear."""
        search = _make_search_cv()
        discovered = search._auto_discover_ranges()

        assert "individual" not in discovered

    def test_skips_self(self):
        """'self' should never appear in discovered ranges."""
        search = _make_search_cv()
        discovered = search._auto_discover_ranges()

        assert "self" not in discovered

    def test_returns_dict_of_search_ranges(self):
        """All discovered values should be _SearchRange instances."""
        search = _make_search_cv()
        discovered = search._auto_discover_ranges()

        for value in discovered.values():
            assert isinstance(value, _SearchRange)


class TestForecastingSearchCVInit:
    """Tests for ForecastingSearchCV constructor."""

    def test_extracts_pkg_cls(self):
        """__init__ should extract the class from the pkg instance."""
        from pytorch_forecasting.models.dlinear import DLinear_pkg_v2

        search = _make_search_cv()
        assert search.pkg_cls is DLinear_pkg_v2

    def test_extracts_model_cfg(self):
        """__init__ should read model_cfg from the pkg instance."""
        search = _make_search_cv()
        assert isinstance(search.base_model_cfg, dict)

    def test_extracts_trainer_cfg(self):
        """__init__ should read trainer_cfg from the pkg instance."""
        search = _make_search_cv()
        assert isinstance(search.base_trainer_cfg, dict)

    def test_extracts_datamodule_cfg(self):
        """__init__ should read datamodule_cfg from the pkg instance."""
        search = _make_search_cv()
        assert search.base_datamodule_cfg["context_length"] == 8
        assert search.base_datamodule_cfg["prediction_length"] == 2

    def test_param_grid_defaults_to_none(self):
        """param_grid should default to None when not provided."""
        search = _make_search_cv()
        assert search.param_grid is None

    def test_param_grid_stored(self):
        """param_grid should be stored when provided."""
        grid = {"moving_avg": [3, 5]}
        search = _make_search_cv(param_grid=grid)
        assert search.param_grid is grid

    def test_n_trials_default(self):
        """n_trials should default to 50."""
        search = _make_search_cv()
        assert search.n_trials == 50

    def test_n_trials_custom(self):
        """n_trials should accept a custom value."""
        search = _make_search_cv(n_trials=10)
        assert search.n_trials == 10

    def test_post_fit_attrs_none_before_fit(self):
        """best_params_, best_estimator_, study_ should be None before fit."""
        search = _make_search_cv()
        assert search.best_params_ is None
        assert search.best_estimator_ is None
        assert search.study_ is None


class TestPredictGuard:
    """Tests for predict() safety checks."""

    def test_predict_before_fit_raises(self):
        """Calling predict() before fit() should raise RuntimeError."""
        search = _make_search_cv()
        with pytest.raises(RuntimeError, match="No model available"):
            search.predict(None)
