"""Test configs for D2 datamodule unified test suite."""

from collections.abc import Callable

from pytorch_forecasting.data.timeseries import TimeSeries
from pytorch_forecasting.tests._data_scenarios import (
    make_encoder_decoder_timeseries,
    make_tslib_timeseries,
)

EXCLUDE_DATA_MODULES = [
    "ClassName",
]

EXCLUDED_TESTS = {}

# Maps datamodule ``batch_format`` tag to a D1 TimeSeries factory used by tests.
# Register new formats here when adding datamodule pkg classes.
DATAMODULE_TEST_TIMESERIES: dict[str, Callable[..., TimeSeries]] = {
    "encoder_decoder": make_encoder_decoder_timeseries,
    "tslib": make_tslib_timeseries,
}


def get_test_timeseries_for_pkg(object_pkg, **kwargs) -> TimeSeries:
    """Return a test TimeSeries for the given datamodule package."""
    batch_format = object_pkg.get_class_tag("batch_format")
    # TODO: add default (if needed) here
    if batch_format not in DATAMODULE_TEST_TIMESERIES:
        raise ValueError(
            f"No test TimeSeries factory registered for batch_format={batch_format!r}. "
            "Add an entry to DATAMODULE_TEST_TIMESERIES in "
            "pytorch_forecasting.tests._datamodule_config."
        )
    return DATAMODULE_TEST_TIMESERIES[batch_format](**kwargs)
