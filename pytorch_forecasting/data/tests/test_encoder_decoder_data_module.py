import pytest

from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
from pytorch_forecasting.tests._data_scenarios import make_datamodule_test_timeseries


@pytest.fixture(scope="session")
def sample_timeseries_data():
    return make_datamodule_test_timeseries()


@pytest.fixture
def encoder_decoder_data_module(sample_timeseries_data):
    return EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=sample_timeseries_data,
        max_encoder_length=24,
        max_prediction_length=12,
        batch_size=4,
        num_workers=0,
    )


@pytest.fixture
def encoder_decoder_data_module_with_static():
    ts = make_datamodule_test_timeseries(
        include_static=True,
        include_static_categorical=True,
    )
    return EncoderDecoderTimeSeriesDataModule(
        time_series_dataset=ts,
        max_encoder_length=24,
        max_prediction_length=12,
        batch_size=4,
        num_workers=0,
    )


def test_encoder_decoder_metadata_max_lengths(encoder_decoder_data_module):
    """Metadata max lengths match context/prediction length helpers."""
    dm = encoder_decoder_data_module
    metadata = dm.metadata

    assert "max_encoder_length" in metadata
    assert "max_prediction_length" in metadata
    assert metadata["max_encoder_length"] == dm._context_length()
    assert metadata["max_prediction_length"] == dm._prediction_length()


def test_encoder_decoder_static_features_metadata(
    encoder_decoder_data_module_with_static,
):
    """Metadata reports static categorical and continuous feature counts."""
    dm = encoder_decoder_data_module_with_static
    dm.setup(stage="fit")
    metadata = dm.metadata

    assert metadata["static_categorical_features"] == 1
    assert metadata["static_continuous_features"] == 1
