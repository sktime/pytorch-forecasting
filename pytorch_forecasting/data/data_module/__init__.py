"""Data Modules (D2 Layer) of pytorch-forecasting v2"""

from pytorch_forecasting.data.data_module._encoder_decoder_data_module import (
    EncoderDecoderTimeSeriesDataModule,
)
from pytorch_forecasting.data.data_module._tslib_data_module import TslibDataModule

__all__ = ["EncoderDecoderTimeSeriesDataModule", "TslibDataModule"]
