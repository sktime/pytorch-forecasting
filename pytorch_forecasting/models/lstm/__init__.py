"""LSTM implementation for forecasting."""

from pytorch_forecasting.models.lstm._lstm import LSTMModel
from pytorch_forecasting.models.lstm._lstm_pkg import LSTMModel_pkg

__all__ = ["LSTMModel", "LSTMModel_pkg"]
