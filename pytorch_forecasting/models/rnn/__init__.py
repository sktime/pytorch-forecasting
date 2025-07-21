"""Simple recurrent model - either with LSTM or GRU cells."""

from pytorch_forecasting.models.rnn._rnn import RecurrentNetwork
from pytorch_forecasting.models.rnn._rnn_pkg import RNN_pkg

__all__ = ["RecurrentNetwork", "RNN_pkg"]
