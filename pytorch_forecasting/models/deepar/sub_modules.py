"""
Implementations of ``nn.RNNBase`` for DeepAR.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Type, Union

import torch
from torch import nn

HiddenState = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class TimeSeriesRNN(ABC, nn.RNNBase):
    """
    Base class for implementations of RNN modules compatible with DeepAR.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def handle_no_encoding(self, out, no_encoding) -> HiddenState:
        """Mask the hidden_state where there is no encoding."""
        pass

    @abstractmethod
    def init_hidden_state(self, x, hidden_size) -> HiddenState:
        """Initialise a hidden_state"""
        pass

    @abstractmethod
    def repeat_interleave(self, hidden_state, n_samples: int) -> HiddenState:
        """Duplicate the hidden_state n_samples times."""
        pass


class TimeSeriesLSTM(TimeSeriesRNN, nn.LSTM):
    """Implementation of LSTM module compatible with DeepAR."""

    def handle_no_encoding(self, hidden_state, no_encoding) -> HiddenState:
        hidden, cell = hidden_state
        hidden = hidden.masked_fill(no_encoding, 0.0)
        cell = cell.masked_fill(no_encoding, 0.0)
        return hidden, cell

    def init_hidden_state(self, x, hidden_size) -> HiddenState:
        hidden = torch.zeros(
            (x["encoder_cont"].size(0), hidden_size),
            device=x["decoder_cont"].device,
            dtype=torch.float,
        )
        cell = torch.zeros(
            (x["encoder_cont"].size(0), hidden_size),
            device=x["decoder_cont"].device,
            dtype=torch.float,
        )
        return hidden, cell

    def repeat_interleave(self, hidden_state, n_samples: int) -> HiddenState:
        hidden, cell = hidden_state
        hidden = hidden.repeat_interleave(n_samples, 1)
        cell = cell.repeat_interleave(n_samples, 1)
        return hidden, cell


class TimeSeriesGRU(TimeSeriesRNN, nn.GRU):
    """Implementation of GRU module compatible with DeepAR."""

    def handle_no_encoding(self, hidden_state, no_encoding) -> HiddenState:
        return hidden_state.masked_fill(no_encoding, 0.0)

    def init_hidden_state(self, x, hidden_size) -> HiddenState:
        hidden = torch.zeros(
            (x["encoder_cont"].size(0), hidden_size),
            device=x["decoder_cont"].device,
            dtype=torch.float,
        )
        return hidden

    def repeat_interleave(self, hidden_state, n_samples: int) -> HiddenState:
        return hidden_state.repeat_interleave(n_samples, 1)


def get_cell(cell_type: str) -> Type[TimeSeriesRNN]:
    if isinstance(cell_type, TimeSeriesRNN):
        rnn = cell_type
    elif cell_type == "LSTM":
        rnn = TimeSeriesLSTM
    elif cell_type == "GRU":
        rnn = TimeSeriesGRU
    else:
        raise ValueError(f"DeepAR does not support {cell_type}. supported: [LSTM, GRU]")
    return rnn
