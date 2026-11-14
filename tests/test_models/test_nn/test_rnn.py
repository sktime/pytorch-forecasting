import itertools

import pytest
import torch
from torch import nn

from pytorch_forecasting.models.nn.rnn import GRU, LSTM, get_rnn


def test_get_lstm_cell():
    cell = get_rnn("LSTM")(10, 10)
    assert isinstance(cell, LSTM)
    assert isinstance(cell, nn.LSTM)


def test_get_gru_cell():
    cell = get_rnn("GRU")(10, 10)
    assert isinstance(cell, GRU)
    assert isinstance(cell, nn.GRU)


def test_get_cell_raises_value_error():
    pytest.raises(ValueError, lambda: get_rnn("ABCDEF"))


@pytest.mark.parametrize(
    "klass,rnn_kwargs",
    itertools.product(
        [LSTM, GRU],
        [
            dict(batch_first=True, num_layers=1),
            dict(batch_first=False, num_layers=2),
        ],
    ),
)
def test_zero_length_sequence(klass, rnn_kwargs):
    rnn = klass(input_size=2, hidden_size=5, **rnn_kwargs)
    x = torch.rand(100, 3, 2)
    lengths = torch.randint(0, 3, size=([3, 100][rnn_kwargs["batch_first"]],))
    _, hidden_state = rnn(x, lengths=lengths, enforce_sorted=False)
    init_hidden_state = rnn.init_hidden_state(x)

    if isinstance(hidden_state, torch.Tensor):
        hidden_state = [hidden_state]
        init_hidden_state = [init_hidden_state]

    for idx in range(len(hidden_state)):
        assert (
            hidden_state[idx].size() == init_hidden_state[idx].size()
        ), "Hidden state sizes should be equal"
        assert (hidden_state[idx][:, lengths == 0] == 0).all() and (
            hidden_state[idx][:, lengths > 0] != 0
        ).all(), "Hidden state should be zero for zero-length sequences"


@pytest.mark.parametrize("klass", [LSTM, GRU])
def test_zero_length_preserves_provided_initial_state(klass):
    """Zero-length sequences must keep the matching provided initial state.

    ``handle_no_encoding`` used ``masked_scatter``, which copies a flattened
    prefix of the source into True positions rather than the same batch rows.
    LSTM also scattered the hidden tensor into the cell slot. Callers such as
    ``TemporalFusionTransformer`` pass a distinct initial hidden and cell from
    ``static_context_initial_hidden_lstm`` / ``static_context_initial_cell_lstm``.
    """
    torch.manual_seed(0)
    batch_size = 4
    seq_len = 5
    input_size = 2
    hidden_size = 6
    rnn = klass(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    x = torch.randn(batch_size, seq_len, input_size)
    lengths = torch.tensor([5, 0, 3, 0])
    zero = lengths == 0
    h0 = torch.randn(1, batch_size, hidden_size)

    if klass is LSTM:
        c0 = torch.randn(1, batch_size, hidden_size)
        assert not torch.allclose(
            h0[:, zero], c0[:, zero]
        ), "Test setup requires distinct initial hidden and cell states"
        _, (hn, cn) = rnn(x, hx=(h0, c0), lengths=lengths, enforce_sorted=False)
        assert torch.allclose(hn[:, zero], h0[:, zero])
        assert torch.allclose(cn[:, zero], c0[:, zero])
    else:
        _, hn = rnn(x, hx=h0, lengths=lengths, enforce_sorted=False)
        assert torch.allclose(hn[:, zero], h0[:, zero])
