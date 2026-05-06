"""
Implementations of flexible GRU and LSTM that can handle sequences of length 0.
"""

from abc import ABC, abstractmethod
from typing import Union

import torch
from torch import nn
from torch.nn.utils import rnn

HiddenState = tuple[torch.Tensor, torch.Tensor] | torch.Tensor


class RNN(ABC, nn.RNNBase):
    """
    Base class flexible RNNs.

    Forward function can handle sequences of length 0.
    """

    @abstractmethod
    def handle_no_encoding(
        self,
        hidden_state: HiddenState,
        no_encoding: torch.BoolTensor,
        initial_hidden_state: HiddenState,
    ) -> HiddenState:
        """
        Mask the hidden_state where there is no encoding.

        Parameters
        ----------
        hidden_state : HiddenState
            Hidden state where some entries need replacement.
        no_encoding : torch.BoolTensor
            Positions that need replacement.
        initial_hidden_state : HiddenState
            Hidden state to use for replacement.

        Returns
        -------
        HiddenState
            Hidden state with propagated initial hidden state where appropriate.
        """
        pass

    @abstractmethod
    def init_hidden_state(self, x: torch.Tensor) -> HiddenState:
        """
        Initialise a hidden_state.

        Parameters
        ----------
        x : torch.Tensor
            Network input.

        Returns
        -------
        HiddenState
            Default (zero-like) hidden state.
        """
        pass

    @abstractmethod
    def repeat_interleave(
        self, hidden_state: HiddenState, n_samples: int
    ) -> HiddenState:
        """
        Duplicate the hidden_state n_samples times.

        Parameters
        ----------
        hidden_state : HiddenState
            Hidden state to repeat.
        n_samples : int
            Number of repetitions.

        Returns
        -------
        HiddenState
            Repeated hidden state.
        """
        pass

    def forward(
        self,
        x: rnn.PackedSequence | torch.Tensor,
        hx: HiddenState = None,
        lengths: torch.LongTensor = None,
        enforce_sorted: bool = True,
    ) -> tuple[rnn.PackedSequence | torch.Tensor, HiddenState]:
        """
        Forward function of rnn that allows zero-length sequences.

        Functions as normal for RNN. Only changes output if lengths are defined.

        Parameters
        ----------
        x : rnn.PackedSequence or torch.Tensor
            Input to RNN. Either packed sequence or tensor of padded sequences.
        hx : HiddenState, optional
            Hidden state. Defaults to None.
        lengths : torch.LongTensor, optional
            Lengths of sequences. If not None, used to determine correct returned
            hidden state. Can contain zeros. Defaults to None.
        enforce_sorted : bool, optional
            If lengths are passed, determines if RNN expects them to be sorted.
            Defaults to True.

        Returns
        -------
        tuple of (rnn.PackedSequence or torch.Tensor, HiddenState)
            Output and hidden state. Output is a packed sequence if input
            was a packed sequence.
        """
        if isinstance(x, rnn.PackedSequence) or lengths is None:
            assert (
                lengths is None
            ), "cannot combine x of type PackedSequence with lengths argument"
            return super().forward(x, hx=hx)
        else:
            min_length = lengths.min()
            max_length = lengths.max()
            assert min_length >= 0, "sequence lengths must be great equals 0"

            if max_length == 0:
                hidden_state = self.init_hidden_state(x)
                if self.batch_first:
                    out = torch.zeros(
                        lengths.size(0),
                        x.size(1),
                        self.hidden_size,
                        dtype=x.dtype,
                        device=x.device,
                    )
                else:
                    out = torch.zeros(
                        x.size(0),
                        lengths.size(0),
                        self.hidden_size,
                        dtype=x.dtype,
                        device=x.device,
                    )
                return out, hidden_state
            else:
                pack_lengths = lengths.where(lengths > 0, torch.ones_like(lengths))
                packed_out, hidden_state = super().forward(
                    rnn.pack_padded_sequence(
                        x,
                        pack_lengths.cpu(),
                        enforce_sorted=enforce_sorted,
                        batch_first=self.batch_first,
                    ),
                    hx=hx,
                )
                # replace hidden cell with initial input if encoder_length
                # is zero to determine correct initial state
                if min_length == 0:
                    no_encoding = (lengths == 0)[
                        None, :, None
                    ]  # shape: n_layers * n_directions x batch_size x hidden_size
                    if hx is None:
                        initial_hidden_state = self.init_hidden_state(x)
                    else:
                        initial_hidden_state = hx
                    # propagate initial hidden state when sequence length was 0
                    hidden_state = self.handle_no_encoding(
                        hidden_state, no_encoding, initial_hidden_state
                    )

                # return unpacked sequence
                out, _ = rnn.pad_packed_sequence(
                    packed_out, batch_first=self.batch_first
                )
                return out, hidden_state


class LSTM(RNN, nn.LSTM):
    """LSTM that can handle zero-length sequences"""

    def handle_no_encoding(
        self,
        hidden_state: HiddenState,
        no_encoding: torch.BoolTensor,
        initial_hidden_state: HiddenState,
    ) -> HiddenState:
        hidden, cell = hidden_state
        hidden = hidden.masked_scatter(no_encoding, initial_hidden_state[0])
        cell = cell.masked_scatter(no_encoding, initial_hidden_state[0])
        return hidden, cell

    def init_hidden_state(self, x: torch.Tensor) -> HiddenState:
        num_directions = 2 if self.bidirectional else 1
        if self.batch_first:
            batch_size = x.size(0)
        else:
            batch_size = x.size(1)
        hidden = torch.zeros(
            (self.num_layers * num_directions, batch_size, self.hidden_size),
            device=x.device,
            dtype=x.dtype,
        )
        cell = torch.zeros(
            (self.num_layers * num_directions, batch_size, self.hidden_size),
            device=x.device,
            dtype=x.dtype,
        )
        return hidden, cell

    def repeat_interleave(
        self, hidden_state: HiddenState, n_samples: int
    ) -> HiddenState:
        hidden, cell = hidden_state
        hidden = hidden.repeat_interleave(n_samples, 1)
        cell = cell.repeat_interleave(n_samples, 1)
        return hidden, cell


class GRU(RNN, nn.GRU):
    """GRU that can handle zero-length sequences"""

    def handle_no_encoding(
        self,
        hidden_state: HiddenState,
        no_encoding: torch.BoolTensor,
        initial_hidden_state: HiddenState,
    ) -> HiddenState:
        return hidden_state.masked_scatter(no_encoding, initial_hidden_state)

    def init_hidden_state(self, x: torch.Tensor) -> HiddenState:
        if self.batch_first:
            batch_size = x.size(0)
        else:
            batch_size = x.size(1)
        num_directions = 2 if self.bidirectional else 1
        hidden = torch.zeros(
            (self.num_layers * num_directions, batch_size, self.hidden_size),
            device=x.device,
            dtype=x.dtype,
        )
        return hidden

    def repeat_interleave(
        self, hidden_state: HiddenState, n_samples: int
    ) -> HiddenState:
        return hidden_state.repeat_interleave(n_samples, 1)


def get_rnn(cell_type: type[RNN] | str) -> type[RNN]:
    """
    Get LSTM or GRU.

    Parameters
    ----------
    cell_type : type[RNN] or str
        RNN class or string identifier, either ``"LSTM"`` or ``"GRU"``.

    Returns
    -------
    type[RNN]
        Returns the GRU or LSTM RNN class.
    """
    if isinstance(cell_type, RNN):
        rnn = cell_type
    elif cell_type == "LSTM":
        rnn = LSTM
    elif cell_type == "GRU":
        rnn = GRU
    else:
        raise ValueError(
            f"RNN type {cell_type} is not supported. supported: [LSTM, GRU]"
        )
    return rnn
