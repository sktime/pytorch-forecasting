import math

import torch
import torch.nn as nn


class mLSTMCell(nn.Module):
    """Implements the Matrix Long Short-Term Memory (mLSTM) Cell.

    Implements the mLSTM algorithm as described in the paper:
    (https://arxiv.org/pdf/2407.10240).

    Parameters
    ----------
    input_size : int
        Size of the input feature vector.
    hidden_size : int
        Number of hidden units in the LSTM cell.
    dropout : float, optional
        Dropout rate applied to inputs and hidden states, by default 0.2.
    layer_norm : bool, optional
        If True, apply Layer Normalization to gates and interactions, by default True.

    Attributes
    ----------
    Wq : nn.Linear
        Linear layer for computing the query vector.
    Wk : nn.Linear
        Linear layer for computing the key vector.
    Wv : nn.Linear
        Linear layer for computing the value vector.
    Wi : nn.Linear
        Linear layer for the input gate.
    Wf : nn.Linear
        Linear layer for the forget gate.
    Wo : nn.Linear
        Linear layer for the output gate.
    dropout : nn.Dropout
        Dropout regularization layer.
    ln_q, ln_k, ln_v, ln_i, ln_f, ln_o : nn.LayerNorm
        Optional layer normalization layers for respective computations.
    """

    def __init__(self, input_size, hidden_size, dropout=0.2, layer_norm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm

        self.Wq = nn.Linear(input_size, hidden_size)
        self.Wk = nn.Linear(input_size, hidden_size)
        self.Wv = nn.Linear(input_size, hidden_size)

        self.Wi = nn.Linear(input_size, hidden_size)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Wo = nn.Linear(input_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        if layer_norm:
            self.ln_q = nn.LayerNorm(hidden_size)
            self.ln_k = nn.LayerNorm(hidden_size)
            self.ln_v = nn.LayerNorm(hidden_size)
            self.ln_i = nn.LayerNorm(hidden_size)
            self.ln_f = nn.LayerNorm(hidden_size)
            self.ln_o = nn.LayerNorm(hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h_prev, c_prev, n_prev):
        """Compute the next hidden, cell, and normalized states in the mLSTM cell.

        Parameters
        ----------
        x : torch.Tensor
            The number of features in the input.
        h_prev : torch.Tensor
            Previous hidden state
        c_prev : torch.Tensor
            Previous cell state
        n_prev : torch.Tensor
            Previous normalized state

        Returns
        -------
        tuple of torch.Tensor:
        h : torch.Tensor
            Current hidden state
        c : torch.Tensor
            Current cell state
        n : torch.Tensor
            Current normalized state
        """

        batch_size = x.size(0)
        assert x.dim() == 2, (
            f"Input should be 2D (batch_size, input_size), got {x.dim()}D"
        )
        assert h_prev.size() == (
            batch_size,
            self.hidden_size,
        ), f"h_prev shape mismatch: {h_prev.size()}"
        assert c_prev.size() == (
            batch_size,
            self.hidden_size,
        ), f"c_prev shape mismatch: {c_prev.size()}"
        assert n_prev.size() == (
            batch_size,
            self.hidden_size,
        ), f"n_prev shape mismatch: {n_prev.size()}"

        x = self.dropout(x)
        h_prev = self.dropout(h_prev)

        q = self.Wq(x)
        k = self.Wk(x) / math.sqrt(self.hidden_size)
        v = self.Wv(x)

        if self.layer_norm:
            q = self.ln_q(q)
            k = self.ln_k(k)
            v = self.ln_v(v)

        i = self.sigmoid(self.ln_i(self.Wi(x)) if self.layer_norm else self.Wi(x))
        f = self.sigmoid(self.ln_f(self.Wf(x)) if self.layer_norm else self.Wf(x))
        o = self.sigmoid(self.ln_o(self.Wo(x)) if self.layer_norm else self.Wo(x))

        k_expanded = k.unsqueeze(-1)
        v_expanded = v.unsqueeze(-2)

        kv_interaction = k_expanded @ v_expanded

        kv_sum = kv_interaction.sum(dim=1)

        c = f * c_prev + i * kv_sum
        n = f * n_prev + i * k

        epsilon = 1e-8
        normalized_n = n / (torch.norm(n, dim=-1, keepdim=True) + epsilon)
        h = o * self.tanh(c * normalized_n)

        return h, c, n

    def init_hidden(self, batch_size, device=None):
        """
        Initialize hidden, cell, and normalization states.
        """
        if device is None:
            device = next(self.parameters()).device
        shape = (batch_size, self.hidden_size)
        return (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )
