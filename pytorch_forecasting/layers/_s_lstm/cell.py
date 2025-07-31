import math

import torch
import torch.nn as nn


class sLSTMCell(nn.Module):
    """Implements the stabilized LSTM cell

    Implements the sLSTM algorithm as described in the paper:
    (https://arxiv.org/pdf/2407.10240).

    Parameters
    ----------
    input_size : int
        Number of input features for the cell.
    hidden_size : int
        Number of features in the hidden state of the cell.
    dropout : float, optional
        Dropout probability for the cell's input and hidden state, by default 0.0.
    use_layer_norm : bool, optional
        Whether to use layer normalization for the cell's internal computations,
        by default True.

    Attributes
    ----------
    input_weights : nn.Linear
        Linear layer for processing input features into gate computations.
    hidden_weights : nn.Linear
        Linear layer for processing hidden state features into gate computations.
    ln_cell : nn.LayerNorm
        Layer normalization for the cell state, applied if use_layer_norm is True.
    ln_hidden : nn.LayerNorm
        Layer normalization for the output hidden state,
        applied if use_layer_norm is True.
    ln_input : nn.LayerNorm
        Layer normalization for input gates, applied if use_layer_norm is True.
    ln_hidden_update : nn.LayerNorm
        Layer normalization for hidden state gates, applied if use_layer_norm is True.
    dropout_layer : nn.Dropout
        Dropout layer applied to inputs and hidden states.
    grad_clip : float
        Gradient clipping threshold to improve training stability.
    eps : float
        Small constant for numerical stability in calculations.
    tanh : nn.Tanh
        Tanh activation function.
    sigmoid : nn.Sigmoid
        Sigmoid activation function.
    """

    def __init__(self, input_size, hidden_size, dropout=0.0, use_layer_norm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.eps = 1e-6

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        if use_layer_norm:
            self.ln_cell = nn.LayerNorm(hidden_size)
            self.ln_hidden = nn.LayerNorm(hidden_size)
            self.ln_input = nn.LayerNorm(4 * hidden_size)
            self.ln_hidden_update = nn.LayerNorm(4 * hidden_size)

        self.dropout_layer = nn.Dropout(dropout)

        self.reset_parameters()

        self.grad_clip = 5.0

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        """Initialize parameters using Xavier/Glorot initialization"""
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    def normalized_exp_gate(self, pre_gate):
        """Compute normalized exponential gate activation"""
        centered = pre_gate - torch.mean(pre_gate, dim=1, keepdim=True)
        exp_val = torch.exp(torch.clamp(centered, min=-5.0, max=5.0))
        normalizer = torch.sum(exp_val, dim=1, keepdim=True) + self.eps
        return exp_val / normalizer

    def forward(self, x, h_prev, c_prev):
        """Forward pass with stabilized exponential gating.

        Parameters
        ----------
        x : torch.Tensor
            The number of features in the input.
        h_prev : torch.Tensor
            Previous hidden state tensor.
        c_prev : torch.Tensor
            Previous cell state tensor.

        Returns
        -------
        h : torch.Tensor
            Updated hidden state tensor.
        c : torch.Tensor
            Updated cell state tensor.
        """

        x = self.dropout_layer(x)
        h_prev = self.dropout_layer(h_prev)

        gates_x = self.input_weights(x)
        gates_h = self.hidden_weights(h_prev)

        if self.use_layer_norm:
            gates_x = self.ln_input(gates_x)
            gates_h = self.ln_hidden_update(gates_h)

        gates = gates_x + gates_h
        i, f, g, o = gates.chunk(4, dim=1)

        i = self.normalized_exp_gate(i)
        f = self.normalized_exp_gate(f)
        gate_sum = i + f
        i = i / (gate_sum + self.eps)
        f = f / (gate_sum + self.eps)

        c_tilde = self.tanh(g)
        c = f * c_prev + i * c_tilde
        if self.use_layer_norm:
            c = self.ln_cell(c)

        o = self.sigmoid(o)
        c_out = self.tanh(c)
        if self.use_layer_norm:
            c_out = self.ln_hidden(c_out)
        h = o * c_out

        return h, c

    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.hidden_size, device=device),
            torch.zeros(batch_size, self.hidden_size, device=device),
        )
