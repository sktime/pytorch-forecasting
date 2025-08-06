import torch
import torch.nn as nn

from pytorch_forecasting.layers._recurrent._mlstm.cell import mLSTMCell


class mLSTMLayer(nn.Module):
    """Implements a mLSTM (Matrix LSTM) layer.

    This class stacks multiple mLSTM cells to form a deep recurrent layer.
    It supports residual connections, layer normalization, and dropout.

    Parameters
    ----------
    input_size : int
        The number of features in the input.
    hidden_size : int
        The number of features in the hidden state.
    num_layers : int
        The number of mLSTM layers to stack.
    dropout : float, optional
        Dropout probability applied to the inputs and intermediate layers,
        by default 0.2.
    layer_norm : bool, optional
        Whether to use layer normalization in each mLSTM cell, by default True.
    residual_conn : bool, optional
        Whether to enable residual connections between layers, by default True.

    Attributes
    ----------
    cells : nn.ModuleList
        A list containing all mLSTM cells in the layer.
    dropout : nn.Dropout
        Dropout layer applied between layers.

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout=0.2,
        layer_norm=True,
        residual_conn=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        self.residual_conn = residual_conn
        self.dropout = nn.Dropout(dropout)

        self.cells = nn.ModuleList(
            [
                mLSTMCell(
                    input_size if i == 0 else hidden_size,
                    hidden_size,
                    dropout,
                    layer_norm,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x, h=None, c=None, n=None):
        """Forward pass through the mLSTM layer.

        Parameters
        ----------
        x : torch.Tensor
            The number of features in the input.
        h : torch.Tensor, optional
            Initial hidden states for all layers
            If None, initialized to zeros, by default None.
        c : torch.Tensor, optional
            Initial cell states for all layers
            If None, initialized to zeros, by default None.
        n : torch.Tensor, optional
            Initial normalized states for all layers
            If None, initialized to zeros, by default None.

        Returns
        -------
        tuple
            output : torch.Tensor
                Final output tensor from the last layer
            (h, c, n) : tuple of torch.Tensor
                Final hidden, cell, and normalized states for all layers:
                - h : torch.Tensor
                - c : torch.Tensor
                - n : torch.Tensor
        """

        x = x.transpose(0, 1)
        batch_size, seq_len, _ = x.size()

        if h is None or c is None or n is None:
            h, c, n = self.init_hidden(batch_size)

        outputs = []

        for t in range(seq_len):
            layer_input = x[:, t, :]
            next_hidden_states = []
            next_cell_states = []
            next_norm_states = []

            for i, cell in enumerate(self.cells):
                h_i, c_i, n_i = cell(layer_input, h[i], c[i], n[i])

                if self.residual_conn and i > 0:
                    h_i = h_i + layer_input

                layer_input = h_i

                next_hidden_states.append(h_i)
                next_cell_states.append(c_i)
                next_norm_states.append(n_i)

            h = torch.stack(next_hidden_states)
            c = torch.stack(next_cell_states)
            n = torch.stack(next_norm_states)

            outputs.append(h[-1])

        output = torch.stack(outputs, dim=1)

        output = output.transpose(0, 1)

        return output, (h, c, n)

    def init_hidden(self, batch_size, device=None):
        """
        Initialize hidden, cell, and normalization states for all layers.
        """
        if device is None:
            device = next(self.parameters()).device
        hidden_states, cell_states, norm_states = zip(
            *[
                self.cells[i].init_hidden(batch_size, device=device)
                for i in range(self.num_layers)
            ]
        )

        return (
            torch.stack(hidden_states),
            torch.stack(cell_states),
            torch.stack(norm_states),
        )
