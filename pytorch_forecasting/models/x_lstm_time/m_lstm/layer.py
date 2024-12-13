import torch
import torch.nn as nn
from pytorch_forecasting.models.x_lstm_time.m_lstm.cell import mLSTMCell


class mLSTMLayer(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, dropout=0.2, layer_norm=True, residual_conn=True, device=None
    ):
        super(mLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        self.residual_conn = residual_conn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dropout = nn.Dropout(dropout).to(self.device)

        self.cells = nn.ModuleList(
            [
                mLSTMCell(input_size if i == 0 else hidden_size, hidden_size, dropout, layer_norm, self.device)
                for i in range(num_layers)
            ]
        )

    def init_hidden(self, batch_size):
        """
        Initialize hidden, cell, and normalization states for all layers.
        """
        hidden_states, cell_states, norm_states = zip(
            *[self.cells[i].init_hidden(batch_size) for i in range(self.num_layers)]
        )

        return (
            torch.stack(hidden_states).to(self.device),
            torch.stack(cell_states).to(self.device),
            torch.stack(norm_states).to(self.device),
        )

    def forward(self, x, h=None, c=None, n=None):
        """
        Forward pass for the m_lstm layer.
        """

        x = x.to(self.device).transpose(0, 1)
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

            h = torch.stack(next_hidden_states).to(self.device)
            c = torch.stack(next_cell_states).to(self.device)
            n = torch.stack(next_norm_states).to(self.device)

            outputs.append(h[-1])

        output = torch.stack(outputs, dim=1)

        output = output.transpose(0, 1)

        return output, (h, c, n)
