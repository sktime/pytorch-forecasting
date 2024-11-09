import torch.nn as nn
import torch
from .layer import mLSTMLayer


class mLSTMNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        dropout=0.0,
        use_layer_norm=True,
        use_residual=True,
        device=None,
    ):
        super(mLSTMNetwork, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mlstm_layer = mLSTMLayer(
            input_size, hidden_size, num_layers, dropout, use_layer_norm, use_residual, self.device
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None, c=None, n=None):
        """
        Forward pass through the mLSTM network.
        """
        output, (h, c, n) = self.mlstm_layer(x, h, c, n)

        output = self.fc(output[-1])

        return output, (h, c, n)

    def init_hidden(self, batch_size):
        """Initialize hidden, cell, and normalization states."""
        return self.mlstm_layer.init_hidden(batch_size)
