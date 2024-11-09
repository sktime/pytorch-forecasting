import torch.nn as nn
import torch
from pytorch_forecasting.models.xLSTMTime.sLSTM.layer import sLSTMLayer


class sLSTMNetwork(nn.Module):
    """
    Stabilized LSTM Network with multiple sLSTM layers.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0, use_layer_norm=True, device=None):
        super(sLSTMNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.slstm_layer = sLSTMLayer(input_size, hidden_size, num_layers, dropout, use_layer_norm, device=self.device)
        self.fc = nn.Linear(hidden_size, output_size).to(self.device)

    def forward(self, x, h=None, c=None):
        """
        Forward pass through the sLSTM network.
        Args:
            x: input tensor (seq_len, batch_size, input_size)
            h: initial hidden states (num_layers, batch_size, hidden_size)
            c: initial cell states (num_layers, batch_size, hidden_size)
        Returns:
            output: tensor of output predictions (seq_len, batch_size, output_size)
            (h, c): final hidden and cell states
        """
        output, (h, c) = self.slstm_layer(x, h, c)
        output = self.fc(output[-1])
        return output, (h, c)

    def init_hidden(self, batch_size):
        """Initialize hidden and cell states for the entire network."""
        return self.slstm_layer.init_hidden(batch_size)
