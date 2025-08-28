import torch
import torch.nn as nn

from pytorch_forecasting.layers._recurrent._slstm.layer import sLSTMLayer


class sLSTMNetwork(nn.Module):
    """Implements the Stabilized LSTM Network with multiple sLSTM layers.

    This network combines sLSTM layers with a fully connected output layer for
    prediction.

    Parameters
    ----------
    input_size : int
        Number of features in the input.
    hidden_size : int
        Number of features in the hidden state of each sLSTM layer.
    num_layers : int
        Number of stacked sLSTM layers in the network.
    output_size : int
        Number of features in the output prediction.
    dropout : float, optional
        Dropout probability for the input of each sLSTM layer, by default 0.0.
    use_layer_norm : bool, optional
        Whether to use layer normalization in each sLSTM layer, by default True.

    Attributes
    ----------
    slstm_layer : sLSTMLayer
        Stacked sLSTM layers used for processing input sequences.
    fc : nn.Linear
        Fully connected layer to generate the final output predictions.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        dropout=0.0,
        use_layer_norm=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        self.slstm_layer = sLSTMLayer(
            input_size,
            hidden_size,
            num_layers,
            dropout,
            use_layer_norm,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None, c=None):
        """
        Forward pass through the sLSTM network.

        Parameters
        ----------
        x : torch.Tensor
           The number of features in the input.
        h : list of torch.Tensor, optional
            Initial hidden states for each layer.
            If None, hidden states are initialized to zeros.
        c : list of torch.Tensor, optional
            Initial cell states for each layer.
            If None, cell states are initialized to zeros.

        Returns
        -------
        output : torch.Tensor
            Tensor containing the final output predictions.
        (h, c) : tuple of lists
            Final hidden and cell states for each layer.
        """
        output, (h, c) = self.slstm_layer(x, h, c)
        output = self.fc(output[-1])
        return output, (h, c)

    def init_hidden(self, batch_size, device=None):
        """Initialize hidden and cell states for the entire network."""
        if device is None:
            device = next(self.parameters()).device
        return self.slstm_layer.init_hidden(batch_size, device=device)
