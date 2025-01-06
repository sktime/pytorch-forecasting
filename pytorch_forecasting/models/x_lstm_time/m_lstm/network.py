import torch
import torch.nn as nn

from pytorch_forecasting.models.x_lstm_time.m_lstm.layer import mLSTMLayer


class mLSTMNetwork(nn.Module):
    """Implements the mLSTM Network, a complete model based on stacked mLSTM layers.

    This network combines stacked mLSTM layers and a fully connected output layer.

    Parameters
    ----------
    input_size : int
        Number of features in the input.
    hidden_size : int
        Number of features in the hidden state of each mLSTM layer.
    num_layers : int
        Number of mLSTM layers to stack.
    output_size : int
        Number of features in the output.
    dropout : float, optional
        Dropout probability for the mLSTM layers, by default 0.0.
    use_layer_norm : bool, optional
        Whether to use layer normalization in the mLSTM layers, by default True.
    use_residual : bool, optional
        Whether to use residual connections in the mLSTM layers, by default True.
    device : torch.device, optional
        Device to run the computations on

    Attributes
    ----------
    mlstm_layer : mLSTMLayer
        Stacked mLSTM layers used for processing input sequences.
    fc : nn.Linear
        Fully connected layer to generate final output.


    """
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
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.mlstm_layer = mLSTMLayer(
            input_size,
            hidden_size,
            num_layers,
            dropout,
            use_layer_norm,
            use_residual,
            self.device,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None, c=None, n=None):
        """Forward pass through the mLSTM Network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size).
        h : torch.Tensor, optional
            Initial hidden states for all layers, shape (num_layers, batch_size, hidden_size).
            If None, initialized to zeros, by default None.
        c : torch.Tensor, optional
            Initial cell states for all layers, shape (num_layers, batch_size, hidden_size).
            If None, initialized to zeros, by default None.
        n : torch.Tensor, optional
            Initial normalized states for all layers, shape (num_layers, batch_size, hidden_size).
            If None, initialized to zeros, by default None.

        Returns
        -------
        tuple
            output : torch.Tensor
                Final output tensor from the fully connected layer, shape (batch_size, output_size).
            (h, c, n) : tuple of torch.Tensor
                Final hidden, cell, and normalized states for all layers:
                - h : torch.Tensor, shape (num_layers, batch_size, hidden_size).
                - c : torch.Tensor, shape (num_layers, batch_size, hidden_size).
                - n : torch.Tensor, shape (num_layers, batch_size, hidden_size).
        """
        output, (h, c, n) = self.mlstm_layer(x, h, c, n)

        output = self.fc(output[-1])

        return output, (h, c, n)

    def init_hidden(self, batch_size):
        """Initialize hidden, cell, and normalization states."""
        return self.mlstm_layer.init_hidden(batch_size)
