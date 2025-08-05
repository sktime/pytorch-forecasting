import torch
import torch.nn as nn

from pytorch_forecasting.layers._mlstm.layer import mLSTMLayer


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
    ):
        super().__init__()

        self.mlstm_layer = mLSTMLayer(
            input_size,
            hidden_size,
            num_layers,
            dropout,
            use_layer_norm,
            use_residual,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None, c=None, n=None):
        """Forward pass through the mLSTM Network.

        Parameters
        ----------
        x : torch.Tensor
           The number of features in the input.
        h : torch.Tensor, optional
            Initial hidden states for all layers.
            If None, initialized to zeros, by default None.
        c : torch.Tensor, optional
            Initial cell states for all layers.
            If None, initialized to zeros, by default None.
        n : torch.Tensor, optional
            Initial normalized states for all layers.
            If None, initialized to zeros, by default None.

        Returns
        -------
        tuple
            output : torch.Tensor
                Final output tensor from the fully connected layer.
            (h, c, n) : tuple of torch.Tensor
                Final hidden, cell, and normalized states for all layers:
                - h : torch.Tensor
                - c : torch.Tensor
                - n : torch.Tensor
        """
        output, (h, c, n) = self.mlstm_layer(x, h, c, n)

        output = self.fc(output[-1])

        return output, (h, c, n)

    def init_hidden(self, batch_size, device=None):
        """Initialize hidden, cell, and normalization states."""
        if device is None:
            device = next(self.parameters()).device
        return self.mlstm_layer.init_hidden(batch_size, device=device)
