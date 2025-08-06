import torch
import torch.nn as nn

from pytorch_forecasting.layers._recurrent._slstm.cell import sLSTMCell


class sLSTMLayer(nn.Module):
    """Implements the sLSTM Layer, which consists of multiple stacked sLSTM cells.

    This layer is designed for sequence modeling tasks, supporting multiple layers
    with optional residual connections and layer normalization.

    Parameters
    ----------
    input_size : int
        Number of features in the input.
    hidden_size : int
        Number of features in the hidden state of each sLSTM cell.
    num_layers : int, optional
        Number of stacked sLSTM layers, by default 1.
    dropout : float, optional
        Dropout probability for the input of each sLSTM cell, by default 0.0.
    use_layer_norm : bool, optional
        Whether to use layer normalization for each sLSTM cell, by default True.
    use_residual : bool, optional
        Whether to use residual connections in each sLSTM layer, by default True.

    Attributes
    ----------
    cells : nn.ModuleList
        List of sLSTMCell objects, one for each layer.
    input_projection : nn.Linear or None
        Linear layer for projecting input to match hidden state size,
        used when residual connections are enabled.
    layer_norm_layers : nn.ModuleList
        List of LayerNorm layers, one for each sLSTM layer (if use_layer_norm is True).
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        dropout=0.0,
        use_layer_norm=True,
        use_residual=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        self.input_projection = None
        if self.use_residual and input_size != hidden_size:
            self.input_projection = nn.Linear(input_size, hidden_size, bias=False)

        self.cells = nn.ModuleList(
            [
                sLSTMCell(
                    input_size if layer == 0 else hidden_size,
                    hidden_size,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
                for layer in range(num_layers)
            ]
        )

        if self.use_layer_norm:
            self.layer_norm_layers = nn.ModuleList(
                [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
            )

    def forward(self, x, h=None, c=None):
        """Forward pass through the sLSTM Layer.

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
            Tensor containing hidden states for each time step.
        (h, c) : tuple of lists
            Final hidden and cell states for each layer.
        """
        seq_len, batch_size, _ = x.size()

        if h is None or c is None:
            h, c = self.init_hidden(batch_size, device=x.device)

        outputs = []

        for t in range(seq_len):
            input_t = x[t]
            layer_input = input_t

            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](layer_input, h[layer], c[layer])

                if self.use_residual:
                    if layer == 0 and self.input_projection is not None:
                        residual = self.input_projection(layer_input)
                    else:
                        residual = (
                            layer_input
                            if (layer_input.size(-1) == self.hidden_size)
                            else 0
                        )
                    h[layer] = h[layer] + residual

                if self.use_layer_norm:
                    h[layer] = self.layer_norm_layers[layer](h[layer])

                layer_input = h[layer]

            outputs.append(h[-1])

        output = torch.stack(outputs)

        h = [hi.detach() for hi in h]
        c = [ci.detach() for ci in c]

        return output, (h, c)

    def init_hidden(self, batch_size, device=None):
        """Initialize hidden and cell states for each layer."""
        if device is None:
            device = next(self.parameters()).device
        return (
            [
                torch.zeros(batch_size, self.hidden_size, device=device)
                for _ in range(self.num_layers)
            ],
            [
                torch.zeros(batch_size, self.hidden_size, device=device)
                for _ in range(self.num_layers)
            ],
        )
