import torch
import torch.nn as nn

from pytorch_forecasting.models.x_lstm_time.s_lstm.cell import sLSTMCell


class sLSTMLayer(nn.Module):
    """
    Enhanced s_lstm Layer that supports multiple s_lstm cells across timesteps and residual connections.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        dropout=0.0,
        use_layer_norm=True,
        use_residual=True,
        device=None,
    ):
        super(sLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.input_projection = None
        if self.use_residual and input_size != hidden_size:
            self.input_projection = nn.Linear(input_size, hidden_size, bias=False).to(
                self.device
            )

        self.cells = nn.ModuleList(
            [
                sLSTMCell(
                    input_size if layer == 0 else hidden_size,
                    hidden_size,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                    device=self.device,
                )
                for layer in range(num_layers)
            ]
        )

        if self.use_layer_norm:
            self.layer_norm_layers = nn.ModuleList(
                [nn.LayerNorm(hidden_size).to(self.device) for _ in range(num_layers)]
            )

    def forward(self, x, h=None, c=None):
        """
        Forward pass through the s_lstm layer for each time step in sequence.
        Args:
            x: input tensor (seq_len, batch_size, input_size)
            h: initial hidden states (num_layers, batch_size, hidden_size)
            c: initial cell states (num_layers, batch_size, hidden_size)
        Returns:
            output: tensor of hidden states (seq_len, batch_size, hidden_size)
            (h, c): final hidden and cell states
        """
        seq_len, batch_size, _ = x.size()

        if h is None or c is None:
            h, c = self.init_hidden(batch_size)

        x = x.to(self.device)
        h = [hi.to(self.device) for hi in h]
        c = [ci.to(self.device) for ci in c]

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

    def init_hidden(self, batch_size):
        """Initialize hidden and cell states for each layer."""
        return (
            [
                torch.zeros(batch_size, self.hidden_size, device=self.device)
                for _ in range(self.num_layers)
            ],
            [
                torch.zeros(batch_size, self.hidden_size, device=self.device)
                for _ in range(self.num_layers)
            ],
        )
