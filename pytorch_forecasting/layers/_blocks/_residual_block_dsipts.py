import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self, in_size: int, out_size: int, dropout_rate: float, activation_fun: str = ""
    ):
        """Residual Block as basic layer of the architecture.

        MLP with one hidden layer, activation and skip connection
        Basically dimension d_model, but better if input_dim and output_dim are explicit

        in_size and out_size to handle dimensions at different stages of the NN

        Parameters
        ----------
        in_size: int
            input size
        out_size: int
            output size
        dropout_rate: float
            dropout
        activation_fun: str, Optional
            activation function to use in the Residual Block. Defaults to nn.ReLU.
        """  # noqa: E501
        import ast

        super().__init__()

        self.direct_linear = nn.Linear(in_size, out_size, bias=False)

        if activation_fun == "":
            self.act = nn.ReLU()
        else:
            activation = ast.literal_eval(activation_fun)
            self.act = activation()
        self.lin = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.final_norm = nn.LayerNorm(out_size)

    def forward(self, x, apply_final_norm=True):
        direct_x = self.direct_linear(x)

        x = self.dropout(self.lin(self.act(x)))

        out = x + direct_x
        if apply_final_norm:
            return self.final_norm(out)
        return out
