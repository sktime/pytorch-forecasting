"""Token embedding module.

Implements a token embedding via a 1D convolution over input features.
The convolution uses circular padding so the receptive field wraps around
the sequence dimension.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token embedding using a 1D convolution.

    Args:
        c_in (int): number of input channels/features.
        d_model (int): output embedding dimension.

    The convolution uses kernel size 3 and circular padding; weights are
    initialized with Kaiming normal initialization.
    """

    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        """Apply convolutional token embedding.

        Args:
            x (torch.Tensor): input tensor of shape `[batch, seq_len, c_in]`.

        Returns:
            torch.Tensor: embedded tensor of shape `[batch, seq_len, d_model]`.
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
