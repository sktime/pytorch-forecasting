"""
Implementation of EncoderLayer for encoder-decoder architectures from `nn.Module`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    """
    Encoder layer for TsLib models.
    Args:
        self_attention (nn.Module): Self-attention mechanism.
        cross_attention (nn.Module, optional): Cross-attention mechanism.
        d_model (int): Dimension of the model.
        d_ff (int, optional):
            Dimension of the feedforward layer. Defaults to 4 * d_model.
        dropout (float): Dropout rate. Defaults to 0.1.
        activation (str): Activation function. Defaults to "relu".
        output_attention (Boolean, optional): Whether to output attention weights. Defaults to False.
    """  # noqa: E501

    def __init__(
        self,
        self_attention,
        cross_attention=None,
        d_model=512,
        d_ff=None,
        dropout=0.1,
        activation="relu",
        output_attention=False,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        if self.cross_attention is not None:
            self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.output_attention = output_attention

    def forward(
        self, x, cross=None, x_mask=None, cross_mask=None, tau=None, delta=None
    ):
        if self.output_attention:
            x, attn = self.self_attention(
                x, x, x, attn_mask=x_mask, tau=tau, delta=None
            )
        else:
            x = self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        x = x + self.dropout(x)
        y = x = self.norm1(x)
        if self.cross_attention is not None:
            B, L, D = cross.shape
            x_glb_ori = x[:, -1, :].unsqueeze(1)
            x_glb = torch.reshape(x_glb_ori, (B, -1, D))
            x_glb_attn = self.dropout(
                self.cross_attention(
                    x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
                )[0]
            )
            x_glb_attn = torch.reshape(
                x_glb_attn,
                (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2]),
            ).unsqueeze(1)
            x_glb = x_glb_ori + x_glb_attn
            x_glb = self.norm2(x_glb)

            y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        if self.output_attention:
            return self.norm3(x + y), attn
        return self.norm3(x + y)
