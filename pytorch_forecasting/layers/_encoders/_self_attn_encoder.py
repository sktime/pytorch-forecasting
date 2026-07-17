"""
Self-attention-only Encoder for PatchTST and similar encoder-only models.

Adapted from thuml/Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/layers/Transformer_EncDec.py
"""

import torch
import torch.nn as nn


class SelfAttnEncoder(nn.Module):
    """
    A stack of ``SelfAttnEncoderLayer`` modules.

    Intended for models (like PatchTST) that use an encoder-only architecture
    where every layer processes its input with self-attention only (no decoder,
    no cross-attention).

    Parameters
    ----------
    attn_layers : list[nn.Module]
        A list of ``SelfAttnEncoderLayer`` instances to stack.
    norm_layer : nn.Module, optional
        Normalization applied after the final encoder layer.
        PatchTST uses ``nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model),
        Transpose(1, 2))`` here so that BatchNorm operates on the channel
        dimension. Defaults to ``None``.
    """

    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        Forward pass through all stacked encoder layers.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch_size, seq_len, d_model)``.
        attn_mask : optional
            Attention mask passed to each layer (usually ``None`` for PatchTST).

        Returns
        -------
        x : torch.Tensor
            Encoded representation of shape ``(batch_size, seq_len, d_model)``.
        attns : list[torch.Tensor or None]
            One attention-weight tensor per layer (or ``None`` if
            ``output_attention=False`` in ``FullAttention``).
        """
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
