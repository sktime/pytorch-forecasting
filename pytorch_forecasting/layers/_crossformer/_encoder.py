"""
Hierarchical Encoder of Crossformer.
"""

import math

import torch
import torch.nn as nn

from pytorch_forecasting.layers._crossformer._attention import (
    CrossFormerTwoStageAttention,
)


class CrossFormerSegmentMerger(nn.Module):
    """
    Segment Merging Layer.

    The adjacent ``win_size`` segments in each dimension will be merged into
    one segment to get a representation of a coarser scale.

    Parameters
    ----------
    d_model : int
        Dimension of the model.
    win_size : int
        Number of adjacent segments merged at each step.
    norm_layer : nn.Module, default=nn.LayerNorm
        Normalization layer applied before the linear projection.
    """

    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, ts_d, L, d_model)``.

        Returns
        -------
        torch.Tensor
            Merged segments of shape ``(B, ts_d, L / win_size, d_model)``.
        """
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i :: self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]

        x = self.norm(x)
        x = self.linear_trans(x)

        return x


class CrossFormerScaleBlock(nn.Module):
    """
    One scale of the Crossformer encoder.

    Optionally merges segments to a coarser scale, then applies ``depth``
    TSA layers.

    Parameters
    ----------
    win_size : int
        Segment merge window; ``1`` disables merging.
    d_model : int
        Dimension of the model.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Dimension of the feed-forward network.
    depth : int
        Number of TSA layers in this scale.
    dropout : float
        Dropout rate.
    seg_num : int, default=10
        Number of segments at this scale.
    factor : int, default=10
        Number of routers in the TSA layer.
    """

    def __init__(
        self, win_size, d_model, n_heads, d_ff, depth, dropout, seg_num=10, factor=10
    ):
        super().__init__()

        if win_size > 1:
            self.merge_layer = CrossFormerSegmentMerger(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None

        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(
                CrossFormerTwoStageAttention(
                    seg_num, factor, d_model, n_heads, d_ff, dropout
                )
            )

    def forward(self, x):
        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x)

        return x


class CrossFormerEncoder(nn.Module):
    """
    The Hierarchical Encoder (HED) of Crossformer.

    Returns the input embedding and the output of every scale so that the
    decoder can cross-attend to each level.

    Parameters
    ----------
    e_blocks : int
        Number of encoder scales.
    win_size : int
        Segment merge window between scales.
    d_model : int
        Dimension of the model.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Dimension of the feed-forward network.
    block_depth : int
        Number of TSA layers per scale.
    dropout : float
        Dropout rate.
    in_seg_num : int, default=10
        Number of segments in the input.
    factor : int, default=10
        Number of routers in the TSA layer.
    """

    def __init__(
        self,
        e_blocks,
        win_size,
        d_model,
        n_heads,
        d_ff,
        block_depth,
        dropout,
        in_seg_num=10,
        factor=10,
    ):
        super().__init__()
        self.encode_blocks = nn.ModuleList()

        self.encode_blocks.append(
            CrossFormerScaleBlock(
                1, d_model, n_heads, d_ff, block_depth, dropout, in_seg_num, factor
            )
        )
        for i in range(1, e_blocks):
            self.encode_blocks.append(
                CrossFormerScaleBlock(
                    win_size,
                    d_model,
                    n_heads,
                    d_ff,
                    block_depth,
                    dropout,
                    math.ceil(in_seg_num / win_size**i),
                    factor,
                )
            )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Embedded input of shape ``(B, ts_d, seg_num, d_model)``.

        Returns
        -------
        list of torch.Tensor
            The input embedding followed by the output of each scale.
        """
        encode_x = []
        encode_x.append(x)

        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)

        return encode_x
