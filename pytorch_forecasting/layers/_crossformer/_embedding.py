"""
Dimension-Segment-Wise (DSW) Embedding from Crossformer.
"""

import torch.nn as nn


class CrossFormerDSWEmbedding(nn.Module):
    """
    Dimension-Segment-Wise Embedding.

    Splits the input series of each variable into non-overlapping segments
    and projects each segment into ``d_model`` dimensions.

    Parameters
    ----------
    seg_len : int
        Length of each segment.
    d_model : int
        Dimension of the model.
    """

    def __init__(self, seg_len, d_model):
        super().__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, time, dimensions)``.

        Returns
        -------
        torch.Tensor
            Embedded segments of shape ``(B, dimensions, segments, d_model)``.
        """
        batch, ts_len, ts_dim = x.shape
        seg_num = ts_len // self.seg_len

        x_segment = x.reshape(batch, seg_num, self.seg_len, ts_dim)
        x_segment = x_segment.permute(0, 3, 1, 2)
        x_segment = x_segment.reshape(-1, self.seg_len)

        x_embed = self.linear(x_segment)
        x_embed = x_embed.reshape(batch, ts_dim, seg_num, -1)

        return x_embed
