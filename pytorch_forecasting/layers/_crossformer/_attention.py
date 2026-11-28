"""
Two-Stage Attention (TSA) Layer from Crossformer.
"""

import torch
import torch.nn as nn

from pytorch_forecasting.layers._attention import AttentionLayer, FullAttention


class CrossFormerTwoStageAttention(nn.Module):
    """
    The Two Stage Attention (TSA) Layer.

    Applies multi-head self-attention across time for each variable,
    then uses learned routers to exchange information across variables.

    Parameters
    ----------
    seg_num : int
        Number of segments.
    factor : int
        Number of routers in the cross-dimension stage.
    d_model : int
        Dimension of the model.
    n_heads : int
        Number of attention heads.
    d_ff : int, optional
        Dimension of the feed-forward network. Defaults to ``4 * d_model``.
    dropout : float, default=0.1
        Dropout rate.
    """

    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.time_attention = AttentionLayer(
            FullAttention(
                mask_flag=False, use_efficient_attention=True, attention_dropout=dropout
            ),
            d_model=d_model,
            n_heads=n_heads,
        )

        self.dim_sender = AttentionLayer(
            FullAttention(
                mask_flag=False, use_efficient_attention=True, attention_dropout=dropout
            ),
            d_model=d_model,
            n_heads=n_heads,
        )

        self.dim_receiver = AttentionLayer(
            FullAttention(
                mask_flag=False, use_efficient_attention=True, attention_dropout=dropout
            ),
            d_model=d_model,
            n_heads=n_heads,
        )

        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch_size, data_dim, seg_num, d_model)``.

        Returns
        -------
        torch.Tensor
            Output of the same shape as the input.
        """
        # Cross Time Stage: Directly apply MSA to each dimension
        batch, ts_d, seg_num, d_model = x.shape
        time_in = x.reshape(batch * ts_d, seg_num, d_model)

        time_enc, _ = self.time_attention(time_in, time_in, time_in, attn_mask=None)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        dim_send = dim_in.reshape(batch, ts_d, seg_num, d_model)
        dim_send = dim_send.permute(0, 2, 1, 3)

        dim_send = dim_send.reshape(batch * seg_num, ts_d, d_model)

        batch_router = (
            self.router.unsqueeze(0)
            .expand(batch, -1, -1, -1)
            .reshape(batch * seg_num, -1, d_model)
        )

        dim_buffer, _ = self.dim_sender(
            batch_router, dim_send, dim_send, attn_mask=None
        )

        dim_receive, _ = self.dim_receiver(
            dim_send, dim_buffer, dim_buffer, attn_mask=None
        )

        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)

        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = dim_enc.reshape(batch, seg_num, ts_d, d_model)

        final_out = final_out.permute(0, 2, 1, 3).contiguous()
        return final_out
