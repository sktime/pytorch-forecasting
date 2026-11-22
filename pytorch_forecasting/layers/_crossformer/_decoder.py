"""
Hierarchical Decoder of Crossformer.
"""

import torch
import torch.nn as nn

from pytorch_forecasting.layers._attention import AttentionLayer, FullAttention
from pytorch_forecasting.layers._crossformer._attention import (
    CrossFormerTwoStageAttention,
)


class CrossFormerDecoderLayer(nn.Module):
    """
    One decoder layer of Crossformer; each layer makes a prediction at its scale.

    Parameters
    ----------
    seg_len : int
        Length of each segment in the output sequence.
    d_model : int
        Dimension of the model.
    n_heads : int
        Number of attention heads.
    d_ff : int, optional
        Dimension of the feed-forward network. Defaults to ``4 * d_model``.
    dropout : float, default=0.1
        Dropout rate.
    out_seg_num : int, default=10
        Number of segments in the decoder input/output.
    factor : int, default=10
        Number of routers in the cross-dimension stage of the TSA layer.
    """

    def __init__(
        self,
        seg_len,
        d_model,
        n_heads,
        d_ff=None,
        dropout=0.1,
        out_seg_num=10,
        factor=10,
    ):
        super().__init__()
        self.self_attention = CrossFormerTwoStageAttention(
            out_seg_num, factor, d_model, n_heads, d_ff, dropout
        )
        self.cross_attention = AttentionLayer(
            FullAttention(
                mask_flag=False,
                use_efficient_attention=True,
                attention_dropout=dropout,
            ),
            d_model=d_model,
            n_heads=n_heads,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        """
        Parameters
        ----------
        x : torch.Tensor
            Output of the last decoder layer, shape ``(B, ts_d, out_seg_num, d_model)``.
        cross : torch.Tensor
            Output of the corresponding encoder layer,
            shape ``(B, ts_d, in_seg_num, d_model)``.

        Returns
        -------
        dec_output : torch.Tensor
            Decoder output, shape ``(B, ts_d, out_seg_num, d_model)``.
        layer_predict : torch.Tensor
            Prediction at this scale, shape ``(B, ts_d * out_seg_num, seg_len)``.
        """
        batch, ts_d = x.shape[0], x.shape[1]

        x = self.self_attention(x)
        seg_num, d_model = x.shape[2], x.shape[3]
        x = x.reshape(batch * ts_d, seg_num, d_model)

        cross = cross.reshape(batch * ts_d, cross.shape[2], cross.shape[3])
        tmp, _ = self.cross_attention(x, cross, cross, attn_mask=None)
        x = x + self.dropout(tmp)
        y = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)

        dec_output = dec_output.reshape(batch, ts_d, seg_num, d_model)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = layer_predict.flatten(1, 2)

        return dec_output, layer_predict


class CrossFormerDecoder(nn.Module):
    """
    The decoder of Crossformer, making the final prediction by adding up
    predictions at each scale.

    Parameters
    ----------
    seg_len : int
        Length of each segment in the output sequence.
    d_layers : int
        Number of decoder layers.
    d_model : int
        Dimension of the model.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Dimension of the feed-forward network.
    dropout : float
        Dropout rate.
    router : bool, default=False
        Unused, kept for compatibility with the reference implementation.
    out_seg_num : int, default=10
        Number of segments in the decoder input/output.
    factor : int, default=10
        Number of routers in the cross-dimension stage of the TSA layer.
    """

    def __init__(
        self,
        seg_len,
        d_layers,
        d_model,
        n_heads,
        d_ff,
        dropout,
        router=False,
        out_seg_num=10,
        factor=10,
    ):
        super().__init__()
        self.seg_len = seg_len
        self.decode_layers = nn.ModuleList()
        for _ in range(d_layers):
            self.decode_layers.append(
                CrossFormerDecoderLayer(
                    seg_len,
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    out_seg_num,
                    factor,
                )
            )

    def forward(self, x, cross):
        """
        Parameters
        ----------
        x : torch.Tensor
            Decoder input (positional embedding),
            shape ``(B, ts_d, out_seg_num, d_model)``.
        cross : list of torch.Tensor
            Encoder outputs at each scale, each of shape
            ``(B, ts_d, in_seg_num_i, d_model)``.

        Returns
        -------
        torch.Tensor
            Prediction of shape ``(B, out_seg_num * seg_len, ts_d)``.
        """
        batch, ts_d, out_seg_num = x.shape[0], x.shape[1], x.shape[2]
        final_predict = None
        for i, layer in enumerate(self.decode_layers):
            x, layer_predict = layer(x, cross[i])
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict

        final_predict = final_predict.reshape(batch, ts_d, out_seg_num, self.seg_len)
        final_predict = final_predict.permute(0, 2, 3, 1).reshape(
            batch, out_seg_num * self.seg_len, ts_d
        )

        return final_predict
