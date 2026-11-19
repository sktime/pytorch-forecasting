import math

import torch
import torch.nn as nn

from pytorch_forecasting.layers import AttentionLayer, FullAttention


class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super().__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape
        seg_num = ts_len // self.seg_len

        x_segment = x.reshape(batch, seg_num, self.seg_len, ts_dim)
        x_segment = x_segment.permute(0, 3, 1, 2)
        x_segment = x_segment.reshape(-1, self.seg_len)

        x_embed = self.linear(x_segment)
        x_embed = x_embed.reshape(batch, ts_dim, seg_num, -1)

        return x_embed


class TwoStageAttentionLayer(nn.Module):
    """
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
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


class SegMerging(nn.Module):
    """
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment
    to get representation of a coarser scale
    we set win_size = 2 in our paper
    """

    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
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


class scale_block(nn.Module):
    """
    We can use one segment merging layer followed by multiple TSA layers in each scale
    the parameter `depth' determines the number of TSA layers used in each scale
    We set depth = 1 in the paper
    """

    def __init__(
        self, win_size, d_model, n_heads, d_ff, depth, dropout, seg_num=10, factor=10
    ):
        super().__init__()

        if win_size > 1:
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None

        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(
                TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, d_ff, dropout)
            )

    def forward(self, x):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x)

        return x


class Encoder(nn.Module):
    """
    The Encoder of Crossformer.
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
            scale_block(
                1, d_model, n_heads, d_ff, block_depth, dropout, in_seg_num, factor
            )
        )
        for i in range(1, e_blocks):
            self.encode_blocks.append(
                scale_block(
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
        encode_x = []
        encode_x.append(x)

        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)

        return encode_x


class DecoderLayer(nn.Module):
    """
    The decoder layer of Crossformer, each layer will make a prediction at its scale.

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
        self.self_attention = TwoStageAttentionLayer(
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


class Decoder(nn.Module):
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
                DecoderLayer(
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
