"""Auto-Correlation mechanism for Autoformer."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoCorrelation(nn.Module):
    """Auto-Correlation mechanism that replaces standard self-attention."""

    def __init__(self, d_model, n_heads=8, dropout=0.1, top_k=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.top_k = top_k
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def time_delay_agg_inference(self, values, corr):
        batch, length, d_k = values.shape
        if self.top_k is None:
            top_k = max(1, int(math.log(length)))
        else:
            top_k = min(self.top_k, length)
        weights, delays = torch.topk(corr, top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        output = torch.zeros_like(values)
        for k in range(top_k):
            delay_k = delays[:, k]
            weight_k = weights[:, k].unsqueeze(-1).unsqueeze(-1)
            rolled_values = []
            for b in range(batch):
                delay_val = int(delay_k[b].item())
                rolled = torch.roll(values[b], shifts=delay_val, dims=0)
                rolled_values.append(rolled)
            rolled_values = torch.stack(rolled_values, dim=0)
            output += weight_k * rolled_values
        return output

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, self.d_k)
        keys = self.key_projection(keys).view(B, S, H, self.d_k)
        values = self.value_projection(values).view(B, S, H, self.d_k)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        outputs = []
        for h in range(H):
            q = queries[:, h, :, :]
            k = keys[:, h, :, :]
            v = values[:, h, :, :]
            q_mean = q.mean(dim=-1)
            k_mean = k.mean(dim=-1)
            q_fft = torch.fft.rfft(q_mean, dim=1)
            k_fft = torch.fft.rfft(k_mean, dim=1)
            res = q_fft * torch.conj(k_fft)
            corr = torch.fft.irfft(res, n=L, dim=1)
            out = self.time_delay_agg_inference(v, corr)
            outputs.append(out)
        output = torch.cat(outputs, dim=-1)
        output = self.out_projection(output)
        output = self.dropout(output)
        return output, None