import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoCorrelation(nn.Module):
    """Auto-Correlation attention mechanism."""

    def __init__(self, d_model, n_heads=8, dropout=0.1, top_k=None):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.top_k = top_k

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def time_delay_agg(self, values, corr):
        B, L, D = values.shape
        k = self.top_k or max(1, int(math.log(L)))
        k = min(k, L)

        weights, delays = torch.topk(corr, k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        idx = torch.arange(L, device=values.device)
        idx = (idx[None, None, :] - delays[:, :, None]) % L

        values = values.unsqueeze(1).expand(-1, k, -1, -1)
        gathered = torch.gather(values, 2, idx[..., None].expand(-1, -1, -1, D))

        return (weights[..., None, None] * gathered).sum(dim=1)

    def forward(self, queries, keys, values, attn_mask=None):
        if attn_mask is not None:
            raise NotImplementedError

        B, L, _ = queries.shape
        H = self.n_heads

        q = self.q_proj(queries).view(B, L, H, self.d_k)
        k = self.k_proj(keys).view(B, L, H, self.d_k)
        v = self.v_proj(values).view(B, L, H, self.d_k)

        outputs = []
        for h in range(H):
            qh = q[:, :, h]
            kh = k[:, :, h]
            vh = v[:, :, h]

            corr = torch.fft.irfft(
                torch.fft.rfft(qh, dim=1) * torch.conj(torch.fft.rfft(kh, dim=1)),
                n=L,
                dim=1,
            ).mean(dim=-1)

            out = self.time_delay_agg(vh, corr)
            outputs.append(out)

        output = torch.cat(outputs, dim=-1)
        return self.dropout(self.out_proj(output)), None
