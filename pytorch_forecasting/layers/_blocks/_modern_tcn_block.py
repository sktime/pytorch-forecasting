"""
ModernTCN block: DWConv + ConvFFN + residual.
"""

import torch.nn as nn


class ModernTCNBlock(nn.Module):
    def __init__(self, d_model, kernel_size, d_ff, dropout):
        super().__init__()
        self.dwconv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.pw_conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.pw_conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.act(self.norm(self.dwconv(x)))
        out = self.drop(self.act(self.pw_conv1(out)))
        out = self.drop(self.pw_conv2(out))
        return out + residual
