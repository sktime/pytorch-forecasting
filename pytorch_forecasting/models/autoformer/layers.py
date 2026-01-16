"""
Autoformer encoder and decoder layers.

These layers combine the reusable AutoCorrelation and AutoformerDecomposition
components to build the complete Autoformer architecture.
"""

import torch.nn as nn

from pytorch_forecasting.layers._attention import AutoCorrelation
from pytorch_forecasting.layers._decomposition import AutoformerDecomposition


class EncoderLayer(nn.Module):
    """
    Single Autoformer encoder layer with auto-correlation and progressive decomposition.
    
    Architecture:
        Input -> Auto-Correlation -> Decomposition -> FFN -> Decomposition -> Output
        
    The layer progressively decomposes the time series at two points,
    extracting and accumulating trend information while refining the seasonal component.
    """

    def __init__(self, d_model, n_heads=8, d_ff=None, dropout=0.1,
                 moving_avg_kernel=25, top_k=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.attention = AutoCorrelation(d_model, n_heads, dropout, top_k)
        self.decomp1 = AutoformerDecomposition(moving_avg_kernel)
        self.decomp2 = AutoformerDecomposition(moving_avg_kernel)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Forward pass of encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor [batch, seq_len, d_model]
        
        Returns:
            tuple:
                - x (torch.Tensor): Seasonal component [batch, seq_len, d_model]
                - trend (torch.Tensor): Accumulated trend [batch, seq_len, d_model]
        """
        new_x, _ = self.attention(x, x, x)
        x = x + self.dropout(new_x)

        x, trend1 = self.decomp1(x)

        y = x.transpose(-1, 1)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.transpose(-1, 1)

        x, trend2 = self.decomp2(x + y)

        trend = trend1 + trend2

        return x, trend


class Encoder(nn.Module):
    """
    Autoformer encoder with multiple layers.
    
    Stacks multiple EncoderLayer modules to build deep representations
    while progressively extracting trend information.
    """

    def __init__(self, n_layers, d_model, n_heads=8, d_ff=None,
                 dropout=0.1, moving_avg_kernel=25, top_k=None):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, moving_avg_kernel, top_k)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Forward pass through all encoder layers.
        
        Args:
            x (torch.Tensor): Input tensor [batch, seq_len, d_model]
        
        Returns:
            tuple:
                - x (torch.Tensor): Encoded seasonal component [batch, seq_len, d_model]
                - trends (list): List of trend tensors from each layer
        """
        trends = []
        for layer in self.layers:
            x, trend = layer(x)
            trends.append(trend)

        x = self.norm(x)
        return x, trends


class DecoderLayer(nn.Module):
    """
    Single Autoformer decoder layer with self and cross auto-correlation.
    
    Architecture:
        Input -> Self Auto-Correlation -> Decomposition ->
        Cross Auto-Correlation -> Decomposition ->
        FFN -> Decomposition -> Output
        
    Three decomposition points for progressive refinement.
    """

    def __init__(self, d_model, n_heads=8, d_ff=None, dropout=0.1,
                 moving_avg_kernel=25, top_k=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.self_attention = AutoCorrelation(d_model, n_heads, dropout, top_k)
        self.cross_attention = AutoCorrelation(d_model, n_heads, dropout, top_k)
        self.decomp1 = AutoformerDecomposition(moving_avg_kernel)
        self.decomp2 = AutoformerDecomposition(moving_avg_kernel)
        self.decomp3 = AutoformerDecomposition(moving_avg_kernel)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, cross):
        """
        Forward pass of decoder layer.
        
        Args:
            x (torch.Tensor): Decoder input [batch, seq_len, d_model]
            cross (torch.Tensor): Encoder output [batch, enc_len, d_model]
        
        Returns:
            tuple:
                - x (torch.Tensor): Seasonal component [batch, seq_len, d_model]
                - trend (torch.Tensor): Accumulated trend [batch, seq_len, d_model]
        """
        new_x, _ = self.self_attention(x, x, x)
        x = x + self.dropout(new_x)
        x, trend1 = self.decomp1(x)

        new_x, _ = self.cross_attention(x, cross, cross)
        x = x + self.dropout(new_x)
        x, trend2 = self.decomp2(x)

        y = x.transpose(-1, 1)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.transpose(-1, 1)

        x, trend3 = self.decomp3(x + y)

        trend = trend1 + trend2 + trend3

        return x, trend


class Decoder(nn.Module):
    """
    Autoformer decoder with multiple layers.
    
    Stacks multiple DecoderLayer modules with cross-attention to encoder output.
    """

    def __init__(self, n_layers, d_model, n_heads=8, d_ff=None,
                 dropout=0.1, moving_avg_kernel=25, top_k=None):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, moving_avg_kernel, top_k)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, cross):
        """
        Forward pass through all decoder layers.
        
        Args:
            x (torch.Tensor): Decoder input [batch, seq_len, d_model]
            cross (torch.Tensor): Encoder output [batch, enc_len, d_model]
        
        Returns:
            tuple:
                - x (torch.Tensor): Decoded seasonal component [batch, seq_len, d_model]
                - trends (list): List of trend tensors from each layer
        """
        trends = []
        for layer in self.layers:
            x, trend = layer(x, cross)
            trends.append(trend)

        x = self.norm(x)
        return x, trends