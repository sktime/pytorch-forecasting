"""
SOFTS Blocks for Star Aggregate-Dispatch Network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class STADModule(nn.Module):
    """
    Star Aggregate-Dispatch (STAD) Module for capturing inter-series dependencies.
    Aggregates the channel features to a central node, processes them, and dispatches back.
    """
    def __init__(self, d_model: int, d_core: int, dropout: float = 0.0):
        super(STADModule, self).__init__()
        # MLP for channel-wise processing
        self.channel_mixing = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Generates weights for aggregation
        self.gen_weight = nn.Linear(d_model, d_core)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor of shape (batch_size, n_channels, seq_len, d_model)
        
        Returns
        -------
        output: torch.Tensor of shape (batch_size, n_channels, seq_len, d_model)
        """
        B, C, L, D = x.shape
        
        # Aggregation Phase: Calculate weights and aggregate to star nodes
        # w: [B, C, L, d_core] -> [B, C, d_core] (pooling over length for simplicity, or sum)
        w = self.gen_weight(x).mean(dim=2)  # Alternative: max pooling
        w = torch.softmax(w, dim=1) # [B, C, d_core]
        
        # x_pooled: [B, C, D]
        x_pooled = x.mean(dim=2)
        
        # Star Node Representation: [B, d_core, D]
        core_node = torch.einsum('bcd,bce->bed', x_pooled, w)
        
        # Process Star Node
        core_node = self.channel_mixing(core_node)
        
        # Dispatch Phase: Dispatch features back to each channel
        dispatch_out = torch.einsum('bed,bce->bcd', core_node, w)
        
        # Add dispatched info back to original sequence (broadcasting over sequence length)
        dispatch_out = dispatch_out.unsqueeze(2).repeat(1, 1, L, 1)
        
        return x + dispatch_out

class SoftsEncoderLayer(nn.Module):
    """
    Single Encoder layer for SOFTS.
    """
    def __init__(self, d_model: int, d_core: int, d_ff: int, dropout: float = 0.0):
        super(SoftsEncoderLayer, self).__init__()
        self.stad = STADModule(d_model=d_model, d_core=d_core, dropout=dropout)
        
        # Feed Forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L, D]
        """
        x = x + self.dropout(self.stad(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
