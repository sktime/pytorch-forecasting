"""
TSMixer block for TSMixer v2 implementation.
"""

import torch
import torch.nn as nn


class TSMixerBlock(nn.Module):
    """
    TSMixer block for applying the time-mixing and feature-mixing MLPs.

    Parameters
    ----------
    sequence_length : int
        Length of the lookback window containing past time steps.
    num_features : int
        Number of expected features in the input.
    hidden_dim : int
        Dimension of the hidden layers.
    dropout : float
        Probability of an element to be zeroed.
    """

    def __init__(
        self,
        sequence_length: int,
        num_features: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Linear(sequence_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sequence_length),
            nn.Dropout(dropout),
        )

        self.channel = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time-mixing and feature-mixing to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor with time-mixing and feature-mixing applied.
        """
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x
