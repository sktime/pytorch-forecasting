import torch
import torch.nn as nn


class IEBlock(nn.Module):
    """
    Information Exchange block used by LightTS.

    Applies spatial projection, channel mixing, and an output projection
    to exchange information across time-series channels.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_nodes: int
    ) -> None:
        """
        Initialize the IEBlock.

        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        hidden_dim : int
            Hidden projection size.
        output_dim : int
            Output feature dimension.
        num_nodes : int
            Number of channels mixed by the block.
        """
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")

        reduced_dim = max(1, hidden_dim // 4)

        self.spatial_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, reduced_dim),
        )

        self.channel_proj = nn.Linear(num_nodes, num_nodes)
        nn.init.eye_(self.channel_proj.weight)
        if self.channel_proj.bias is not None:
            nn.init.zeros_(self.channel_proj.bias)

        self.output_proj = nn.Linear(reduced_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the IEBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, input_dim, num_nodes)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch_size, output_dim, num_nodes)``.
        """

        x = self.spatial_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1) + self.channel_proj(x.permute(0, 2, 1))
        x = self.output_proj(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)
