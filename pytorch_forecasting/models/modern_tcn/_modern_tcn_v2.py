"""
ModernTCN Model for PyTorch Forecasting v2.
--------------------------------------------
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers import RevIN
from pytorch_forecasting.layers._blocks._modern_tcn_block import ModernTCNBlock
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class ModernTCN(BaseModel):
    """
    ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis.

    GitHub Repository:https://github.com/luodhhh/ModernTCN

    Research Paper: https://openreview.net/forum?id=vpJMJerXHU

    Parameters
    ----------
    loss : nn.Module
        Loss function for training.
    d_model : int
        Embedding dimension per patch.
    kernel_size : int
        Kernel size for the depthwise convolution.
    n_blocks : int
        Number of ModernTCN blocks.
    d_ff : int
        Hidden dimension in the pointwise conv FFN.
    patch_size : int
        Number of time steps per patch.
    dropout : float
        Dropout rate.
    use_revin : bool
        Whether to use RevIN normalization.
    """

    @classmethod
    def _pkg(cls):
        """Return the package class for this model."""
        from pytorch_forecasting.models.modern_tcn._modern_tcn_pkg_v2 import (
            ModernTCN_pkg_v2,
        )

        return ModernTCN_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        d_model: int = 64,
        kernel_size: int = 51,
        n_blocks: int = 2,
        d_ff: int = 256,
        patch_size: int = 8,
        dropout: float = 0.1,
        use_revin: bool = True,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        self.metadata = metadata or {}

        self.n_quantiles = 1
        if hasattr(loss, "quantiles") and loss.quantiles is not None:
            self.n_quantiles = len(loss.quantiles)

        self.context_length = self.metadata.get("max_encoder_length", 0)
        self.prediction_length = self.metadata.get("max_prediction_length", 0)
        self.n_cont_features = self.metadata.get("encoder_cont", 0)
        self.target_dim = self.metadata.get("target", 1)

        self.n_channels = self.n_cont_features + self.target_dim
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_patches = self.context_length // patch_size
        self.use_revin = use_revin

        if self.use_revin:
            self.revin = RevIN(num_features=self.n_channels)

        self.patch_embed = nn.Linear(patch_size, d_model)

        self.blocks = nn.ModuleList(
            [
                ModernTCNBlock(d_model, kernel_size, d_ff, dropout)
                for _ in range(n_blocks)
            ]
        )

        self.head = nn.Linear(
            self.n_patches * d_model,
            self.prediction_length * self.n_quantiles,
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of ModernTCN."""
        encoder_cont = x["encoder_cont"]
        target = x["target_past"]
        if target.ndim == 2:
            target = target.unsqueeze(-1)

        input_data = torch.cat([encoder_cont, target], dim=-1)

        if self.use_revin:
            input_data = self.revin(input_data, mode="norm")

        B, L, C = input_data.shape

        x_enc = input_data.permute(0, 2, 1)
        x_enc = x_enc.reshape(B, C, self.n_patches, self.patch_size)
        x_enc = self.patch_embed(x_enc)
        x_enc = x_enc.permute(0, 1, 3, 2)
        x_enc = x_enc.reshape(B * C, self.d_model, self.n_patches)

        for block in self.blocks:
            x_enc = block(x_enc)

        x_enc = x_enc.reshape(B, C, self.d_model, self.n_patches)
        x_enc = x_enc.reshape(B, C, -1)

        out = self.head(x_enc)
        out = out.reshape(B, C, self.prediction_length, self.n_quantiles)
        out = out.permute(0, 2, 1, 3)

        target_indices = list(range(self.n_cont_features, C))
        out = out[:, :, target_indices, :]

        if len(target_indices) == 1:
            out = out.squeeze(2)

        return {"prediction": out}
