"""
SOFTS Model Implementation for PyTorch Forecasting v2.
-------------------------------------------------------
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers._encoders._softs_encoder import SOFTSEncoderLayer
from pytorch_forecasting.layers._normalization import RevIN
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class SOFTS(BaseModel):
    """
    SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion.

    GitHub Link: https://github.com/Secilia-Cxy/SOFTS/

    Research Paper: https://arxiv.org/abs/2404.14197

    Parameters
    ----------
    hidden_size: int
        Embedding size of individual time series channel, default = 512
    d_core: int
        Hidden dimension of the central core node, default = 512
    d_ff: int
        Dimension of the feed-forward network, default = 2048
    n_layers: int
        Number of encoder layers, default = 2
    dropout: float
        Dropout rate, default = 0.1
    use_revin: bool
        Whether to use RevIN, default = True
    optimizer: Optimizer | str
        Optimizer to use for training, default = "adam"
    optimizer_params: dict | None
        Parameters for the optimizer, default = None
    lr_scheduler: str | None
        Learning rate scheduler to use, default = None
    lr_scheduler_params: dict | None
        Parameters for the learning rate scheduler, default = None
    """

    @classmethod
    def _pkg(cls):
        from pytorch_forecasting.models.softs._softs_pkg_v2 import SOFTS_pkg_v2

        return SOFTS_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        hidden_size: int = 512,
        d_core: int = 512,
        d_ff: int = 2048,
        n_layers: int = 2,
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
        self.context_length = self.metadata.get("max_encoder_length", 0)
        self.prediction_length = self.metadata.get("max_prediction_length", 0)

        self.cont_dim = self.metadata.get("encoder_cont", 0)
        self.target_dim = self.metadata.get("target", 1)

        self.use_revin = use_revin
        self.n_quantiles = (
            len(loss.quantiles)
            if hasattr(loss, "quantiles") and loss.quantiles is not None
            else 1
        )

        self._init_network(hidden_size, d_core, d_ff, n_layers, dropout)

    def _init_network(self, d_model, d_core, d_ff, n_layers, dropout):
        # Normalization
        if self.use_revin:
            self.revin = RevIN(num_features=self.cont_dim + self.target_dim)

        # Embedding Layer
        self.embedding = nn.Linear(1, d_model)

        # Encoder Blocks
        self.encoder = nn.ModuleList(
            [
                SOFTSEncoderLayer(
                    d_model=d_model, d_core=d_core, d_ff=d_ff, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        # Final Projection
        self.projection = nn.Linear(
            self.context_length * d_model, self.prediction_length * self.n_quantiles
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Form Input: [Batch_Size, Context_Length, Features]
        available_features = []
        target_indices = []
        current_idx = 0

        if "encoder_cont" in x and x["encoder_cont"].size(-1) > 0:
            available_features.append(x["encoder_cont"])
            current_idx += x["encoder_cont"].size(-1)

        if "target_past" in x and x["target_past"].size(-1) > 0:
            target_data = x["target_past"]
            if target_data.ndim == 2:
                target_data = target_data.unsqueeze(-1)
            n_targets = target_data.size(-1)
            target_indices = list(range(current_idx, current_idx + n_targets))
            available_features.append(target_data)

        input_data = torch.cat(available_features, dim=-1)

        # RevIN
        if self.use_revin:
            input_data = self.revin(input_data, mode="norm")

        # Independent projection for channels: [B, C, L, d_model]
        x_enc = input_data.permute(0, 2, 1).unsqueeze(-1)
        x_enc = self.embedding(x_enc)

        # Process through SOFTS STAD Encoder
        for layer in self.encoder:
            x_enc = layer(x_enc)

        # Output projection
        B, C, L, D = x_enc.shape
        x_enc = x_enc.reshape(B, C, -1)
        out = self.projection(x_enc)

        # Reshape for predictions
        out = out.reshape(B, C, self.prediction_length, self.n_quantiles)
        out = out.permute(0, 2, 1, 3)

        if self.n_quantiles == 1:
            out = out.squeeze(-1)

        # De-normalize
        if self.use_revin:
            if out.ndim == 4:
                # temporarily reshape to 3D for RevIN [B, Pred_len * quantiles, C]
                out = out.permute(0, 1, 3, 2).reshape(B, -1, C)
                out = self.revin(out, mode="denorm")
                out = out.reshape(
                    B, self.prediction_length, self.n_quantiles, C
                ).permute(0, 1, 3, 2)
            else:
                out = self.revin(out, mode="denorm")

        # Extract only the target features from output instead of
        # passing all covariates to loss
        if target_indices:
            if out.ndim == 4:
                out = out[:, :, target_indices, :]
            else:
                out = out[:, :, target_indices]

        if "target_scale" in x and hasattr(self, "transform_output"):
            out = self.transform_output(out, x["target_scale"])

        return {"prediction": out}

    def transform_output(
        self,
        y_hat: torch.Tensor | list[torch.Tensor],
        target_scale: torch.Tensor | dict[str, torch.Tensor] | None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Transform the output of the model back to the original scale.

        Support:
        - EncoderDecoderTimeSeriesDataModule: target_scale is a scalar tensor
        """
        if target_scale is None:
            return y_hat

        # EncoderDecoderTimeSeriesDataModule provides a plain tensor
        if isinstance(target_scale, torch.Tensor):
            scale = target_scale
            while scale.dim() < y_hat.dim():
                scale = scale.unsqueeze(-1)
            return y_hat * scale
