"""
SOFTS Model Implementation for PyTorch Forecasting v2.
-------------------------------------------------------
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers._blocks._softs_block import SoftsEncoderLayer
from pytorch_forecasting.layers._normalization import RevIN
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class Softs(TslibBaseModel):
    """
    SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion.
    """

    @classmethod
    def _pkg(cls):
        from pytorch_forecasting.models.softs._softs_pkg_v2 import Softs_pkg_v2

        return Softs_pkg_v2

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
            metadata=metadata,
        )
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self.use_revin = use_revin
        self.n_quantiles = (
            len(loss.quantiles)
            if hasattr(loss, "quantiles") and loss.quantiles is not None
            else 1
        )

        self._init_network(hidden_size, d_core, d_ff, n_layers, dropout)

    def _init_network(self, d_model, d_core, d_ff, n_layers, dropout):
        # 1. Normalization
        if self.use_revin:
            self.revin = RevIN(num_features=self.cont_dim + self.target_dim)

        # 2. Embedding Layer
        self.embedding = nn.Linear(1, d_model)  # Each variate treated independently

        # 3. Encoder Blocks
        self.encoder = nn.ModuleList(
            [
                SoftsEncoderLayer(
                    d_model=d_model, d_core=d_core, d_ff=d_ff, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        # 4. Final Projection
        self.projection = nn.Linear(
            self.context_length * d_model, self.prediction_length * self.n_quantiles
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Form Input: [Batch_Size, Context_Length, Features]
        available_features = []
        target_indices = []
        current_idx = 0
        
        if "history_cont" in x and x["history_cont"].size(-1) > 0:
            available_features.append(x["history_cont"])
            current_idx += x["history_cont"].size(-1)
            
        if "history_target" in x and x["history_target"].size(-1) > 0:
            n_targets = x["history_target"].size(-1)
            target_indices = list(range(current_idx, current_idx + n_targets))
            available_features.append(x["history_target"])

        input_data = torch.cat(available_features, dim=-1)  # [B, L, C]

        # RevIN
        if self.use_revin:
            input_data = self.revin(input_data, mode="norm")

        # Independent projection for channels: [B, C, L, d_model]
        x_enc = input_data.permute(0, 2, 1).unsqueeze(-1)  # [B, C, L, 1]
        x_enc = self.embedding(x_enc)  # [B, C, L, d_model]

        # Process through SOFTS STAD Encoder
        for layer in self.encoder:
            x_enc = layer(x_enc)

        # Output projection
        B, C, L, D = x_enc.shape
        x_enc = x_enc.reshape(B, C, -1)  # [B, C, L * d_model]
        out = self.projection(x_enc)  # [B, C, Pred_Len * quantiles]

        # Reshape for predictions
        out = out.reshape(B, C, self.prediction_length, self.n_quantiles)
        out = out.permute(0, 2, 1, 3) # -> [B, Pred_len, C, quantiles]
        
        if self.n_quantiles == 1:
            out = out.squeeze(-1)  # -> [B, Pred_len, C]

        # De-normalize
        if self.use_revin:
            if out.ndim == 4:
                # temporarily reshape to 3D for RevIN [B, Pred_len * quantiles, C]
                out = out.permute(0, 1, 3, 2).reshape(B, -1, C)
                out = self.revin(out, mode="denorm")
                out = out.reshape(B, self.prediction_length, self.n_quantiles, C).permute(0, 1, 3, 2)
            else:
                out = self.revin(out, mode="denorm")

        # Extract only the target features from output instead of passing all covariates to loss
        if target_indices:
            if out.ndim == 4:
                out = out[:, :, target_indices, :]
            else:
                out = out[:, :, target_indices]

        if "target_scale" in x and hasattr(self, "transform_output"):
            out = self.transform_output(out, x["target_scale"])

        return {"prediction": out}
