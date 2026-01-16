from typing import Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers import RevIN
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class Autoformer(BaseModel):
    @classmethod
    def _pkg(cls):
        from pytorch_forecasting.models.autoformer._autoformer_v2_pkg import Autoformer_pkg_v2
        return Autoformer_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        d_model: int = 64,
        enc_layers: int = 2,
        dec_layers: int = 1,
        label_len: int = 48,
        pred_len: int = 96,
        moving_avg: int = 25,
        top_k: Optional[int] = None,
        out_channels: Optional[Union[int, list[int]]] = 1,
        use_revin: bool = False,
        persistence_weight: float = 0.0,
        logging_metrics: Optional[list[nn.Module]] = None,
        optimizer: Optional[Union[Optimizer, str]] = "adam",
        optimizer_params: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
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

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "optimizer"])

        self.metadata = metadata or {}
        self.n_quantiles = 1
        if hasattr(loss, "quantiles") and loss.quantiles is not None:
            self.n_quantiles = len(loss.quantiles)

        self.max_encoder_length = self.metadata.get("max_encoder_length", None)
        self.max_prediction_length = self.metadata.get("max_prediction_length", None)
        self.encoder_cont = self.metadata.get("encoder_cont", 0)

        self.encoder_input_dim = self.encoder_cont + 1

        self.d_model = d_model
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.label_len = label_len
        self.pred_len = pred_len
        self.moving_avg = moving_avg
        self.top_k = top_k
        self.use_revin = use_revin
        self.persistence_weight = persistence_weight

        if out_channels != 1:
            raise ValueError("out_channels must be 1 for v2 Autoformer")
        self.out_channels = out_channels

        if self.use_revin:
            self.revin = RevIN(num_features=self.encoder_input_dim)

        # 🔽 UPDATED IMPORT — shared layers (NO autoformer/layers folder)
        from pytorch_forecasting.layers.autoformer import Encoder, Decoder

        self.input_proj = nn.Linear(self.encoder_input_dim, self.d_model)

        self.enc_embedding = nn.Linear(self.d_model, self.d_model)
        self.dec_embedding = nn.Linear(self.d_model, self.d_model)

        self.encoder = Encoder(
            self.enc_layers,
            self.d_model,
            moving_avg_kernel=self.moving_avg,
            top_k=self.top_k,
        )
        self.decoder = Decoder(
            self.dec_layers,
            self.d_model,
            moving_avg_kernel=self.moving_avg,
            top_k=self.top_k,
        )

        self.projection = nn.Linear(self.d_model, self.out_channels)
        self.trend_proj = nn.Linear(self.d_model, self.out_channels)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

    def _prepare_decoder_input(self, enc_out):
        B, L_enc, C = enc_out.shape
        start = max(0, L_enc - self.label_len)
        dec_input = enc_out[:, start:L_enc, :]
        zeros = torch.zeros(
            B,
            self.pred_len,
            C,
            device=enc_out.device,
            dtype=enc_out.dtype,
        )
        return torch.cat([dec_input, zeros], dim=1)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoder_cont = x["encoder_cont"]
        target = x["target_past"]

        input_tensor = torch.cat((encoder_cont, target), dim=-1)

        if self.use_revin:
            x_norm = self.revin(input_tensor, mode="norm").transpose(1, 2)
        else:
            x_norm = input_tensor

        x_proj = self.input_proj(x_norm)
        enc_in = self.enc_embedding(x_proj)

        enc_out, enc_trends = self.encoder(enc_in)

        dec_in = self._prepare_decoder_input(enc_out)
        dec_in = self.dec_embedding(dec_in)

        dec_out, _ = self.decoder(dec_in, enc_out)
        pred_seasonal = dec_out[:, -self.pred_len :, :]

        if len(enc_trends) > 0:
            enc_trend_final = enc_trends[-1]
            trend_mean = enc_trend_final.mean(dim=1, keepdim=True)
            trend_pred = trend_mean.repeat(1, self.pred_len, 1)
        else:
            B = pred_seasonal.size(0)
            trend_pred = torch.zeros(
                B,
                self.pred_len,
                self.d_model,
                device=pred_seasonal.device,
            )

        trend_pred = self.trend_proj(trend_pred)
        seasonal_out = self.projection(pred_seasonal)

        out = seasonal_out + trend_pred

        if self.n_quantiles > 1:
            out = out.unsqueeze(-1).expand(-1, -1, -1, self.n_quantiles)
        else:
            out = out.unsqueeze(-1)

        return {"prediction": out}
