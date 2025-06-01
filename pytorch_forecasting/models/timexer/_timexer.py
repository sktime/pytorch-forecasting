"""
Time Series Transformer with eXogenous variables (TimeXer)
---------------------------------------------------------
"""

#######################################################
# Note: This is an example version to demonstrate the
# working of the TimeXer model with the exisiting v2
# designs. The pending work includes building the D2
# layer and base tslib model.
######################################################

from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.base._base_model_v2 import BaseModel
from pytorch_forecasting.models.timexer.sub_modules import (
    AttentionLayer,
    DataEmbedding_inverted,
    Encoder,
    EncoderLayer,
    EnEmbedding,
    FlattenHead,
    FullAttention,
)


class TimeXer(BaseModel):
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        loss: nn.Module,
        logging_metrics: Optional[list[nn.Module]] = None,
        optimizer: Optional[Union[Optimizer, str]] = "adam",
        optimizer_params: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[dict] = None,
        task_name: str = "long_term_forecast",
        features: str = "MS",
        enc_in: int = None,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable] = "torch.nn.functional.relu",
        patch_length: int = 24,
        use_norm: bool = False,
        factor: int = 5,
        embed_type: str = "fixed",
        freq: str = "h",
        metadata: Optional[dict] = None,
        target_positions: torch.LongTensor = None,
    ):
        """An implementation of the TimeXer model.
        TimeXer empowers the canonical transformer with the ability to reconcile
        endogenous and exogenous information without any architectural modifications
        and achieves consistent state-of-the-art performance across twelve real-world
        forecasting benchmarks.
        TimeXer employs patch-level and variate-level representations respectively for
        endogenous and exogenous variables, with an endogenous global token as a bridge
        in-between. With this design, TimeXer can jointly capture intra-endogenous
        temporal dependencies and exogenous-to-endogenous correlations.
        TimeXer model for time series forecasting with exogenous variables.
        """
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params or {},
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params or {},
        )

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.task_name = task_name
        self.features = features
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.activation = activation
        self.patch_length = patch_length
        self.use_norm = use_norm
        self.factor = factor
        self.embed_type = embed_type
        self.freq = freq
        self.metadata = metadata
        self.n_target_vars = self.metadata["target"]
        self.target_positions = target_positions
        self.enc_in = self.metadata["encoder_cont"]
        self.patch_num = self.context_length // self.patch_length
        self.dropout = dropout

        self.n_quantiles = None

        if isinstance(loss, QuantileLoss):
            self.n_quantiles = len(loss.quantiles)

        self.en_embedding = EnEmbedding(
            self.n_target_vars,
            self.d_model,
            self.patch_length,
            self.dropout,
        )

        self.ex_embedding = DataEmbedding_inverted(
            self.context_length,
            self.d_model,
            self.embed_type,
            self.freq,
            self.dropout,
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        self.d_model,
                        self.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        self.d_model,
                        self.n_heads,
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
        )
        self.head_nf = self.d_model * (self.patch_num + 1)
        self.head = FlattenHead(
            self.enc_in,
            self.head_nf,
            self.prediction_length,
            head_dropout=self.dropout,
            n_quantiles=self.n_quantiles,
        )

    def _forecast(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forecast for univariate or multivariate with single target (MS) case.
        Args:
            x: Dictionary containing entries for encoder_cat, encoder_cont
        """
        batch_size = x["encoder_cont"].shape[0]
        encoder_cont = x["encoder_cont"]
        encoder_time_idx = x.get("encoder_time_idx", None)
        past_target = x.get(
            "target",
            torch.zeros(batch_size, self.prediction_length, 0, device=self.device),
        )

        if encoder_time_idx is not None and encoder_time_idx.dim() == 2:
            # change [batch_size, time_steps] to [batch_size, time_steps, features]
            encoder_time_idx = encoder_time_idx.unsqueeze(-1)

        en_embed, n_vars = self.en_embedding(past_target.permute(0, 2, 1))
        ex_embed = self.ex_embedding(encoder_cont, encoder_time_idx)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )

        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        if self.n_quantiles is not None:
            dec_out = dec_out.permute(0, 2, 1, 3)
        else:
            dec_out = dec_out.permute(0, 2, 1)

        return dec_out

    def _forecast_multi(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forecast for multivariate with multiple targets (M) case.
        Args:
            x: Dictionary containing entries for encoder_cat, encoder_cont
        Returns:
            Dictionary with predictions
        """

        batch_size = x["encoder_cont"].shape[0]
        encoder_cont = x.get(
            "encoder_cont",
            torch.zeros(batch_size, self.prediction_length, device=self.device),
        )
        encoder_time_idx = x.get("encoder_time_idx", None)
        encoder_targets = x.get(
            "target",
            torch.zeros(batch_size, self.prediction_length, device=self.device),
        )
        en_embed, n_vars = self.en_embedding(encoder_targets.permute(0, 2, 1))
        ex_embed = self.ex_embedding(encoder_cont, encoder_time_idx)

        # batch_size x sequence_length x d_model
        enc_out = self.encoder(en_embed, ex_embed)

        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )  # batch_size x n_vars x sequence_length x d_model

        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        if self.n_quantiles is not None:
            dec_out = dec_out.permute(0, 2, 1, 3)
        else:
            dec_out = dec_out.permute(0, 2, 1)

        return dec_out

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        Args:
            x: Dictionary containing model inputs
        Returns:
            Dictionary with model outputs
        """
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):  # noqa: E501
            if self.features == "M":
                out = self._forecast_multi(x)
            else:
                out = self._forecast(x)
            prediction = out[:, : self.prediction_length, :]

            # note: prediction.size(2) is the number of target variables i.e n_targets
            target_indices = range(prediction.size(2))

            if self.n_quantiles is not None:
                prediction = [prediction[..., i, :] for i in target_indices]
            else:

                if len(target_indices) == 1:
                    prediction = prediction[..., 0]
                else:
                    prediction = [prediction[..., i] for i in target_indices]
            return {"prediction": prediction}
        else:
            return None
