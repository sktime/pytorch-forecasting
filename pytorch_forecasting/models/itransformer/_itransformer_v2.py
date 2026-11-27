"""
iTransformer model for PyTorch Forecasting.
-------------------------------------------
"""

#################################################
# NOTE: This is an experimental implementation  #
# of iTransformer model for PTF v2.             #
# It is an unstable API and subject to change.  #
#################################################

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers import (
    AttentionLayer,
    DataEmbedding_inverted,
    FullAttention,
    iTransformerEncoder,
    iTransformerEncoderLayer,
)
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class iTransformer(TslibBaseModel):
    """
    iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.

    iTransformer embeds each variate's entire lookback sequence as a single token,
    applies self-attention across variates to capture multivariate correlations,
    and applies feed-forward networks per variate to learn temporal representations.

    Parameters
    ----------
    loss : nn.Module
        Loss function for training step.
    hidden_size : int, default=512
        Dimension of the model embeddings (``d_model`` in the paper).
    n_heads : int, default=8
        Number of attention heads in multi-head self-attention.
    e_layers : int, default=2
        Number of Transformer encoder layers.
    d_ff : int, default=2048
        Dimension of the feed-forward network in each encoder layer.
    dropout : float, default=0.1
        Dropout rate applied to embeddings, attention, and encoder layers.
    factor : int, default=1
        Attention factor passed to ``FullAttention``.
    activation : str, default="gelu"
        Activation function for the encoder feed-forward network
        (``"relu"`` or ``"gelu"``).
    use_norm : bool, default=True
        Whether to apply non-stationary instance normalization and de-normalization.
    use_efficient_attention : bool, default=False
        Whether to use PyTorch's native Scaled Dot Product Attention (SDPA).
    logging_metrics : list[nn.Module] | None, default=None
        List of metrics to log during training, validation, and testing.
    optimizer : Optimizer | str | None, default="adam"
        Optimizer to use for training.
    optimizer_params : dict | None, default=None
        Parameters for the optimizer.
    lr_scheduler : str | None, default=None
        Learning rate scheduler to use.
    lr_scheduler_params : dict | None, default=None
        Parameters for the learning rate scheduler.
    metadata : dict | None, default=None
        Metadata for the model from ``TslibDataModule``.

    References
    ----------
    [1] https://arxiv.org/abs/2310.06625
    [2] https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py

    Notes
    -----
    [1] This implementation supports only continuous features. Categorical variables
        will be accommodated in future versions.
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.itransformer import iTransformerForecaster

        return iTransformerForecaster

    def __init__(
        self,
        loss: nn.Module,
        hidden_size: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        factor: int = 1,
        activation: str = "gelu",
        use_norm: bool = True,
        use_efficient_attention: bool = False,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ):
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})."
            )

        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            metadata=metadata,
        )

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.factor = factor
        self.activation = activation
        self.use_norm = use_norm
        self.use_efficient_attention = use_efficient_attention

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """Initialise the iTransformer network layer components."""
        self.n_quantiles = None
        if isinstance(self._loss, QuantileLoss):
            self.n_quantiles = len(self._loss.quantiles)

        output_dim = self.prediction_length
        if self.n_quantiles is not None:
            output_dim = self.prediction_length * self.n_quantiles

        self.enc_embedding = DataEmbedding_inverted(
            c_in=self.context_length,
            d_model=self.hidden_size,
            dropout=self.dropout,
        )

        self.encoder = iTransformerEncoder(
            [
                iTransformerEncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                            use_efficient_attention=self.use_efficient_attention,
                        ),
                        self.hidden_size,
                        self.n_heads,
                    ),
                    self.hidden_size,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.hidden_size),
        )

        self.projection = nn.Linear(self.hidden_size, output_dim, bias=True)

    def _prepare_input_data(self, x: dict[str, torch.Tensor]):
        """Prepare input data and target indices for model input."""
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

        if not available_features:
            raise ValueError("No valid input features found in the input dictionary.")

        input_data = torch.cat(available_features, dim=-1)

        target_indices = (
            torch.tensor(target_indices, dtype=torch.long, device=input_data.device)
            if target_indices
            else None
        )

        return input_data, target_indices

    def _reshape_output(self, output: torch.Tensor) -> torch.Tensor:
        """Reshape projected output for point or quantile predictions."""
        if self.n_quantiles is not None:
            batch_size, n_targets, _ = output.shape
            output = output.reshape(
                batch_size, n_targets, self.prediction_length, self.n_quantiles
            ).permute(0, 2, 1, 3)
            if n_targets == 1:
                output = output.squeeze(2)
        else:
            output = output.permute(0, 2, 1)  # (batch, pred_len, target_dim)

        return output

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the iTransformer model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary containing input tensors from ``TslibDataModule``.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing ``"prediction"`` tensor.
        """
        x_enc, target_indices = self._prepare_input_data(x)
        _, _, n_vars = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # (B, N, output_dim)
        dec_out = self.projection(enc_out)[:, :n_vars, :]

        if target_indices is not None:
            dec_out = dec_out[:, target_indices, :]

        if self.use_norm:
            target_stdev = (
                stdev[:, 0, target_indices]
                if target_indices is not None
                else stdev[:, 0, :]
            ).unsqueeze(-1)
            target_means = (
                means[:, 0, target_indices]
                if target_indices is not None
                else means[:, 0, :]
            ).unsqueeze(-1)
            dec_out = dec_out * target_stdev + target_means

        prediction = self._reshape_output(dec_out)

        if "target_scale" in x and hasattr(self, "transform_output"):
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}
