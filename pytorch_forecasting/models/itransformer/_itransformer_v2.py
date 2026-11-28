"""
iTransformer: Inverted Transformer for Multivariate Time Series Forecasting.
-----------------------------------------------------------------------------
"""

#################################################
# NOTE: This is an experimental implementation  #
# of iTransformer for PTF v2.                   #
# It is an unstable API and subject to change.  #
#################################################

from typing import Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from pytorch_forecasting.layers import (
    AttentionLayer,
    DataEmbedding_inverted,
    FullAttention,
)
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel

# ---------------------------------------------------------------------------
# Private self-attention-only encoder (the existing Encoder/EncoderLayer in
# layers/ are TimeXer cross-attention layers and cannot be reused here).
# ---------------------------------------------------------------------------


class _ITransformerEncoderLayer(nn.Module):
    """Single iTransformer encoder layer: pure self-attention + FFN.

    Mirrors TSLib's ``Transformer_EncDec.EncoderLayer`` with one attention
    block (no cross-attention, no global token).
    """

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class _ITransformerEncoder(nn.Module):
    """Stack of _ITransformerEncoderLayer with optional layer-norm."""

    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# iTransformer model
# ---------------------------------------------------------------------------


class ITransformer(TslibBaseModel):
    """iTransformer: Inverted Transformer for Multivariate Time Series Forecasting.

    iTransformer inverts the roles of the time and variate dimensions: each
    *variate*'s full look-back series is treated as a single token, projected
    with a linear layer into a d-dimensional embedding, and then passed through
    a standard Transformer encoder that captures *inter-variate* correlations.
    A final linear layer maps each variate embedding to the prediction horizon.

    Parameters
    ----------
    loss : nn.Module
        Loss function for training step.
    d_model : int, default=512
        Dimension of the token embeddings (and Transformer hidden size).
    n_heads : int, default=8
        Number of attention heads in the multi-head self-attention blocks.
    e_layers : int, default=3
        Number of Transformer encoder layers.
    d_ff : int, default=512
        Feed-forward hidden dimension inside each encoder layer.
    dropout : float, default=0.1
        Dropout rate used throughout the model.
    activation : str, default='gelu'
        Activation function inside the feed-forward sub-layer.
        One of ``'relu'`` or ``'gelu'``.
    factor : int, default=1
        Attention factor passed to ``FullAttention`` (used when
        ``use_efficient_attention=False``).
    use_norm : bool, default=True
        Whether to apply instance normalisation (subtract mean, divide by
        std) before the encoder and undo it after the projection, following
        the Non-Stationary Transformer convention used in the original paper.
        Set to ``False`` if the datamodule already normalises the targets
        (e.g. via ``TorchNormalizer``) to avoid double-normalising.
    logging_metrics : list[nn.Module] | None, default=None
        List of metrics to log during training, validation, and testing.
    optimizer : Optimizer | str | None, default='adam'
        Optimizer to use for training.
    optimizer_params : dict | None, default=None
        Parameters for the optimizer.
    lr_scheduler : str | None, default=None
        Learning rate scheduler.
    lr_scheduler_params : dict | None, default=None
        Parameters for the learning rate scheduler.
    metadata : dict | None, default=None
        Metadata dict injected by ``TslibDataModule``; provides
        ``context_length``, ``prediction_length``, ``target_dim``, etc.

    References
    ----------
    .. [1] Liu, Y. et al. (2024). "iTransformer: Inverted Transformers Are
       Effective for Time Series Forecasting." ICLR 2024.
       https://arxiv.org/abs/2310.06625
    .. [2] https://github.com/thuml/Time-Series-Library/blob/main/models/iTransformer.py

    Notes
    -----
    **Continuous features only.** Categorical variables are not supported by
    this implementation; they will be ignored if present in the datamodule.

    **No time-mark tokens.** The original paper uses timestamp features
    (``x_mark_enc``).  This implementation does *not* include them: the
    ``DataEmbedding_inverted`` layer is called with ``x_mark=None``, so the
    variate token is built solely from the look-back values.

    **Normalization.** When ``use_norm=True`` the model applies instance
    normalisation internally (per the paper) *before* passing data to the
    encoder and reverses it *after* the projection.  When the ``TslibDataModule``
    is configured with a ``TorchNormalizer`` or similar target scaler, the
    targets fed into the model are already in normalised space; in that case
    set ``use_norm=False`` (or leave the datamodule scaler as identity) to avoid
    double normalisation.  Regardless of ``use_norm``, ``transform_output`` is
    called on the final prediction to undo the datamodule's ``target_scale``.
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.itransformer._itransformer_forecaster_v2 import (  # noqa: E501
            ITransformerForecaster,
        )

        return ITransformerForecaster

    def __init__(
        self,
        loss: nn.Module,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        factor: int = 1,
        use_norm: bool = True,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
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

        warnings.warn(
            "ITransformer is an experimental model implemented on TslibBaseModel (v2). "
            "It is an unstable version and may be subject to unannounced changes. "
            "Please use with caution.",
            UserWarning,
            stacklevel=2,
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.factor = factor
        self.use_norm = use_norm

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """Initialise the iTransformer network layers."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads}) for multi-head attention to work properly."
            )

        # Total input variates: continuous covariates + targets
        self.enc_in = self.cont_dim + self.target_dim

        self.n_quantiles = None
        if isinstance(self._loss, QuantileLoss):
            self.n_quantiles = len(self._loss.quantiles)

        # Inverted embedding: maps (batch, variate, context_length) →
        # (batch, variate, d_model) treating each variate as a token.
        # DataEmbedding_inverted(c_in=context_length, d_model=d_model, dropout)
        self.enc_embedding = DataEmbedding_inverted(
            self.context_length, self.d_model, self.dropout
        )

        # Self-attention-only encoder (pure variate-level attention)
        self.encoder = _ITransformerEncoder(
            [
                _ITransformerEncoderLayer(
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
                for _ in range(self.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
        )

        # Output projection: (batch, variate, d_model) →
        # (batch, variate, prediction_length) or (batch, variate, pred*n_q)
        output_dim = (
            self.prediction_length * self.n_quantiles
            if self.n_quantiles is not None
            else self.prediction_length
        )
        self.projection = nn.Linear(self.d_model, output_dim, bias=True)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _prepare_input_data(self, x: dict[str, torch.Tensor]):
        """Concatenate history_cont and history_target; return target indices."""
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

        input_data = torch.cat(available_features, dim=-1)  # (B, T, N)
        target_indices = (
            torch.tensor(target_indices, dtype=torch.long, device=input_data.device)
            if target_indices
            else None
        )
        return input_data, target_indices

    def _reshape_output(self, output: torch.Tensor) -> torch.Tensor:
        """Reshape projection output to (B, pred_len, n_targets [, n_quantiles])."""
        if self.n_quantiles is not None:
            # output: (B, n_targets, pred_len * n_quantiles)
            B, N, _ = output.shape
            output = output.reshape(B, N, self.prediction_length, self.n_quantiles)
            # → (B, pred_len, n_targets * n_quantiles) to match QuantileLoss
            # convention used by sibling models (pred_len, n_quantiles) for
            # single-target; for multi-target we follow DLinear which squeezes
            # target channel and returns (B, pred_len, n_quantiles).
            # Keep (B, pred_len, n_quantiles) by selecting target channels:
            # target selection is already done before this call, so N == n_targets.
            # We need (B, pred_len, n_quantiles) for a single target, or
            # (B, pred_len, n_targets*n_quantiles) for multi-target - but the
            # framework matches loss shape; return (B, pred_len, n_quantiles)
            # for the single-target common case by collapsing along N:
            output = output.permute(0, 2, 1, 3)  # (B, pred_len, N, n_quantiles)
            B, P, N, Q = output.shape
            output = output.reshape(B, P, N * Q)
        else:
            # output: (B, n_targets, pred_len)
            output = output.permute(0, 2, 1)  # (B, pred_len, n_targets)
        return output

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the iTransformer model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Batch dictionary from TslibDataModule, expected to contain at
            least ``history_target`` (B, T, n_targets) and optionally
            ``history_cont`` (B, T, n_cont) and ``target_scale``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"prediction": tensor}`` where ``tensor`` has shape
            ``(B, prediction_length, n_targets)`` for point losses, or
            ``(B, prediction_length, n_targets * n_quantiles)`` for
            ``QuantileLoss``.
        """
        input_data, target_indices = self._prepare_input_data(x)
        # input_data: (B, T, N)  where N = enc_in
        if input_data.shape[1] != self.context_length:
            raise ValueError(
                f"Input sequence length ({input_data.shape[1]}) does not match "
                f"configured context_length ({self.context_length})."
            )

        if self.use_norm:
            # Instance normalisation along the time axis (per variate)
            means = input_data.mean(dim=1, keepdim=True).detach()
            input_data = input_data - means
            stdev = torch.sqrt(
                torch.var(input_data, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            input_data = input_data / stdev

        # DataEmbedding_inverted: (B, T, N) → permute → (B, N, T) →
        # value_embedding (Linear T→d_model) → (B, N, d_model)
        enc_out = self.enc_embedding(input_data, x_mark=None)
        # enc_out: (B, N, d_model)

        enc_out = self.encoder(enc_out)
        # enc_out: (B, N, d_model)

        # Keep only target variates before projection
        if target_indices is not None:
            enc_out = enc_out[:, target_indices, :]
        # enc_out: (B, n_targets, d_model)

        dec_out = self.projection(enc_out)
        # dec_out: (B, n_targets, pred_len [* n_quantiles])

        if self.use_norm:
            # Reverse instance normalisation (only for target channels)
            if target_indices is not None:
                # stdev/means: (B, 1, N) → select target cols → (B, 1, n_targets)
                stdev_t = stdev[:, :, target_indices]  # (B, 1, n_targets)
                means_t = means[:, :, target_indices]
            else:
                stdev_t = stdev
                means_t = means

            if self.n_quantiles is None:
                # dec_out: (B, n_targets, pred_len)
                # stdev_t: (B, 1, n_targets) → (B, n_targets, 1)
                stdev_t = stdev_t.permute(0, 2, 1)
                means_t = means_t.permute(0, 2, 1)
                dec_out = dec_out * stdev_t + means_t
            else:
                # dec_out: (B, n_targets, pred_len * n_quantiles)
                # Apply per-variate scale/shift to the whole prediction block
                stdev_t = stdev_t.permute(0, 2, 1)
                means_t = means_t.permute(0, 2, 1)
                dec_out = dec_out * stdev_t + means_t

        prediction = self._reshape_output(dec_out)
        # prediction: (B, pred_len, n_targets) or (B, pred_len, n_targets*n_q)

        if "target_scale" in x and hasattr(self, "transform_output"):
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}
