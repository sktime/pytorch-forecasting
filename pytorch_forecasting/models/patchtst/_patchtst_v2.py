"""
PatchTST model for Pytorch Forecasting.
-----------------------------------------
"""

#################################################
# NOTE: This is an experimental implementation  #
# of PatchTST model for PTF v2.                 #
# It is an unstable API and subject to change.  #
#################################################

from typing import Any, Optional, Union
import warnings

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers._attention._attention_layer import AttentionLayer
from pytorch_forecasting.layers._attention._full_attention import FullAttention
from pytorch_forecasting.layers._embeddings._patch_embedding import PatchEmbedding
from pytorch_forecasting.layers._encoders._self_attn_encoder import SelfAttnEncoder
from pytorch_forecasting.layers._encoders._self_attn_encoder_layer import (
    SelfAttnEncoderLayer,
)
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class _Transpose(nn.Module):
    """Helper module to transpose two dimensions
    (used in the norm_layer of the encoder).
    """

    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class _FlattenHead(nn.Module):
    """
    Prediction head for PatchTST.

    Flattens the patch dimension and applies a linear projection from
    (d_model * n_patches) → prediction_length. Handles quantile output
    when n_quantiles > 1.

    Parameters
    ----------
    n_vars : int
        Number of input variables (channels). Used only for shape reference.
    nf : int
        Input feature size = d_model * n_patches.
    target_window : int
        Number of future time steps (= prediction_length, or
        prediction_length * n_quantiles when quantile loss is used).
    head_dropout : float
        Dropout probability in the head. Defaults to 0.
    """

    def __init__(
        self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0
    ):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_vars, d_model, n_patches)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_vars, target_window)
        """
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTST(TslibBaseModel):
    """
    PatchTST: Patch Time Series Transformer.

    PatchTST divides each time series variable's history into fixed-size,
    potentially overlapping patches, embeds each patch as a token, and then
    processes all patch tokens through a standard Transformer encoder.
    Variables are processed independently ("Channel Independence"), which
    greatly reduces the sequence length seen by the Transformer (from T to
    T/stride) and allows the model to scale efficiently to long histories.

    Parameters
    ----------
    loss : nn.Module
        Loss function for the training step. Supports ``MAE``, ``MSE``,
        and ``QuantileLoss`` from ``pytorch_forecasting.metrics``.
    d_model : int, default=128
        Hidden dimension of the Transformer (patch embedding size).
    n_heads : int, default=16
        Number of attention heads in each Transformer encoder layer.
    e_layers : int, default=3
        Number of Transformer encoder layers.
    d_ff : int, default=256
        Inner dimension of the position-wise feed-forward network in each
        encoder layer.
    patch_len : int, default=16
        Number of time steps in each patch.
    stride : int, default=8
        Step size between consecutive patches. With ``stride < patch_len``
        patches overlap.
    dropout : float, default=0.2
        Dropout probability applied in the embedding, attention, and prediction
        head.
    activation : str, default="gelu"
        Activation function for the feed-forward network in each encoder layer.
        Must be ``"relu"`` or ``"gelu"``.
    logging_metrics : list[nn.Module] or None, default=None
        Additional metrics to log during training.
    optimizer : str or Optimizer or None, default="adam"
        Optimizer to use.
    optimizer_params : dict or None, default=None
        Extra keyword arguments forwarded to the optimizer constructor.
    lr_scheduler : str or None, default=None
        Name of the learning-rate scheduler to use.
    lr_scheduler_params : dict or None, default=None
        Extra keyword arguments forwarded to the scheduler constructor.
    metadata : dict or None, default=None
        Dataset metadata injected by ``TslibDataModule``. Contains
        ``context_length``, ``prediction_length``, and ``feature_indices``.

    References
    ----------
    [1] Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
        with Transformers", ICLR 2023.
        https://arxiv.org/pdf/2211.14730.pdf
    [2] https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py

    Notes
    -----
    This implementation supports only continuous features. Categorical variables
    will be accommodated in future versions.
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.patchtst._patchtst_pkg_v2 import PatchTST_pkg_v2

        return PatchTST_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        d_model: int = 128,
        n_heads: int = 16,
        e_layers: int = 3,
        d_ff: int = 256,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.2,
        activation: str = "gelu",
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
            "PatchTST is an experimental model implemented on TslibBaseModelV2. "
            "It is an unstable version and may be subject to unannounced changes. "
            "Please use with caution."
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.patch_len = patch_len
        self.stride = stride
        self.dropout_rate = dropout
        self.activation = activation

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """
        Initialize the PatchTST network components.

        Sets up:
        - The number of input variables (enc_in = continuous + target features)
        - The padding for patching (= stride so no data is lost)
        - The number of patches produced per variable
        - ``self.patch_embedding``: maps each patch to a d_model-dim vector
        - ``self.encoder``: a stack of Transformer encoder layers with
          BatchNorm after the final layer (from the original paper)
        - ``self.head``: a linear projection from patches → prediction horizon
        """
        self.enc_in = self.cont_dim + self.target_dim

        padding = self.stride

        # patch embedding
        self.patch_embedding = PatchEmbedding(
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            padding=padding,
            dropout=self.dropout_rate,
        )

        # transformer encoder
        self.encoder = SelfAttnEncoder(
            attn_layers=[
                SelfAttnEncoderLayer(
                    attention=AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            attention_dropout=self.dropout_rate,
                            output_attention=False,
                        ),
                        d_model=self.d_model,
                        n_heads=self.n_heads,
                    ),
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    dropout=self.dropout_rate,
                    activation=self.activation,
                )
                for _ in range(self.e_layers)
            ],
            # BatchNorm applied over the d_model dimension after the final layer
            norm_layer=nn.Sequential(
                _Transpose(1, 2),
                nn.BatchNorm1d(self.d_model),
                _Transpose(1, 2),
            ),
        )

        # prediction head
        # calculate n_patches given replication padding
        n_patches = int((self.context_length - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_model * n_patches

        # Handle quantile loss: we output Q quantiles per future step
        self.n_quantiles = None
        if isinstance(self.loss, QuantileLoss):
            self.n_quantiles = len(self.loss.quantiles)

        target_window = self.prediction_length
        if self.n_quantiles is not None:
            target_window = self.prediction_length * self.n_quantiles

        self.head = _FlattenHead(
            n_vars=self.enc_in,
            nf=self.head_nf,
            target_window=target_window,
            head_dropout=self.dropout_rate,
        )

    def _encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the PatchTST forward pass on pre-processed input data.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, context_length, enc_in)``.

        Returns
        -------
        torch.Tensor
            Predictions of shape:
            - ``(batch_size, prediction_length, target_dim)`` for point forecasts
            - ``(batch_size, prediction_length, n_quantiles)`` for quantile forecasts
        """
        # Instance normalization (Non-stationary Transformer trick)
        # Scale each variable independently
        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # patching and embedding
        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)

        # transformer encoder
        enc_out, _ = self.encoder(enc_out)

        # reshape back to (batch, enc_in, d_model, n_patches)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # → (batch, enc_in, d_model, n_patches)
        enc_out = enc_out.permute(0, 1, 3, 2)

        # prediction head
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # de-normalize
        stdev = stdev.squeeze(-1).unsqueeze(1)
        means = means.squeeze(-1).unsqueeze(1)

        dec_out = dec_out * stdev + means

        return dec_out

    def _prepare_input_data(self, x: dict[str, torch.Tensor]):
        """Prepare input data and target indices from the PTF input dictionary."""
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

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the PatchTST model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary of input tensors from ``TslibDataModule``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"prediction": tensor}`` where ``tensor`` has shape
            ``(batch_size, prediction_length, target_dim)`` or
            ``(batch_size, prediction_length, n_quantiles)``.
        """
        input_data, target_indices = self._prepare_input_data(x)

        # Run the full PatchTST forward pass
        prediction = self._encoder(input_data)

        # Select only the target variable columns (discard covariates from output)
        if target_indices is not None:
            if self.n_quantiles is not None:
                # prediction: (batch, pred_len * n_quantiles, enc_in)
                prediction = prediction[:, :, target_indices]
            else:
                prediction = prediction[:, :, target_indices]

        # Reshape quantile output:
        # (batch, pred_len * Q, target_dim) → (batch, pred_len, Q)
        if self.n_quantiles is not None:
            batch_size = prediction.shape[0]
            prediction = prediction.reshape(
                batch_size, self.prediction_length, self.n_quantiles
            )

        # Inverse normalisation built into BaseModel if target_scale exists
        if "target_scale" in x and hasattr(self, "transform_output"):
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}
