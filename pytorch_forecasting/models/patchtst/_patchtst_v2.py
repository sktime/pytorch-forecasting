"""
PatchTST model for Pytorch Forecasting.
-----------------------------------------
"""

#################################################
# NOTE: This is an experimental implementation  #
# of PatchTST model for PTF v2.                 #
# It is an unstable API and subject to change.  #
#################################################

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers import (
    AttentionLayer,
    FlattenHead,
    FullAttention,
    Transpose,
)
from pytorch_forecasting.layers._embeddings._patch_embedding import PatchEmbedding
from pytorch_forecasting.layers._encoders._self_attn_encoder import SelfAttnEncoder
from pytorch_forecasting.layers._encoders._self_attn_encoder_layer import (
    SelfAttnEncoderLayer,
)
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class PatchTST(TslibBaseModel):
    """PatchTST: Patch Time Series Transformer.

    PatchTST divides each time series variable's history into fixed-size,
    potentially overlapping patches, embeds each patch as a token, and then
    processes all patch tokens through a standard Transformer encoder.
    Variables are processed independently ("Channel Independence"), which
    greatly reduces the sequence length seen by the Transformer and allows
    the model to scale efficiently to long histories.

    Parameters
    ----------
    loss : nn.Module
        Loss function for training. Supports ``MAE``, ``MSE``,
        and ``QuantileLoss`` from ``pytorch_forecasting.metrics``.
    d_model : int, default=128
        Hidden dimension of the Transformer (patch embedding size).
    n_heads : int, default=16
        Number of attention heads in each encoder layer.
    e_layers : int, default=3
        Number of Transformer encoder layers.
    d_ff : int, default=256
        Dimension of the feed-forward network in each encoder layer.
    patch_len : int, default=16
        Number of time steps in each patch.
    stride : int, default=8
        Step size between consecutive patches. ``stride < patch_len``
        produces overlapping patches.
    dropout : float, default=0.2
        Dropout probability used throughout the model.
    activation : str, default="gelu"
        Activation function for the encoder feed-forward layers.
    logging_metrics : list[nn.Module] or None, default=None
        Additional metrics to log during training, validation, and testing.
    optimizer : str or Optimizer or None, default="adam"
        Optimizer to use for training.
    optimizer_params : dict or None, default=None
        Parameters for the optimizer.
    lr_scheduler : str or None, default=None
        Learning rate scheduler to use.
    lr_scheduler_params : dict or None, default=None
        Parameters for the learning rate scheduler.
    metadata : dict or None, default=None
        Metadata for the model from TslibDataModule.

    References
    ----------
    [1] Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
        with Transformers", ICLR 2023. https://arxiv.org/pdf/2211.14730.pdf
    [2] https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py

    Notes
    -----
    [1] This implementation supports only continuous features. Categorical
        variables will be accommodated in future versions.
    [2] The ``PatchTST`` model obtains many of its attributes from the
        ``TslibBaseModel`` class, which handles metadata parsing and
        model initialization boilerplate.
    """  # noqa: E501

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.patchtst._patchtst_pkg_v2 import (
            PatchTST_pkg_v2,
        )

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
        import warnings

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
        self.dropout = dropout
        self.activation = activation

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """Initialise the PatchTST architecture network components."""

        self.enc_in = self.cont_dim + self.target_dim
        padding = self.stride

        self.patch_embedding = PatchEmbedding(
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            padding=padding,
            dropout=self.dropout,
        )

        self.encoder = SelfAttnEncoder(
            attn_layers=[
                SelfAttnEncoderLayer(
                    attention=AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        d_model=self.d_model,
                        n_heads=self.n_heads,
                    ),
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(self.d_model),
                Transpose(1, 2),
            ),
        )

        n_patches = int((self.context_length - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_model * n_patches

        self.n_quantiles = None
        if isinstance(self.loss, QuantileLoss):
            self.n_quantiles = len(self.loss.quantiles)

        self.head = FlattenHead(
            n_vars=self.target_dim,
            nf=self.head_nf,
            target_window=self.prediction_length,
            head_dropout=self.dropout,
            n_quantiles=self.n_quantiles,
        )

    def _encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input time series through patching and self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, context_length, enc_in)``.

        Returns
        -------
        torch.Tensor
            Predictions of shape
            ``(batch_size, prediction_length, target_dim)`` or
            ``(batch_size, prediction_length, n_quantiles)``.
        """
        # instance normalization (RevIN-style, per variable)
        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # patching + embedding
        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)

        # transformer encoder
        enc_out, _ = self.encoder(enc_out)

        # reshape: (batch * n_vars, n_patches, d_model)
        #       -> (batch, n_vars, d_model, n_patches)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        enc_out = enc_out.permute(0, 1, 3, 2)

        # extract target channels before prediction head
        enc_target = enc_out[:, -self.target_dim :, :, :]

        dec_out = self.head(enc_target)

        # de-normalize (only for point forecasts; quantiles are scale-free)
        if self.n_quantiles is None:
            target_means = means[:, :, -self.target_dim :]
            target_stdev = stdev[:, :, -self.target_dim :]
            dec_out = dec_out * target_stdev + target_means

        return dec_out

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

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the PatchTST model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary containing input tensors.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing output tensors. These can include
            - prediction: output of shape
              ``(batch_size, prediction_length, target_dim)``
        """  # noqa: E501
        input_data, _ = self._prepare_input_data(x)

        prediction = self._encoder(input_data)

        if "target_scale" in x and hasattr(self, "transform_output"):
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}
