"""N-HiTS v2 model for time series forecasting."""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import MASE, Metric
from pytorch_forecasting.models.base._base_model_v2 import BaseModel
from pytorch_forecasting.models.nhits.sub_modules import NHiTS as NHiTSModule


class NHiTS_v2(BaseModel):
    """N-HiTS model for time series forecasting.

    Based on the article
    `N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting
    <https://arxiv.org/abs/2201.12886>`_.

    Parameters
    ----------
    naive_level : bool, default=True
        Whether to use a naive (last-observation) level at the start of the
        residual loop.
    shared_weights : bool, default=True
        Whether to share weights across blocks within each stack.
    activation : str, default="ReLU"
        Activation function for MLP layers.
        One of ``"ReLU"``, ``"Softplus"``, ``"Tanh"``, ``"SELU"``,
        ``"LeakyReLU"``, ``"PReLU"``, ``"Sigmoid"``.
    initialization : str, default="lecun_normal"
        Weight initialisation scheme.
        One of ``"orthogonal"``, ``"he_uniform"``, ``"he_normal"``,
        ``"glorot_uniform"``, ``"glorot_normal"``, ``"lecun_normal"``.
    n_blocks : list of int, default=[1, 1, 1]
        Number of blocks per stack.
    n_layers : int or list of int, default=2
        Number of fully-connected layers per block.
        If int, the same value is used for all stacks.
    hidden_size : int, default=512
        Width of the fully-connected layers inside each block.
    pooling_sizes : list of int, optional
        Pooling kernel sizes per stack.
        Defaults to an exponential schedule derived from ``prediction_length``.
    downsample_frequencies : list of int, optional
        Downsampling factors for the forecast interpolation per stack.
        Defaults to a heuristic based on ``pooling_sizes``.
    pooling_mode : str, default="max"
        Pooling mode, one of ``"max"`` or ``"average"``.
    interpolation_mode : str, default="linear"
        Interpolation mode for the identity basis function.
        One of ``"linear"``, ``"nearest"``, or ``"cubic-<batch_size>"``.
    batch_normalization : bool, default=False
        Whether to apply batch normalisation in the MLP layers.
    dropout : float, default=0.0
        Dropout probability applied in the MLP layers.
    backcast_loss_ratio : float, default=0.0
        Weight of the backcast loss relative to the forecast loss.
        When 0, only the forecast loss is used.
        When > 0, the total loss is
        ``(1 - backcast_loss_ratio) * forecast_loss
        + backcast_loss_ratio * backcast_loss``.
    loss : Metric, optional
        Loss to optimise. Defaults to
        :class:`~pytorch_forecasting.metrics.MASE`.
    logging_metrics : list of nn.Module, optional
        Additional metrics logged during training and validation.
    optimizer : Optimizer or str, optional
        Optimizer used for training. Default is ``"adam"``.
    optimizer_params : dict, optional
        Parameters forwarded to the optimizer constructor.
    lr_scheduler : str, optional
        Learning rate scheduler name.
    lr_scheduler_params : dict, optional
        Parameters forwarded to the LR scheduler constructor.
    metadata : dict, optional
        Dataset metadata produced by
        :class:`~pytorch_forecasting.data.data_module\
.EncoderDecoderTimeSeriesDataModule`.
        Must contain keys ``"max_encoder_length"`` and
        ``"max_prediction_length"``.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`~pytorch_forecasting.models.base._base_model_v2.BaseModel`.
    """

    @classmethod
    def _pkg(cls):
        """Return the package class for this model.

        Returns
        -------
        NHiTS_pkg_v2 : type
            Package class associated with this model.
        """
        from pytorch_forecasting.models.nhits._nhits_pkg_v2 import NHiTS_pkg_v2

        return NHiTS_pkg_v2

    def __init__(
        self,
        *,
        naive_level: bool = True,
        shared_weights: bool = True,
        activation: str = "ReLU",
        initialization: str = "lecun_normal",
        n_blocks: list[int] | None = None,
        n_layers: int | list[int] = 2,
        hidden_size: int = 512,
        pooling_sizes: list[int] | None = None,
        downsample_frequencies: list[int] | None = None,
        pooling_mode: str = "max",
        interpolation_mode: str = "linear",
        batch_normalization: bool = False,
        dropout: float = 0.0,
        backcast_loss_ratio: float = 0.0,
        loss: Metric | None = None,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs,
    ):
        if loss is None:
            loss = MASE()

        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        self.metadata = metadata

        self.context_length = metadata["max_encoder_length"]
        self.prediction_length = metadata["max_prediction_length"]
        output_size = [metadata["target"]]
        static_size = (
            metadata["static_categorical_features"]
            + metadata["static_continuous_features"]
        )
        encoder_covariate_size = metadata["encoder_cont"]
        decoder_covariate_size = metadata["decoder_cont"]

        if n_blocks is None:
            n_blocks = [1, 1, 1]
        n_stacks = len(n_blocks)

        if pooling_sizes is None:
            pooling_sizes = np.exp2(
                np.round(
                    np.linspace(0.49, np.log2(self.prediction_length / 2), n_stacks)
                )
            )
            pooling_sizes = [max(1, int(x)) for x in pooling_sizes[::-1]]

        if downsample_frequencies is None:
            downsample_frequencies = [
                max(1, min(self.prediction_length, int(np.power(x, 1.5))))
                for x in pooling_sizes
            ]

        if isinstance(n_layers, int):
            n_layers = [n_layers] * n_stacks

        self._n_stacks = n_stacks
        self._backcast_loss_ratio = backcast_loss_ratio

        # NHiTSModule expects hidden_size as a nested list of shape
        # [n_stacks, n_layers_per_block], where each value is the MLP layer width.
        # This is different from the hidden_size parameter above, which is a single int.
        hidden_size_per_stack = n_stacks * [2 * [hidden_size]]

        self.model = NHiTSModule(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            # Sizes are derived from DataModule metadata so they automatically
            # reflect the actual dataset configuration. When the v2 DataModule
            # adds covariate or static feature support, these values will update
            # without any changes required here.
            output_size=output_size,
            static_size=static_size,
            encoder_covariate_size=encoder_covariate_size,
            decoder_covariate_size=decoder_covariate_size,
            static_hidden_size=hidden_size,
            n_blocks=n_blocks,
            n_layers=n_layers,
            hidden_size=hidden_size_per_stack,
            pooling_sizes=pooling_sizes,
            downsample_frequencies=downsample_frequencies,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            dropout=dropout,
            activation=activation,
            initialization=initialization,
            batch_normalization=batch_normalization,
            shared_weights=shared_weights,
            naive_level=naive_level,
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the NHiTS model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input batch containing:

            * ``"target_past"`` : tensor of shape
              ``(batch_size, context_length, 1)``
            * ``"encoder_mask"`` : tensor of shape
              ``(batch_size, context_length, 1)`` or
              ``(batch_size, context_length)``, 1 for valid, 0 for padded
            * ``"encoder_cont"`` : tensor of shape
              ``(batch_size, context_length, n_encoder_cont)``, optional
            * ``"decoder_cont"`` : tensor of shape
              ``(batch_size, prediction_length, n_decoder_cont)``, optional
            * ``"static_categorical_features"`` : tensor of shape
              ``(batch_size, 1, n_static_cat)``, optional
            * ``"static_continuous_features"`` : tensor of shape
              ``(batch_size, 1, n_static_cont)``, optional

        Returns
        -------
        out : dict[str, torch.Tensor]
            Dictionary containing:

            * ``"prediction"`` : tensor of shape
              ``(batch_size, prediction_length, 1)``
            * ``"backcast"`` : tensor of shape
              ``(batch_size, context_length, 1)``
            * ``"block_forecasts"`` : stacked block forecast contributions
            * ``"block_backcasts"`` : stacked block backcast contributions
        """
        target = x["target_past"]  # [batch, context_length, 1]
        encoder_mask = x["encoder_mask"]
        if encoder_mask.dim() == 3:
            encoder_mask = encoder_mask.squeeze(-1)  # [batch, context_length]

        # Continuous covariates: (batch, seq_len, n_features) or None.
        # NHiTSBlock guards with ``if self.encoder/decoder_covariate_size > 0``.
        encoder_x_t = x.get("encoder_cont")  # (batch, context_length, n_enc_cont)
        decoder_x_t = x.get(
            "decoder_cont"
        )  # (batch, prediction_length, n_decoder_cont)

        # Static features: concatenate categorical and continuous into
        # x_s of shape (batch, static_size).  DataModule stores them as
        # (batch, 1, n_feats), so squeeze the middle dim before cat.
        x_s = None
        st_parts = []
        for key in ("static_categorical_features", "static_continuous_features"):
            feat = x.get(key)
            if feat is not None:
                feat = feat.squeeze(1).float()  # (batch, n_feats)
                if feat.shape[-1] > 0:
                    st_parts.append(feat)
        if st_parts:
            x_s = torch.cat(st_parts, dim=-1)  # (batch, static_size)

        forecast, backcast, block_forecasts, block_backcasts = self.model(
            encoder_y=target,
            encoder_mask=encoder_mask,
            encoder_x_t=encoder_x_t,
            decoder_x_t=decoder_x_t,
            x_s=x_s,
        )

        return {
            "prediction": forecast,
            "backcast": backcast,
            "block_forecasts": block_forecasts,
            "block_backcasts": block_backcasts,
        }

    def _compute_loss(
        self,
        x: dict[str, torch.Tensor],
        y: torch.Tensor,
        prefix: str,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the combined forecast and optional backcast loss.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input batch, as produced by the dataloader.
        y : torch.Tensor
            Ground-truth future target of shape
            ``(batch_size, prediction_length)``.
        prefix : str
            Logging prefix, either ``"train"`` or ``"val"``.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss value.
        out : dict[str, torch.Tensor]
            Raw model output from :meth:`forward`.
        """
        out = self(x)
        forecast_loss = self.loss(out["prediction"], y)

        if self._backcast_loss_ratio > 0.0:
            backcast_target = x["target_past"].squeeze(-1)  # [batch, context_length]
            backcast_loss = self.loss(out["backcast"], backcast_target)
            loss = (
                1 - self._backcast_loss_ratio
            ) * forecast_loss + self._backcast_loss_ratio * backcast_loss
        else:
            loss = forecast_loss

        self.log(
            f"{prefix}_loss",
            loss,
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log_metrics(out["prediction"], y, prefix=prefix)
        return loss, out

    def training_step(self, batch, batch_idx):
        """Training step.

        Parameters
        ----------
        batch : tuple of (dict[str, torch.Tensor], torch.Tensor)
            Batch of data from the dataloader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        output : dict[str, torch.Tensor]
            Dictionary with key ``"loss"``.
        """
        x, y = batch
        loss, _ = self._compute_loss(x, y, "train")
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Parameters
        ----------
        batch : tuple of (dict[str, torch.Tensor], torch.Tensor)
            Batch of data from the dataloader.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        output : dict[str, torch.Tensor]
            Dictionary with key ``"val_loss"``.
        """
        x, y = batch
        loss, _ = self._compute_loss(x, y, "val")
        return {"val_loss": loss}
