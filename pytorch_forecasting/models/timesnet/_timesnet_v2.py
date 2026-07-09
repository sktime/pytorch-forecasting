from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers._timesnet import TimesBlock
from pytorch_forecasting.metrics import MAE, Metric
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class TimesNet_v2(BaseModel):
    """TimesNet: Temporal 2D-Variation Modelling for General Time Series Analysis.

    TimesNet converts a 1-D time series into a set of 2-D representations by
    detecting dominant periodicities via FFT, then applies multi-scale 2-D
    Inception convolutions (``TimesBlock``) to capture intra- and inter-period
    variations simultaneously.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary produced by
        ``EncoderDecoderTimeSeriesDataModule.metadata``.
        Required keys:

        * ``"max_encoder_length"``  – context window length ``T``
        * ``"max_prediction_length"`` – forecast horizon ``H``
        * ``"encoder_cont"`` – number of continuous past covariates
        * ``"encoder_cat"``  – number of categorical past covariates
        * ``"target"``       – number of target variables
    cat_cardinalities : list of int, optional
        Vocabulary sizes (number of unique values + 1 for unknown) for **each**
        categorical encoder feature, in the same order as the features appear
        in the batch ``"encoder_cat"`` tensor.
        The length **must** equal ``metadata["encoder_cat"]``.
        Example: ``[7, 30]`` → 2 categorical features with 7 and 30 unique
        values respectively.
        Defaults to ``[]`` (no categorical features).
    cat_embedding_dim : int, default=16
        Embedding dimension applied to every categorical feature.
    e_layers : int, default=2
        Number of stacked ``TimesBlock`` layers.
    d_model : int, default=64
        Hidden width of the model (output dimension of the ``prepare``
        linear projection applied after the temporal extension step).
    top_k : int, default=5
        Number of dominant periodicities to detect and process inside each
        ``TimesBlock`` via FFT.
    d_ff : int, default=64
        Inner (feedforward) dimension of the ``Inception_Block_V1`` layers
        inside each ``TimesBlock``.
    num_kernels : int, default=6
        Number of parallel ``Conv2d`` kernels in each ``Inception_Block_V1``.
    loss : Metric, optional
        Loss function.  Defaults to ``MAE()``.
    logging_metrics : list of nn.Module, optional
        Additional metrics to log at every train / val / test step.
    optimizer : str or Optimizer, default="adam"
        Optimizer name (``"adam"`` or ``"sgd"``) or an instantiated
        ``torch.optim.Optimizer``.
    optimizer_params : dict, optional
        Extra keyword arguments forwarded to the optimizer constructor.
    lr_scheduler : str, optional
        Learning-rate scheduler name.
        Supported values: ``"reduce_lr_on_plateau"``, ``"step_lr"``.
    lr_scheduler_params : dict, optional
        Extra keyword arguments forwarded to the scheduler constructor.

    Notes
    -----
    * **Future covariates** (``decoder_cont`` / ``decoder_cat``) are
      intentionally **not** consumed: TimesNet is an encoder-only
      architecture that generates the full forecast from past context alone.
    * ``cat_cardinalities`` must be supplied manually because the D2 metadata
      dictionary records the *count* of categorical features but not their
      vocabulary sizes.  If your dataset has no categorical features, leave
      this at its default (empty list).

    References
    ----------
    [1] Wu, H. et al. "TimesNet: Temporal 2D-Variation Modeling for General
        Time Series Analysis." ICLR 2023. https://arxiv.org/abs/2210.02186
    [2] Original DSIPTS source:
        https://github.com/DSIP-FBK/DSIPTS/blob/main/dsipts/src/dsipts/models/TimesNet.py
    """

    @classmethod
    def _pkg(cls):
        """Return the package-layer class for this model."""
        from pytorch_forecasting.models.timesnet._timesnet_v2_pkg import (
            TimesNet_pkg_v2,
        )

        return TimesNet_pkg_v2

    def __init__(
        self,
        loss: Metric | None = None,
        cat_cardinalities: list[int] | None = None,
        cat_embedding_dim: int = 16,
        e_layers: int = 2,
        d_model: int = 64,
        top_k: int = 5,
        d_ff: int = 64,
        num_kernels: int = 6,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ) -> None:
        if loss is None:
            loss = MAE()

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

        self.past_steps: int = self.metadata["max_encoder_length"]  # T
        self.future_steps: int = self.metadata["max_prediction_length"]  # H
        self.n_cont: int = self.metadata.get("encoder_cont", 0)  # cont past covariates
        self.n_targets: int = self.metadata.get("target", 1)  # target variable(s)
        n_cat: int = self.metadata.get("encoder_cat", 0)  # categorical past features

        # Validate cat_cardinalities
        cat_cardinalities = list(cat_cardinalities) if cat_cardinalities else []
        if len(cat_cardinalities) != n_cat:
            raise ValueError(
                f"len(cat_cardinalities)={len(cat_cardinalities)} must equal "
                f"metadata['encoder_cat']={n_cat}. "
                "Provide exactly one vocabulary size per categorical feature "
                "(or leave cat_cardinalities=[] when there are no categorical "
                "features)."
            )

        # Store for _init_network and introspection
        self.cat_cardinalities: list[int] = cat_cardinalities
        self.cat_embedding_dim: int = cat_embedding_dim
        self.e_layers: int = e_layers
        self.d_model: int = d_model
        self.top_k: int = top_k
        self.d_ff: int = d_ff
        self.num_kernels: int = num_kernels

        self._init_network()

    def _init_network(self) -> None:
        """Construct and register every trainable sub-module.

        Called once from ``__init__``.  All layer sizes are derived from
        the instance attributes set earlier in ``__init__``.
        """

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, self.cat_embedding_dim)
                for cardinality in self.cat_cardinalities
            ]
        )

        # Compute combined input channel width C_in
        emb_out_dim: int = len(self.cat_cardinalities) * self.cat_embedding_dim
        past_channels: int = self.n_targets + self.n_cont  # pure numeric width
        C_in: int = past_channels + emb_out_dim  # total input width

        # Cache for use in forward()
        self._past_channels = past_channels
        self._C_in = C_in

        # Temporal extension:
        self.predict_linear = nn.Linear(
            self.past_steps,
            self.past_steps + self.future_steps,
        )

        # Feature projection
        self.prepare = nn.Linear(C_in, self.d_model)

        # Stack of TimesBlocks
        self.timesnet_blocks = nn.ModuleList(
            [
                TimesBlock(
                    seq_len=self.past_steps,
                    pred_len=self.future_steps,
                    top_k=self.top_k,
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    num_kernels=self.num_kernels,
                )
                for _ in range(self.e_layers)
            ]
        )

        # Post-block layer normalisation
        self.layer_norm = nn.LayerNorm(self.d_model)

        # Output projection
        self.projection = nn.Linear(self.d_model, self.n_targets)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of TimesNet.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Batch dictionary produced by
            ``EncoderDecoderTimeSeriesDataModule``.  The following keys are
            consumed:

            * ``"target_past"``  : ``(B, T, n_targets)`` or ``(B, T)``
              Historical target values.
            * ``"encoder_cont"`` : ``(B, T, n_cont)``
              Past continuous covariates.
            * ``"encoder_cat"``  : ``(B, T, n_cat)``  (long integer tensor)
              Past categorical covariates.  Optional if ``n_cat == 0``.

            All other batch keys (``decoder_cont``, ``decoder_cat``,
            ``encoder_mask``, etc.) are silently ignored because TimesNet is
            an encoder-only architecture.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"prediction": tensor}`` where ``tensor`` has shape
            ``(B, H, n_targets)``.
        """

        target_past = x["target_past"]  # (B, T, n_targets)
        encoder_cont = x["encoder_cont"]  # (B, T, n_cont)

        if target_past.dim() == 2:
            target_past = target_past.unsqueeze(-1)  # (B, T, 1)

        x_num = torch.cat([target_past, encoder_cont], dim=-1)

        encoder_cat = x.get("encoder_cat", None)  # (B, T, n_cat)

        if (
            len(self.cat_embeddings) > 0
            and encoder_cat is not None
            and encoder_cat.shape[-1] > 0
        ):
            emb_list = [
                self.cat_embeddings[i](encoder_cat[:, :, i].long())
                for i in range(len(self.cat_embeddings))
            ]
            x_cat_emb = torch.cat(emb_list, dim=-1)  # (B, T, emb_out_dim)
            enc_out = torch.cat([x_num, x_cat_emb], dim=-1)  # (B, T, C_in)
        else:
            enc_out = x_num  # (B, T, past_channels)

        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        enc_out = self.prepare(enc_out)  # (B, T+H, d_model)

        for block in self.timesnet_blocks:
            enc_out = self.layer_norm(block(enc_out))

        dec_out = self.projection(enc_out)  # (B, T+H, n_targets)

        out = dec_out[:, -self.future_steps :, :]  # (B, H, n_targets)

        return {"prediction": out}
