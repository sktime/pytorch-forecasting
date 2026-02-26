"""
Reformer: Efficient Transformer for Long-Range Sequence Modeling
----------------------------------------------------------------
"""

from typing import Any, Optional, Union
import warnings as warn

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class Reformer(TslibBaseModel):
    """
    An implementation of the Reformer model for pytorch-forecasting-v2.

    Parameters
    ----------
    loss : nn.Module
        Loss function to use for training.
    enc_in : int, optional
        Number of input features for the encoder. If not provided, it will be set
        to the number of continuous features in the dataset.
    d_model : int, default=512
        Dimension of the model embeddings and hidden representations.
    n_heads : int, default=8
        Number of attention heads in the LSH multi-head attention mechanism.
    e_layers : int, default=2
        Number of encoder layers in the Reformer architecture.
    d_ff : int, default=2048
        Dimension of the feed-forward network inside each encoder layer.
    dropout : float, default=0.1
        Dropout rate applied throughout the model for regularization.
    activation : str, default='gelu'
        Activation function used in the feed-forward network. Common choices
        are ``'relu'`` and ``'gelu'``.
    embed : str, default='timeF'
        Type of time feature embedding to use. Use ``'timeF'`` for time-frequency
        embeddings or ``'fixed'`` / ``'learned'`` for positional embeddings.
    task_name : str, default='long_term_forecast'
        Forecasting task type. Either ``'long_term_forecast'`` or
        ``'short_term_forecast'``. Short-term forecasting applies instance
        normalization before encoding and denormalizes the output.
    bucket_size : int, default=4
        Size of the LSH attention buckets. Queries and keys are hashed into
        buckets of this size before computing attention within each bucket.
    n_hashes : int, default=4
        Number of LSH hashing rounds. More rounds improve attention approximation
        quality at the cost of additional computation.
    logging_metrics : list[nn.Module] or None, default=None
        List of additional metrics to log during training, validation, and testing.
    optimizer : Optimizer or str or None, default='adam'
        Optimizer to use for training. Can be a string name (e.g., ``'adam'``,
        ``'sgd'``) or an instantiated :class:`torch.optim.Optimizer`.
    optimizer_params : dict or None, default=None
        Keyword arguments passed to the optimizer constructor. If ``None``,
        the optimizer's default parameters are used.
    lr_scheduler : str or None, default=None
        Learning rate scheduler to apply after each epoch. If ``None``, no
        scheduler is used.
    lr_scheduler_params : dict or None, default=None
        Keyword arguments passed to the learning rate scheduler constructor.
        If ``None``, the scheduler's default parameters are used.
    metadata : dict or None, default=None
        Metadata dictionary produced by ``TslibDataModule``. Must contain at
        minimum a ``'freq'`` key describing the time series frequency (e.g.,
        ``'h'`` for hourly). Used to configure time-feature embeddings and to
        infer dataset-level properties such as the number of continuous features.

    References
    ----------
    [1] Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020). Reformer: The Efficient
        Transformer. https://arxiv.org/abs/2001.04451
    [2] https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py

    Notes
    -----
    This implementation handles only continuous variables.
    """

    @classmethod
    def _pkg(cls):
        """
        Return the package class that contains the Reformer model implementation.

        Returns
        -------
        Reformer_pkg_v2
            The package class used to instantiate the underlying Reformer network.
        """
        from pytorch_forecasting.models.reformer.reformer_pkg_v2 import Reformer_pkg_v2

        return Reformer_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        enc_in: int = None,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        embed: str = "timeF",
        task_name: str = "long_term_forecast",
        bucket_size: int = 4,
        n_hashes: int = 4,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the Reformer model.

        Calls the parent ``TslibBaseModel.__init__`` to handle shared setup
        (loss, optimizer, scheduler, metadata), then stores all Reformer-specific
        hyperparameters and calls ``_init_network`` to build the model layers.

        Parameters
        ----------
        loss : nn.Module
            Loss function to use for training.
        enc_in : int, optional
            Number of encoder input features. Defaults to ``cont_dim`` when
            ``None``.
        d_model : int, default=512
            Hidden dimension of all model layers.
        n_heads : int, default=8
            Number of LSH attention heads.
        e_layers : int, default=2
            Number of stacked encoder layers.
        d_ff : int, default=2048
            Inner dimension of the position-wise feed-forward sublayers.
        dropout : float, default=0.1
            Dropout probability.
        activation : str, default='gelu'
            Non-linearity for the feed-forward sublayers.
        embed : str, default='timeF'
            Time-feature embedding strategy.
        task_name : str, default='long_term_forecast'
            Controls whether short- or long-term forecasting logic is used.
        bucket_size : int, default=4
            LSH bucket size for approximating full attention.
        n_hashes : int, default=4
            Number of LSH hashing rounds.
        logging_metrics : list[nn.Module] or None, default=None
            Extra metrics logged during training and evaluation.
        optimizer : Optimizer or str or None, default='adam'
            Training optimizer.
        optimizer_params : dict or None, default=None
            Optimizer constructor arguments.
        lr_scheduler : str or None, default=None
            Learning rate scheduler name.
        lr_scheduler_params : dict or None, default=None
            Scheduler constructor arguments.
        metadata : dict or None, default=None
            Dataset metadata from ``TslibDataModule``. Must contain ``'freq'``.
        **kwargs : Any
            Additional keyword arguments forwarded to ``TslibBaseModel``.
        """
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            metadata=metadata,
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.embed = embed
        self.task_name = task_name
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self.enc_in = enc_in
        self.freq = metadata["freq"]
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """
        Build and register all sub-modules of the Reformer architecture.

        Constructs the following components:

        - ``enc_embedding`` : :class:`DataEmbedding` that projects raw input
          features (plus optional time marks) into the ``d_model``-dimensional
          hidden space.
        - ``encoder`` : :class:`ReformerEncoder` consisting of ``e_layers``
          stacked :class:`ReformerEncoderLayer` blocks, each containing LSH
          self-attention (:class:`ReformerLayer`) and a position-wise
          feed-forward sublayer, followed by layer normalisation.
        - ``projection`` : :class:`torch.nn.Linear` that maps encoder outputs
          from ``d_model`` dimensions to ``target_dim`` output channels.

        """
        from pytorch_forecasting.layers import (
            DataEmbedding,
            ReformerEncoder,
            ReformerEncoderLayer,
            ReformerLayer,
        )

        self.enc_in = self.enc_in or self.cont_dim
        self.enc_embedding = DataEmbedding(
            self.enc_in,
            self.d_model,
            self.embed,
            self.freq,
            self.dropout,
        )

        self.encoder = ReformerEncoder(
            [
                ReformerEncoderLayer(
                    ReformerLayer(
                        None,
                        self.d_model,
                        self.n_heads,
                        bucket_size=self.bucket_size,
                        n_hashes=self.n_hashes,
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

        self.projection = nn.Linear(self.d_model, self.target_dim, bias=True)

    def _long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Perform long-term forecasting via encoder-only pass with a placeholder.

        Parameters
        ----------
        x_enc : torch.Tensor
            Historical encoder input of shape ``(batch, context_length, enc_in)``.
        x_mark_enc : torch.Tensor or None
            Time-feature marks for the encoder sequence, shape
            ``(batch, context_length, time_features)``, or ``None`` when
            ``embed != 'timeF'``.
        x_dec : torch.Tensor
            Decoder input (used only to extract the future placeholder tokens),
            shape ``(batch, label_length + prediction_length, enc_in)``.
        x_mark_dec : torch.Tensor or None
            Time-feature marks for the decoder sequence, or ``None``.

        Returns
        -------
        torch.Tensor
            Full encoder output of shape
            ``(batch, context_length + prediction_length, target_dim)``.
            Slice ``[:, -prediction_length:, :]`` to obtain the forecast.
        """
        x_enc = torch.cat([x_enc, x_dec[:, -self.prediction_length :, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat(
                [x_mark_enc, x_mark_dec[:, -self.prediction_length :, :]], dim=1
            )

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)

        return dec_out

    def _short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Perform short-term forecasting with instance normalization.

        Parameters
        ----------
        x_enc : torch.Tensor
            Historical encoder input of shape ``(batch, context_length, enc_in)``.
        x_mark_enc : torch.Tensor or None
            Time-feature marks for the encoder sequence, or ``None``.
        x_dec : torch.Tensor
            Decoder input for extracting future placeholder tokens, shape
            ``(batch, label_length + prediction_length, enc_in)``.
        x_mark_dec : torch.Tensor or None
            Time-feature marks for the decoder sequence, or ``None``.

        Returns
        -------
        torch.Tensor
            Denormalized forecast of shape
            ``(batch, context_length + prediction_length, target_dim)``.
            Slice ``[:, -prediction_length:, :]`` to obtain the forecast.
        """
        mean_enc = x_enc[:, :, 0:1].mean(1, keepdim=True).detach()
        std_enc = torch.sqrt(
            torch.var(x_enc[:, :, 0:1], dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        all_mean = x_enc.mean(1, keepdim=True).detach()
        all_std = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = (x_enc - all_mean) / all_std

        x_enc = torch.cat([x_enc, x_dec[:, -self.prediction_length :, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat(
                [x_mark_enc, x_mark_dec[:, -self.prediction_length :, :]], dim=1
            )

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def _forecast(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Route the input batch to the appropriate forecasting method.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Batch dictionary expected to contain the following keys:

            - ``'history_cont'`` : ``(batch, context_length, enc_in)`` – historical
              continuous covariates used as encoder input.
            - ``'future_cont'`` : ``(batch, prediction_length, future_features)`` –
              future continuous covariates used as decoder placeholder input.
            - ``'history_time_idx'`` *(optional)* : ``(batch, context_length)`` –
              integer or float time indices for encoder time-feature embedding.
            - ``'future_time_idx'`` *(optional)* : ``(batch, prediction_length)`` –
              integer or float time indices for decoder time-feature embedding.

        Returns
        -------
        torch.Tensor
            Raw model output of shape
            ``(batch, context_length + prediction_length, target_dim)``.
        """
        x_enc = x["history_cont"]
        x_dec = x["future_cont"]
        if x_enc.shape[-1] != x_dec.shape[-1]:
            diff = x_enc.shape[-1] - x_dec.shape[-1]
            if diff > 0:
                x_dec = torch.nn.functional.pad(x_dec, (0, diff))
            else:
                x_dec = x_dec[..., : x_enc.shape[-1]]

        if self.embed == "timeF":
            x_mark_enc = x["history_time_idx"].unsqueeze(-1).float()
            x_mark_dec = x["future_time_idx"].unsqueeze(-1).float()
        else:
            x_mark_enc = None
            x_mark_dec = None

        if self.task_name == "short_term_forecast":
            dec_out = self._short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out

        dec_out = self._long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass for Reformer model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Batch dictionary. See :meth:`_forecast` for the expected keys.
            Additionally, if the key ``'target_scale'`` is present its value
            is forwarded to :meth:`transform_output` for inverse scaling.

        Returns
        -------
        dict[str, torch.Tensor]

            - ``'prediction'`` : ``(batch, prediction_length, target_dim)`` –
              the model's forecast, optionally rescaled to the original target
              units.
        """
        out = self._forecast(x)
        prediction = out[:, : self.prediction_length, :]

        if "target_scale" in x:
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}
