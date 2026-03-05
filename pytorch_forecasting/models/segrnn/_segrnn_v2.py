"""
Segment Recurrent Neural Network (SegRNN)
-----------------------------------------
"""

from typing import Any
import warnings as warn

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class SegRNN(TslibBaseModel):
    """
    An implementation of the SegRNN model for v2 of pytorch-forecasting.

    SegRNN replaces the traditional token-by-token iteration in RNN-based
    forecasters with segment-by-segment iterations. Rather than processing
    one time step per RNN step, the input sequence is divided into
    non-overlapping segments of length ``seg_len``; each segment is linearly
    projected to a ``d_model``-dimensional embedding and processed as a
    single token. The decoder similarly predicts ``seg_num_y`` segments at
    once, conditioned on learnable positional and channel embeddings that are
    concatenated and fed as the initial hidden state.

    This design drastically reduces the number of RNN recurrences
    (from ``seq_len`` to ``seq_len // seg_len``) and allows the model to
    scale to long prediction horizons without the vanishing-gradient issues
    associated with long unrolled RNNs.

    Parameters
    ----------
    loss : nn.Module
        Loss function to use for training.
    enc_in : int, optional
        Number of input features (channels). If not provided, inferred from
        the number of continuous features in the dataset (``cont_dim``).
    d_model : int, default=512
        Hidden dimension of the GRU and all embedding layers.
    dropout : float, default=0.1
        Dropout rate applied in the prediction head.
    seg_len : int, default=24
        Length of each non-overlapping segment used to partition the input
        sequence. Must evenly divide both ``context_length`` and
        ``prediction_length``; a warning is issued (and integer division is
        applied) when this condition is not met.
    logging_metrics : list[nn.Module] or None, default=None
        List of metrics to log during training, validation, and testing.
    optimizer : Optimizer or str or None, default='adam'
        Optimizer to use for training.
    optimizer_params : dict or None, default=None
        Parameters for the optimizer.
    lr_scheduler : str or None, default=None
        Learning rate scheduler name.
    lr_scheduler_params : dict or None, default=None
        Parameters for the learning rate scheduler.
    metadata : dict or None, default=None
        Metadata dictionary produced by ``TslibDataModule``. Used to infer
        ``context_length``, ``prediction_length``, and ``cont_dim``.

    References
    ----------
    [1] Lin, S. et al. (2023). SegRNN: Segment Recurrent Neural Network for
        Long-Term Time Series Forecasting.
        https://arxiv.org/abs/2308.11200
    [2] https://github.com/thuml/Time-Series-Library/blob/main/models/SegRNN.py

    Notes
    -----
    [1] Only continuous variables are supported. 
    [2] The model is channel-independent at the segment level: each channel
        is processed independently by reshaping ``(B, C, seg_num_x, seg_len)``
        to ``(B*C, seg_num_x, d_model)`` before the GRU forward pass.
    """

    @classmethod
    def _pkg(cls):
        """Package containing the SegRNN model."""
        from pytorch_forecasting.models.segrnn._segrnn_pkg_v2 import SegRNN_pkg_v2

        return SegRNN_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        enc_in: int = None,
        d_model: int = 512,
        dropout: float = 0.1,
        seg_len: int = 1,
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


        self.enc_in = enc_in
        self.d_model = d_model
        self.dropout = dropout
        self.seg_len = seg_len

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """
        Build and register all sub-modules of the SegRNN architecture.

        Constructs the following components:

        - ``valueEmbedding`` : two-layer MLP (``Linear → ReLU``) that maps
          each raw segment of length ``seg_len`` to a ``d_model``-dimensional
          token.
        - ``rnn`` : single-layer unidirectional GRU that encodes the
          sequence of embedded segments and generates future segments via
          teacher-forcing-free recurrence.
        - ``pos_emb`` : learnable positional embedding of shape
          ``(seg_num_y, d_model // 2)`` identifying each future segment slot.
        - ``channel_emb`` : learnable channel embedding of shape
          ``(enc_in, d_model // 2)`` identifying each input channel.
        - ``predict`` : ``Dropout → Linear`` head that maps each
          ``d_model``-dimensional GRU output back to a segment of length
          ``seg_len``.

        Raises
        ------
        ValueError
            If ``seg_len`` is larger than ``context_length`` or
            ``prediction_length``.
        """
        self.enc_in = self.enc_in or self.cont_dim

        if self.seg_len > self.context_length:
            raise ValueError(
                f"seg_len ({self.seg_len}) must be <= context_length "
                f"({self.context_length})."
            )
        if self.seg_len > self.prediction_length:
            raise ValueError(
                f"seg_len ({self.seg_len}) must be <= prediction_length "
                f"({self.prediction_length})."
            )

        if self.context_length % self.seg_len != 0:
            warn.warn(
                f"context_length ({self.context_length}) is not divisible by "
                f"seg_len ({self.seg_len}). Trailing time steps will be ignored."
            )
        if self.prediction_length % self.seg_len != 0:
            warn.warn(
                f"prediction_length ({self.prediction_length}) is not divisible "
                f"by seg_len ({self.seg_len}). Trailing time steps will be ignored."
            )

        self.seg_num_x = self.context_length // self.seg_len
        self.seg_num_y = self.prediction_length // self.seg_len

        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )

        # Positional embedding  : (seg_num_y, d_model // 2)
        # Channel embedding     : (enc_in,    d_model // 2)
        # Both halves are concatenated to form a full d_model query vector.
        self.pos_emb = nn.Parameter(
            torch.randn(self.seg_num_y, self.d_model // 2)
        )
        self.channel_emb = nn.Parameter(
            torch.randn(self.enc_in, self.d_model // 2)
        )

        self.pred_nn = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len),
        )

    def _encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the full SegRNN encode-decode pass on a single batch.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, context_length, enc_in)``.

        Returns
        -------
        torch.Tensor
            Forecast tensor of shape ``(B, prediction_length, enc_in)``
            (or ``(B, seg_num_y * seg_len, enc_in)`` when
            ``prediction_length`` is not divisible by ``seg_len``).
        """
        B = x.size(0)

        # --- instance normalization (last-value subtraction) -----------------
        # seq_last: (B, 1, enc_in) — anchor for reversible normalization.
        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last).permute(0, 2, 1)  # (B, enc_in, context_length)

        # --- segment embedding -----------------------------------------------
        # (B, enc_in, context_length)
        #   → (B*enc_in, seg_num_x, seg_len)
        #   → (B*enc_in, seg_num_x, d_model)
        x = self.valueEmbedding(
            x.reshape(B * self.enc_in, self.seg_num_x, self.seg_len)
        )

        # --- encoder RNN pass ------------------------------------------------
        # x    : (B*enc_in, seg_num_x, d_model)
        # hn   : (1, B*enc_in, d_model)
        _, hn = self.rnn(x)

        # --- build decoder queries -------------------------------------------
        # pos_emb     : (seg_num_y, d_model//2)
        #               → (enc_in, seg_num_y, d_model//2)
        # channel_emb : (enc_in, d_model//2)
        #               → (enc_in, seg_num_y, d_model//2)
        # cat → (enc_in, seg_num_y, d_model)
        #     → (enc_in * seg_num_y, 1, d_model)
        #     → (B * enc_in * seg_num_y, 1, d_model)
        pos_emb = torch.cat(
            [
                self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
                self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1),
            ],
            dim=-1,
        ).view(-1, 1, self.d_model).repeat(B, 1, 1)
        # pos_emb : (B * enc_in * seg_num_y, 1, d_model)

        # Expand encoder hidden state to match decoder query batch size.
        # hn : (1, B*enc_in, d_model)
        #    → repeat seg_num_y times along the last dim
        #    → (1, B*enc_in, d_model * seg_num_y)
        #    → view → (1, B*enc_in*seg_num_y, d_model)
        hn_expanded = (
            hn.repeat(1, 1, self.seg_num_y)
            .view(1, B * self.enc_in * self.seg_num_y, self.d_model)
        )

        # --- decoder RNN pass ------------------------------------------------
        # hy : (1, B*enc_in*seg_num_y, d_model)
        _, hy = self.rnn(pos_emb, hn_expanded)

        # --- prediction head -------------------------------------------------
        # hy  : (1, B*enc_in*seg_num_y, d_model)
        #     → predict → (1, B*enc_in*seg_num_y, seg_len)
        #     → view  → (B, enc_in, prediction_length)
        #     → permute → (B, prediction_length, enc_in)
        y = self.pred_nn(hy).view(B, self.enc_in, self.seg_num_y * self.seg_len)
        y = y.permute(0, 2, 1)

        # --- reverse normalization -------------------------------------------
        return y + seq_last

    def _forecast(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract tensors from the batch dict and delegate to :meth:`_encoder`.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Batch dictionary.  Expected key:

            - ``'history_cont'`` : ``(B, context_length, enc_in)`` – historical
              continuous features.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, seg_num_y * seg_len, enc_in)``.
        """
        return self._encoder(x["history_cont"])

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the SegRNN model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Batch dictionary.  See :meth:`_forecast` for the expected keys.
            If the key ``'target_scale'`` is present its value is forwarded to
            :meth:`transform_output` for inverse scaling.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{'prediction': tensor}`` where ``tensor`` has shape
            ``(B, prediction_length, target_dim)``.
        """
        out = self._forecast(x)
        prediction = out[:, : self.prediction_length, : self.target_dim]

        if "target_scale" in x:
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}