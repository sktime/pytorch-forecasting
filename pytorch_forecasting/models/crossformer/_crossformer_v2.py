"""
Crossformer model for PyTorch Forecasting (v2).
-----------------------------------------------

Port of `Crossformer: Transformer Utilizing Cross-Dimension Dependency for
Multivariate Time Series Forecasting` (ICLR 2023) to the ``TslibBaseModel``
API.

References
----------
[1] https://arxiv.org/abs/2211.13328
[2] https://github.com/Thinklab-SJTU/Crossformer
"""

from math import ceil
from typing import Any
from warnings import warn

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import Metric
from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel
from pytorch_forecasting.models.crossformer.sub_modules import (
    Decoder,
    DSW_embedding,
    Encoder,
)


class CrossFormer(TslibBaseModel):
    """
    Crossformer: Transformer Utilizing Cross-Dimension Dependency for
    Multivariate Time Series Forecasting.

    The model embeds each variable's history into non-overlapping segments
    (DSW embedding), processes them with a hierarchical encoder-decoder of
    Two-Stage Attention (TSA) layers, and predicts all future time steps at
    once. Only the target dimensions of the output are returned.

    Parameters
    ----------
    loss : Metric
        Loss function for training step.
    seg_len : int, default=5
        Length of each segment the input sequence is split into.
    win_size : int, default=2
        Window size for segment merging between encoder scales.
    factor : int, default=10
        Number of routers used in the cross-dimension stage of the TSA layer.
    d_model : int, default=256
        Dimension of the model.
    d_ff : int, optional
        Dimension of the feed-forward network in the TSA layer.
        Defaults to ``4 * d_model``.
    n_heads : int, default=8
        Number of attention heads. ``d_model`` must be divisible by ``n_heads``.
    e_layers : int, default=3
        Number of encoder scale blocks. The decoder uses ``e_layers + 1``
        layers, mirroring the reference implementation.
    dropout : float, default=0.1
        Dropout rate.
    baseline : bool, default=False
        If True, add the mean of the target history to the predictions
        (persistence-style baseline ablation from the reference implementation).
    logging_metrics : Optional[list[nn.Module]], default=None
        List of metrics to log during training, validation, and testing.
    optimizer : Optional[Union[Optimizer, str]], default='adam'
        Optimizer to use for training.
    optimizer_params : Optional[dict], default=None
        Parameters for the optimizer.
    lr_scheduler : Optional[str], default=None
        Learning rate scheduler to use.
    lr_scheduler_params : Optional[dict], default=None
        Parameters for the learning rate scheduler.
    metadata : Optional[dict], default=None
        Metadata for the model from TslibDataModule.

    Notes
    -----
    - Categorical covariates and known future covariates are not used; the
      model is a pure encoder-only forecaster over the target history and
      continuous covariate history.
    - The number of input dimensions is ``target_dim + cont_dim`` (target
      variables first, then continuous covariates). The decoder predicts all
      dimensions jointly and only the target dimensions are returned.
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.crossformer._crossformer_forecaster_v2 import (
            CrossFormerForecaster,
        )

        return CrossFormerForecaster

    def __init__(
        self,
        loss: Metric,
        seg_len: int = 5,
        win_size: int = 2,
        factor: int = 10,
        d_model: int = 256,
        d_ff: int | None = None,
        n_heads: int = 8,
        e_layers: int = 3,
        dropout: float = 0.1,
        baseline: bool = False,
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

        warn(
            "CrossFormer is an experimental model implemented on "
            "TslibBaseModelV2. It is an unstable version and may be subject "
            "to unannounced changes. Please use with caution.",
            UserWarning,
        )

        self.seg_len = seg_len
        self.win_size = win_size
        self.factor = factor
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.dropout = dropout
        self.baseline = baseline
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self.data_dim = self.target_dim + self.cont_dim
        if self.data_dim == 0:
            raise ValueError(
                "CrossFormer requires at least one input variable "
                "(target or continuous covariate)."
            )

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )

        self.n_quantiles = 1
        if hasattr(self._loss, "quantiles") and self._loss.quantiles is not None:
            self.n_quantiles = len(self._loss.quantiles)

        # padding to handle segment lengths that do not divide the sequence
        self.pad_in_len = ceil(1.0 * self.context_length / self.seg_len) * self.seg_len
        self.pad_out_len = (
            ceil(1.0 * self.prediction_length / self.seg_len) * self.seg_len
        )
        self.in_len_add = self.pad_in_len - self.context_length

        self._init_network()

    def _init_network(self):
        """Initialize the Crossformer encoder-decoder architecture."""
        in_seg_num = self.pad_in_len // self.seg_len
        out_seg_num = self.pad_out_len // self.seg_len

        self.enc_value_embedding = DSW_embedding(self.seg_len, self.d_model)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, self.data_dim, in_seg_num, self.d_model)
        )
        self.pre_norm = nn.LayerNorm(self.d_model)

        self.encoder = Encoder(
            self.e_layers,
            self.win_size,
            self.d_model,
            self.n_heads,
            self.d_ff,
            block_depth=1,
            dropout=self.dropout,
            in_seg_num=in_seg_num,
            factor=self.factor,
        )

        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, self.data_dim, out_seg_num, self.d_model)
        )
        self.decoder = Decoder(
            self.seg_len,
            self.e_layers + 1,
            self.d_model,
            self.n_heads,
            self.d_ff,
            self.dropout,
            out_seg_num=out_seg_num,
            factor=self.factor,
        )

    def _forecast(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the Crossformer model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input data containing the target and continuous covariate history.

        Returns
        -------
        torch.Tensor
            Predictions of shape
            ``(batch_size, prediction_length, target_dim [, n_quantiles])``.
        """
        history_target = x.get(
            "history_target",
            torch.zeros(
                x["history_cont"].shape[0],
                self.context_length,
                self.target_dim,
                device=self.device,
            ),
        )
        if history_target.dim() == 2:
            history_target = history_target.unsqueeze(-1)

        history_cont = x.get(
            "history_cont",
            torch.zeros(
                history_target.shape[0],
                history_target.shape[1],
                0,
                device=self.device,
            ),
        )

        # concatenate target variables first, then continuous covariates so
        # that the first ``target_dim`` dimensions of the output are targets
        series = torch.cat([history_target, history_cont], dim=-1)

        if self.baseline:
            base = history_target.mean(dim=1, keepdim=True)
        else:
            base = 0

        batch_size = series.shape[0]
        if self.in_len_add != 0:
            series = torch.cat(
                (
                    series[:, :1, :].expand(-1, self.in_len_add, -1),
                    series,
                ),
                dim=1,
            )

        x_seq = self.enc_value_embedding(series)
        x_seq = x_seq + self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        enc_out = self.encoder(x_seq)

        dec_in = self.dec_pos_embedding.expand(batch_size, -1, -1, -1)
        predict_y = self.decoder(dec_in, enc_out)

        prediction = predict_y[:, : self.prediction_length, : self.target_dim]

        if self.baseline:
            prediction = base + prediction

        if self.target_dim == 1:
            prediction = prediction.squeeze(-1)

        prediction = prediction.unsqueeze(-1)
        if self.n_quantiles > 1:
            prediction = prediction.expand(*prediction.shape[:-1], self.n_quantiles)

        return prediction

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the CrossFormer model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input data.

        Returns
        -------
        dict[str, torch.Tensor]
            Model predictions.
        """
        prediction = self._forecast(x)

        if "target_scale" in x:
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}
