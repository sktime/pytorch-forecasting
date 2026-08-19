"""SCINet v2 model for time series forecasting."""

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers._encoders._scinet_encoder import SCITree
from pytorch_forecasting.metrics import MAE, Metric
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class SCINet_v2(BaseModel):
    """SCINet v2 model for time series forecasting.

    Based on the paper
    `SCINet: Time Series Modeling and Forecasting with Sample Convolution
    and Interaction Networks
    <https://arxiv.org/abs/2106.09305>`_ by Liu et al. (NeurIPS 2022),
    adapted from the authors' original implementation at
    `cure-lab/SCINet <https://github.com/cure-lab/SCINet>`_.

    The model recursively splits the input sequence into even- and
    odd-indexed sub-sequences, applies interactive convolutional
    transformations at each level of a binary tree, and reconstructs
    the enhanced sequence before decoding with a FC layer.

    Parameters
    ----------
    num_stacks : int, default=1
        Number of stacked SCITree modules.
    num_levels : int, default=3
        Depth of the binary decomposition tree.
        Input sequence length must satisfy
        ``context_length % (2 ** num_levels) == 0``.
    hid_size : int, default=1
        Channel expansion factor for the hidden conv layers inside
        each SCI-Block.  Hidden channels = n_channels * hid_size.
    kernel_size : int, default=5
        Kernel width for all Conv1d layers.
    dropout : float, default=0.5
        Dropout probability inside each SCI-Block.
    loss : Metric, optional
        Loss to optimise. Defaults to
        :class:`~pytorch_forecasting.metrics.MAE`.
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
    metadata : dict
        Dataset metadata produced by
        :class:`~pytorch_forecasting.data.data_module\
.EncoderDecoderTimeSeriesDataModule`.
        Must contain ``"max_encoder_length"`` and
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
        SCINet_pkg_v2 : type
            Package class associated with this model.
        """
        from pytorch_forecasting.models.scinet._scinet_pkg_v2 import SCINet_pkg_v2

        return SCINet_pkg_v2

    def __init__(
        self,
        num_stacks: int = 1,
        num_levels: int = 3,
        hid_size: int = 1,
        kernel_size: int = 5,
        dropout: float = 0.5,
        loss: Metric | None = None,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs,
    ):
        if metadata is None:
            raise ValueError("metadata is required")
        if loss is None:
            loss = MAE()

        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            **kwargs,
        )
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        self.metadata = metadata

        self.context_length = metadata["max_encoder_length"]
        self.prediction_length = metadata["max_prediction_length"]
        self.n_channels = metadata.get("target", 1)

        # Validate that context_length is divisible by 2^num_levels
        required_divisor = 2**num_levels
        if self.context_length % required_divisor != 0:
            raise ValueError(
                f"context_length ({self.context_length}) must be divisible by "
                f"2 ** num_levels ({required_divisor}). "
                f"Reduce num_levels or adjust context_length."
            )

        self.trees = nn.ModuleList(
            [
                SCITree(self.n_channels, num_levels, hid_size, kernel_size, dropout)
                for _ in range(num_stacks)
            ]
        )
        self.fc = nn.Linear(
            self.context_length * self.n_channels,
            self.prediction_length * self.n_channels,
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the SCINet model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input batch containing:

            * ``"target_past"`` : tensor of shape
              ``(batch_size, context_length, n_channels)``

        Returns
        -------
        out : dict[str, torch.Tensor]
            Dictionary containing:

            * ``"prediction"`` : tensor of shape
              ``(batch_size, prediction_length, n_channels)``
        """
        enc = x["target_past"]

        for tree in self.trees:
            enc = tree(enc) + enc  # residual connection

        batch_size = enc.shape[0]
        prediction = self.fc(enc.reshape(batch_size, -1)).reshape(
            batch_size, self.prediction_length, self.n_channels
        )
        return {"prediction": prediction}
