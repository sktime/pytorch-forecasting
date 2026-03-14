"""SCINet v2 model for time series forecasting."""

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import MAE, Metric
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


def _make_conv_module(
    in_channels: int,
    hid_channels: int,
    kernel_size: int,
    dropout: float,
) -> nn.Sequential:
    """Build a single conv sub-module used inside an SCI-Block.

    Parameters
    ----------
    in_channels : int
        Number of input (and output) channels.
    hid_channels : int
        Intermediate channel width after the first convolution.
    kernel_size : int
        Kernel width for both Conv1d layers.
    dropout : float
        Dropout probability applied between the two convolutions.

    Returns
    -------
    nn.Sequential
        Conv1d → LeakyReLU → Dropout → Conv1d → Tanh pipeline.
    """
    pad = kernel_size // 2
    return nn.Sequential(
        nn.ReplicationPad1d(pad),
        nn.Conv1d(in_channels, hid_channels, kernel_size),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
        nn.ReplicationPad1d(pad),
        nn.Conv1d(hid_channels, in_channels, kernel_size),
        nn.Tanh(),
    )


class SCIBlock(nn.Module):
    """Single Sample-Convolution-and-Interaction block.

    Splits the input sequence into even- and odd-indexed sub-sequences,
    applies four distinct convolutional modules (phi, psi, rho, eta) to
    produce interactive, bi-directionally modulated outputs.

    Parameters
    ----------
    n_channels : int
        Number of input feature channels (C).
    hid_size : int, default=1
        Channel expansion factor for the hidden convolution layer.
        Hidden channels = n_channels * hid_size.
    kernel_size : int, default=5
        Kernel width for all Conv1d layers.
    dropout : float, default=0.5
        Dropout probability inside each conv module.
    """

    def __init__(
        self,
        n_channels: int,
        hid_size: int = 1,
        kernel_size: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()
        hid_channels = max(1, n_channels * hid_size)
        self.phi = _make_conv_module(n_channels, hid_channels, kernel_size, dropout)
        self.psi = _make_conv_module(n_channels, hid_channels, kernel_size, dropout)
        self.rho = _make_conv_module(n_channels, hid_channels, kernel_size, dropout)
        self.eta = _make_conv_module(n_channels, hid_channels, kernel_size, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the SCI interaction to even and odd sub-sequences.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, C)``.

        Returns
        -------
        even_out : torch.Tensor
            Shape ``(B, T//2, C)``.
        odd_out : torch.Tensor
            Shape ``(B, T//2, C)``.
        """
        even = x[:, 0::2, :]  # (B, T/2, C)
        odd = x[:, 1::2, :]  # (B, T/2, C)

        # Transpose to (B, C, T/2) for Conv1d
        even_t = even.permute(0, 2, 1)
        odd_t = odd.permute(0, 2, 1)

        # Multiplicative interaction
        even_scaled = even * torch.exp(self.phi(odd_t).permute(0, 2, 1))
        odd_scaled = odd * torch.exp(self.psi(even_t).permute(0, 2, 1))

        # Additive interaction
        even_out = even_scaled + self.rho(odd_scaled.permute(0, 2, 1)).permute(0, 2, 1)
        odd_out = odd_scaled + self.eta(even_scaled.permute(0, 2, 1)).permute(0, 2, 1)

        return even_out, odd_out


class SCITree(nn.Module):
    """Recursive binary tree of SCI-Blocks.

    At each level the sequence is split into two halves; each half is
    processed by a child ``SCITree`` of depth ``num_levels - 1``.
    The outputs are interleaved back into a sequence of the original
    length before being returned.

    Parameters
    ----------
    n_channels : int
        Number of feature channels.
    num_levels : int
        Depth of the binary decomposition tree (>= 1).
    hid_size : int, default=1
        Channel expansion factor forwarded to every ``SCIBlock``.
    kernel_size : int, default=5
        Kernel width forwarded to every ``SCIBlock``.
    dropout : float, default=0.5
        Dropout probability forwarded to every ``SCIBlock``.
    """

    def __init__(
        self,
        n_channels: int,
        num_levels: int,
        hid_size: int = 1,
        kernel_size: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.block = SCIBlock(n_channels, hid_size, kernel_size, dropout)

        if num_levels > 1:
            self.even_tree = SCITree(
                n_channels, num_levels - 1, hid_size, kernel_size, dropout
            )
            self.odd_tree = SCITree(
                n_channels, num_levels - 1, hid_size, kernel_size, dropout
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Recursively decompose, transform, and reconstruct the sequence.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, C)`` where ``T`` must be divisible by
            ``2 ** num_levels``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, C)``.
        """
        even_out, odd_out = self.block(x)

        if self.num_levels > 1:
            even_out = self.even_tree(even_out)
            odd_out = self.odd_tree(odd_out)

        # Interleave even and odd back into original order
        B, T_half, C = even_out.shape
        out = torch.empty(B, T_half * 2, C, device=x.device, dtype=x.dtype)
        out[:, 0::2, :] = even_out
        out[:, 1::2, :] = odd_out
        return out


class SCINetCore(nn.Module):
    """Core SCINet algorithm.

    Stacks ``num_stacks`` SCITree modules with residual connections,
    then decodes with a fully-connected layer.

    Parameters
    ----------
    context_length : int
        Length of the encoder input sequence. Must satisfy
        ``context_length % (2 ** num_levels) == 0``.
    prediction_length : int
        Number of future steps to predict.
    n_channels : int
        Number of input channels.
    num_stacks : int, default=1
        Number of stacked SCITree modules.
    num_levels : int, default=3
        Depth of the binary decomposition tree inside each SCITree.
    hid_size : int, default=1
        Channel expansion factor for the hidden conv layers.
    kernel_size : int, default=5
        Kernel width for all Conv1d layers.
    dropout : float, default=0.5
        Dropout probability inside each SCI-Block.
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        n_channels: int,
        num_stacks: int = 1,
        num_levels: int = 3,
        hid_size: int = 1,
        kernel_size: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_channels = n_channels

        self.trees = nn.ModuleList(
            [
                SCITree(n_channels, num_levels, hid_size, kernel_size, dropout)
                for _ in range(num_stacks)
            ]
        )
        self.fc = nn.Linear(context_length * n_channels, prediction_length * n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of SCINet.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, C)``.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, prediction_length, C)``.
        """
        for tree in self.trees:
            x = tree(x) + x  # residual connection

        B = x.shape[0]
        return self.fc(x.reshape(B, -1)).reshape(
            B, self.prediction_length, self.n_channels
        )


class SCINet_v2(BaseModel):
    """SCINet v2 model for time series forecasting.

    Based on the paper
    `SCINet: Time Series Modeling and Forecasting with Sample Convolution
    and Interaction Networks
    <https://arxiv.org/abs/2106.09305>`_ (NeurIPS 2022).

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
        SCINet_v2_pkg_v2 : type
            Package class associated with this model.
        """
        from pytorch_forecasting.models.scinet._scinet_pkg_v2 import SCINet_v2_pkg_v2

        return SCINet_v2_pkg_v2

    def __init__(
        self,
        *,
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

        self.model = SCINetCore(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            n_channels=self.n_channels,
            num_stacks=num_stacks,
            num_levels=num_levels,
            hid_size=hid_size,
            kernel_size=kernel_size,
            dropout=dropout,
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
        prediction = self.model(enc)
        return {"prediction": prediction}
