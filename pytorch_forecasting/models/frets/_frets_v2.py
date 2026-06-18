"""FreTS v2 model for time series forecasting."""

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.layers import FreTSCore
from pytorch_forecasting.metrics import MAE, Metric
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class FreTS(BaseModel):
    """FreTS v2 model for time series forecasting.

    Based on the paper
    `FreTS: Frequency-domain MLPs are More Effective Learners for
    Long-term Time Series Forecasting
    <https://arxiv.org/abs/2311.06184>`_.

    The model applies FFT to transform the input into the frequency domain,
    learns dominant frequency patterns via lightweight diagonal complex MLPs,
    reconstructs the signal via IFFT, and decodes with a FC layer.

    Parameters
    ----------
    embed_size : int, default=128
        Dimension of the learnable token embedding.
    hidden_size : int, default=256
        Hidden size of the FC output head.
    channel_independence : bool, default=True
        If True, each channel is processed independently (only temporal
        frequency mixing). If False, cross-channel frequency mixing is
        applied first.
    sparsity_threshold : float, default=0.01
        Soft-shrinkage threshold for frequency coefficient sparsity.
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
        FreTS_pkg_v2 : type
            Package class associated with this model.
        """
        from pytorch_forecasting.models.frets._frets_pkg_v2 import FreTS_pkg_v2

        return FreTS_pkg_v2

    def __init__(
        self,
        *,
        embed_size: int = 128,
        hidden_size: int = 256,
        channel_independence: bool = True,
        sparsity_threshold: float = 0.01,
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

        self.model = FreTSCore(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            n_channels=self.n_channels,
            embed_size=embed_size,
            hidden_size=hidden_size,
            channel_independence=channel_independence,
            sparsity_threshold=sparsity_threshold,
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the FreTS model.

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
