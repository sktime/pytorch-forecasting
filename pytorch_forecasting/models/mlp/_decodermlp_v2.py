"""
DecoderMLP v2 model for time series forecasting.

MLP that predicts output only based on information available in the decoder
(known future covariates).
"""

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import MAE, Metric
from pytorch_forecasting.models.base._base_model_v2 import BaseModel
from pytorch_forecasting.models.mlp.submodules import FullyConnectedModule


class DecoderMLP(BaseModel):
    """MLP on the decoder - v2 implementation.

    MLP that predicts output only based on information available in the decoder,
    i.e., known future covariates. This is a simple yet effective baseline model
    for time series forecasting when strong future covariates are available.

    Parameters
    ----------
    loss : nn.Module
        Loss function to use for training.
    activation_class : str, default="ReLU"
        Name of the PyTorch activation class (e.g., ``"ReLU"``, ``"GELU"``).
    hidden_size : int, default=300
        Hidden layer size of the fully connected network.
    n_hidden_layers : int, default=3
        Number of hidden layers in the MLP.
    dropout : float, default=0.1
        Dropout rate applied after each hidden layer.
    norm : bool, default=True
        Whether to apply LayerNorm after each hidden layer.
    logging_metrics : list[nn.Module] or None, default=None
        Metrics to log during training, validation, and testing.
    optimizer : Optimizer or str or None, default="adam"
        Optimizer to use for training.
    optimizer_params : dict or None, default=None
        Parameters for the optimizer.
    lr_scheduler : str or None, default=None
        Learning rate scheduler to use.
    lr_scheduler_params : dict or None, default=None
        Parameters for the learning rate scheduler.
    metadata : dict or None, default=None
        Metadata from ``EncoderDecoderTimeSeriesDataModule``.

    Examples
    --------
    >>> from pytorch_forecasting.models.mlp._decodermlp_v2 import DecoderMLP
    >>> from pytorch_forecasting.metrics import MAE
    >>> metadata = {
    ...     "max_encoder_length": 10,
    ...     "max_prediction_length": 3,
    ...     "decoder_cont": 2,
    ...     "decoder_cat": 0,
    ...     "target": 1,
    ...     "encoder_cont": 2,
    ... }
    >>> model = DecoderMLP(loss=MAE(), metadata=metadata)
    >>> import torch
    >>> x = {
    ...     "decoder_cont": torch.randn(4, 3, 2),
    ...     "decoder_cat": torch.zeros(4, 3, 0),
    ... }
    >>> out = model(x)
    >>> out["prediction"].shape
    torch.Size([4, 3, 1])
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.mlp._decodermlp_pkg_v2 import (
            DecoderMLP_pkg_v2,
        )

        return DecoderMLP_pkg_v2

    def __init__(
        self,
        loss: Metric | None = None,
        *,
        activation_class: str = "ReLU",
        hidden_size: int = 300,
        n_hidden_layers: int = 3,
        dropout: float = 0.1,
        norm: bool = True,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ):
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
        self.metadata = metadata

        # Extract dimensions from metadata
        self.prediction_length = metadata["max_prediction_length"]
        self.decoder_cont_dim = metadata.get("decoder_cont", 0)
        self.decoder_cat_dim = metadata.get("decoder_cat", 0)
        self.target_dim = metadata.get("target", 1)

        # Determine quantile count from loss
        self.n_quantiles = 1
        if hasattr(loss, "quantiles") and loss.quantiles is not None:
            self.n_quantiles = len(loss.quantiles)

        # Compute input/output sizes
        input_size = self.decoder_cont_dim + self.decoder_cat_dim
        if input_size == 0:
            # Fallback: use encoder_cont as input dimension if no decoder
            # features are available (the model still needs some input)
            input_size = metadata.get("encoder_cont", 1)

        output_size = self.target_dim * self.n_quantiles

        # Build MLP network
        self.mlp = FullyConnectedModule(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            n_hidden_layers=n_hidden_layers,
            activation_class=getattr(nn, activation_class),
            dropout=dropout,
            norm=norm,
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass of the DecoderMLP model.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary containing input tensors from the dataloader.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - ``prediction``: tensor of shape
              ``(batch_size, prediction_length, n_quantiles)``
              or ``(batch_size, prediction_length, 1)`` for point forecasts.
        """
        # Collect decoder inputs
        inputs = []

        decoder_cont = x.get("decoder_cont")
        if decoder_cont is not None and decoder_cont.shape[-1] > 0:
            inputs.append(decoder_cont)

        decoder_cat = x.get("decoder_cat")
        if decoder_cat is not None and decoder_cat.shape[-1] > 0:
            inputs.append(decoder_cat)

        if inputs:
            network_input = torch.cat(inputs, dim=-1)
        else:
            # Fallback: use encoder_cont if no decoder features are available
            network_input = x["encoder_cont"]

        # network_input shape: (batch_size, prediction_length, input_size)
        batch_size = network_input.size(0)
        seq_len = network_input.size(1)

        # Run through MLP: flatten time steps, apply MLP, reshape
        prediction = self.mlp(
            network_input.reshape(-1, self.mlp.input_size)
        ).reshape(batch_size, seq_len, -1)

        return {"prediction": prediction}
