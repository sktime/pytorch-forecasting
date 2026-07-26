"""Decoder-only MLP for pytorch-forecasting v2."""

########################################################################################
# Disclaimer: This implementation is based on the new v2 data pipeline and is
# experimental, please use with care.
########################################################################################

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.base._base_model_v2 import BaseModel
from pytorch_forecasting.models.mlp.submodules import FullyConnectedModule


class DecoderMLP_v2(BaseModel):
    """MLP on the decoder for pytorch-forecasting v2.

    Predicts each future step purely from information known in the decoder
    (future-known covariates and static features). It intentionally does not use the
    encoder or target history -- it is a lightweight, covariate-driven baseline. The
    original v1 ``DecoderMLP`` was authored by the pytorch-forecasting team.

    Parameters
    ----------
    loss : nn.Module
        Loss function for training (required).
    hidden_size : int, default=300
        Hidden layer width of the MLP.
    n_hidden_layers : int, default=3
        Number of hidden layers.
    dropout : float, default=0.1
        Dropout probability.
    norm : bool, default=True
        Whether to apply ``LayerNorm`` in the MLP.
    activation_class : str, default="ReLU"
        Name of a ``torch.nn`` activation class.
    logging_metrics : Optional[list[nn.Module]], default=None
        Metrics to log during training, validation, and testing.
    optimizer : Optional[Union[Optimizer, str]], default="adam"
        Optimizer to use for training.
    optimizer_params : Optional[dict], default=None
        Parameters for the optimizer.
    lr_scheduler : Optional[str], default=None
        Learning rate scheduler to use.
    lr_scheduler_params : Optional[dict], default=None
        Parameters for the learning rate scheduler.
    metadata : Optional[dict], default=None
        Metadata from ``EncoderDecoderTimeSeriesDataModule``.
    """

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.mlp._decodermlp_pkg_v2 import DecoderMLP_pkg_v2

        return DecoderMLP_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        hidden_size: int = 300,
        n_hidden_layers: int = 3,
        dropout: float = 0.1,
        norm: bool = True,
        activation_class: str = "ReLU",
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.dropout = dropout
        self.norm = norm
        self.activation_class = activation_class
        self.metadata = metadata if metadata is not None else {}

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        # all dimensions are derived from metadata -- never hardcoded
        self.decoder_cont_dim = self.metadata.get("decoder_cont", 0)
        self.decoder_cat_dim = self.metadata.get("decoder_cat", 0)
        self.static_cat_dim = self.metadata.get("static_categorical_features", 0)
        self.static_cont_dim = self.metadata.get("static_continuous_features", 0)
        self.prediction_length = self.metadata.get("max_prediction_length", 1)
        self.target_dim = self.metadata.get("target", 1)

        self._init_network()

    def _init_network(self):
        """Build the per-step MLP from the derived dimensions."""
        self.input_size = (
            self.decoder_cont_dim
            + self.decoder_cat_dim
            + self.static_cat_dim
            + self.static_cont_dim
        )

        self.n_quantiles = None
        if isinstance(self.loss, QuantileLoss):
            self.n_quantiles = len(self.loss.quantiles)
        self.output_size = self.n_quantiles if self.n_quantiles is not None else 1

        self.mlp = FullyConnectedModule(
            input_size=max(1, self.input_size),
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            n_hidden_layers=self.n_hidden_layers,
            activation_class=getattr(nn, self.activation_class),
            dropout=self.dropout,
            norm=self.norm,
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass: per-step MLP over decoder and static features.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input dictionary containing at least ``decoder_cont`` and, when the
            corresponding metadata dimensions are non-zero, ``decoder_cat``,
            ``static_continuous_features`` and ``static_categorical_features``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"prediction": tensor}`` of shape
            ``(batch_size, prediction_length, output_size)``.
        """
        decoder_cont = x["decoder_cont"]
        batch_size = decoder_cont.shape[0]
        pred_len = self.prediction_length
        device = decoder_cont.device
        dtype = decoder_cont.dtype

        features = []
        if self.decoder_cont_dim > 0:
            features.append(x["decoder_cont"])
        if self.decoder_cat_dim > 0:
            features.append(x["decoder_cat"].to(dtype))
        if self.static_cont_dim > 0:
            features.append(x["static_continuous_features"].expand(-1, pred_len, -1))
        if self.static_cat_dim > 0:
            features.append(
                x["static_categorical_features"].to(dtype).expand(-1, pred_len, -1)
            )

        if features:
            network_input = torch.cat(features, dim=-1)
        else:
            network_input = torch.zeros(
                batch_size, pred_len, 1, device=device, dtype=dtype
            )

        prediction = self.mlp(network_input.reshape(-1, self.mlp.input_size)).reshape(
            batch_size, pred_len, self.output_size
        )

        return {"prediction": prediction}
