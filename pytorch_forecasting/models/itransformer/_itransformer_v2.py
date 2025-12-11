from typing import Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class iTransformer(TslibBaseModel):
    """
    An implementation of iTransformer model for v2 of pytorch-forecasting.

    iTransformer repurposes the Transformer architecture by applying attention
    and feed-forward networks on inverted dimensions. Instead of treating
    timestamps as tokens (like traditional Transformers), iTransformer embeds
    individual time series as variate tokens. The attention mechanism captures
    multivariate correlations, while the feed-forward network learns nonlinear
    representations for each variate. This inversion enables better handling
    of long lookback windows, improved generalization across different variates,
    and state-of-the-art performance on real-world forecasting tasks without
    modifying the basic Transformer components.

    Parameters
    ----------
    loss: nn.Module
        Loss function to use for training.
    output_attention: bool, default=False
        Whether to output attention weights.
    factor: int, default=5
        Factor for the attention mechanism, controlling keys and values.
    d_model: int, default=512
        Dimension of the model embeddings and hidden representations.
    d_ff: int, default=2048
        Dimension of the feed-forward network.
    activation: str, default='relu'
        Activation function to use in the feed-forward network.
    dropout: float, default=0.1
        Dropout rate for regularization.
    n_heads: int, default=8
        Number of attention heads in the multi-head attention mechanism.
    e_layers: int, default=3
        Number of encoder layers in the transformer architecture.
    logging_metrics: Optional[list[nn.Module]], default=None
        List of metrics to log during training, validation, and testing.
    optimizer: Optional[Union[Optimizer, str]], default='adam'
        Optimizer to use for training. Can be a string name or an instance of an optimizer.
    optimizer_params: Optional[dict], default=None
        Parameters for the optimizer. If None, default parameters for the optimizer will be used.
    lr_scheduler: Optional[str], default=None
        Learning rate scheduler to use. If None, no scheduler is used.
    lr_scheduler_params: Optional[dict], default=None
        Parameters for the learning rate scheduler. If None, default parameters for the scheduler will be used.
    metadata: Optional[dict], default=None
        Metadata for the model from TslibDataModule. This can include information about the dataset,
        such as the number of time steps, number of features, etc. It is used to initialize the model
        and ensure it is compatible with the data being used.

    References
    ----------
    [1] https://arxiv.org/pdf/2310.06625
    [2] https://github.com/thuml/iTransformer/blob/main/model/iTransformer.py

    Notes
    -----
    [1] The `iTransformer` model obtains many of its attributes from the `TslibBaseModel` class, which is a base class
        where a lot of the boiler plate code for metadata handling and model initialization is implemented.
    """  # noqa: E501

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.itransformer._itransformer_pkg_v2 import (
            iTransformer_pkg_v2,
        )

        return iTransformer_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        output_attention: bool = False,
        factor: int = 5,
        d_model: int = 512,
        d_ff: int = 2048,
        activation: str = "relu",
        dropout: float = 0.1,
        n_heads: int = 8,
        e_layers: int = 3,
        logging_metrics: Optional[list[nn.Module]] = None,
        optimizer: Optional[Union[Optimizer, str]] = "adam",
        optimizer_params: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
        **kwargs,
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

        self.output_attention = output_attention
        self.factor = factor
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.dropout = dropout
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.freq = self.metadata.get("freq", "h")

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """
        Initialize the network for iTransformer's architecture.
        """
        from pytorch_forecasting.layers import (
            AttentionLayer,
            DataEmbedding_inverted,
            Encoder,
            EncoderLayer,
            FullAttention,
        )

        self.enc_embedding = DataEmbedding_inverted(
            self.context_length, self.d_model, self.dropout
        )

        self.n_quantiles = None

        if hasattr(self.loss, "quantiles") and self.loss.quantiles is not None:
            self.n_quantiles = len(self.loss.quantiles)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    self_attention=AttentionLayer(
                        FullAttention(
                            False,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=self.output_attention,
                        ),
                        self.d_model,
                        self.n_heads,
                    ),
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
        )
        if self.n_quantiles is not None:
            self.projector = nn.Linear(
                self.d_model, self.prediction_length * self.n_quantiles, bias=True
            )
        else:
            self.projector = nn.Linear(self.d_model, self.prediction_length, bias=True)

    def _forecast(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the iTransformer model.
        Args:
            x (dict[str, torch.Tensor]): Input data.
            Returns:
                dict[str, torch.Tensor]: Model predictions.
        """
        x_enc = x["history_target"]
        x_mark_enc = x["history_cont"]

        _, _, N = x_enc.shape  # B L N
        # Embedding
        # B L N -> B N E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp)
        # B N E -> B N E
        # the dimensions of embedded time series has been inverted
        enc_out, attns = self.encoder(enc_out, x_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[
            :, :, :N
        ]  # filter covariates
        if self.output_attention:
            return dec_out, attns
        return dec_out

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the iTransformer model.
        Args:
            x (dict[str, torch.Tensor]): Input data.
        Returns:
            dict[str, torch.Tensor]: Model predictions.
        """
        dec_out, attns = self._forecast(x)

        if self.n_quantiles is not None:
            batch_size = dec_out.shape[0]
            dec_out = dec_out.reshape(
                batch_size, self.prediction_length, self.n_quantiles
            )

        prediction = dec_out[:, -self.prediction_length :, :]

        if "target_scale" in x:
            prediction = self.transform_output(prediction, x["target_scale"])

        if self.output_attention:
            return {"prediction": prediction, "attention": attns}
        return {"prediction": prediction}
