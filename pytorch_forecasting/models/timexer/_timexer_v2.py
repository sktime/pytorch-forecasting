"""
Time Series Transformer with eXogenous variables (TimeXer)
----------------------------------------------------------
"""

################################################################
# NOTE: This implementation of TimeXer derives from PR #1797.  #
# It is experimental and seeks to clarify design decisions.    #
# IT IS STRICTLY A PART OF THE v2 design of PTF. It overrides  #
# the v1 version introduced in PTF by PR #1797                  #
################################################################

from typing import Any, Optional, Union
import warnings as warn

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class TimeXer(TslibBaseModel):
    """
    An implementation of TimeXer model for v2 of pytorch-forecasting.

    TimeXer empowers the canonical transformer with the ability to reconcile
    endogenous and exogenous information without any architectural modifications
    and achieves consistent state-of-the-art performance across twelve real-world
    forecasting benchmarks.

    TimeXer employs patch-level and variate-level representations respectively for
    endogenous and exogenous variables, with an endogenous global token as a bridge
    in-between. With this design, TimeXer can jointly capture intra-endogenous
    temporal dependencies and exogenous-to-endogenous correlations.

    Parameters
    ----------
    loss: nn.Module
        Loss function to use for training.
    enc_in: int, optional
        Number of input features for the encoder. If not provided, it will be set to
        the number of continuous features in the dataset.
    hidden_size: int, default=512
        Dimension of the model embeddings and hidden representations of features.
    n_heads: int, default=8
        Number of attention heads in the multi-head attention mechanism.\
    e_layers: int, default=2
        Number of encoder layers in the transformer architecture.
    d_ff: int, default=2048
        Dimension of the feed-forward network in the transformer architecture.
    dropout: float, default=0.1
        Dropout rate for regularization. This is used throughout the model to prevent overfitting.
    patch_length: int, default=24
        Length of each non-overlapping patch for endogenous variable tokenization.
    factor: int, default=5
        Factor for the attention mechanism, controlling the number of keys and values.
    activation: str, default='relu'
        Activation function to use in the feed-forward network. Common choices are 'relu', 'gelu', etc.
    use_efficient_attention: bool, default=False
        If set to True, will use PyTorch's native, optimized Scaled Dot Product
        Attention implementation which can reduce computation time and memory
        consumption for longer sequences. PyTorch automatically selects the
        optimal backend (FlashAttention-2, Memory-Efficient Attention, or their
        own C++ implementation) based on user's input properties, hardware
        capabilities, and build configuration.
    endogenous_vars: Optional[list[str]], default=None
        List of endogenous variable names to be used in the model. If None, all historical values
        for the target variable are used.
    exogenous_vars: Optional[list[str]], default=None
        List of exogenous variable names to be used in the model. If None, all historical values
        for continuous variables are used.
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
    [1] https://arxiv.org/abs/2402.19072
    [2] https://github.com/thuml/TimeXer

    Notes
    -----
    [1] This implementation handles only continuous variables in the context length. Categorical variables
        support will be added in the future.
    [2] The `TimeXer` model obtains many of its attributes from the `TslibBaseModel` class, which is a base class
        where a lot of the boiler plate code for metadata handling and model initialization is implemented.
    """  # noqa: E501

    @classmethod
    def _pkg(cls):
        """Package containing the model."""
        from pytorch_forecasting.models.timexer._timexer_pkg_v2 import TimeXer_pkg_v2

        return TimeXer_pkg_v2

    def __init__(
        self,
        loss: nn.Module,
        enc_in: int = None,
        hidden_size: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        patch_length: int = 4,
        factor: int = 5,
        activation: str = "relu",
        use_efficient_attention: bool = False,
        endogenous_vars: Optional[list[str]] = None,
        exogenous_vars: Optional[list[str]] = None,
        logging_metrics: Optional[list[nn.Module]] = None,
        optimizer: Optional[Union[Optimizer, str]] = "adam",
        optimizer_params: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
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

        warn.warn(
            "TimeXer is an experimental model implemented on TslibBaseModelV2. "
            "It is an unstable version and maybe subject to unannouced changes."
            "Please use with caution. Feedback on the design and implementation is"
            ""
            "welcome. On the issue #1833 - https://github.com/sktime/pytorch-forecasting/issues/1833",
        )

        self.enc_in = enc_in
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.patch_length = patch_length
        self.activation = activation
        self.use_efficient_attention = use_efficient_attention
        self.factor = factor
        self.endogenous_vars = endogenous_vars
        self.exogenous_vars = exogenous_vars
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        self._init_network()

    def _init_network(self):
        """
        Initialize the network for TimeXer's architecture.
        """

        from pytorch_forecasting.layers import (
            AttentionLayer,
            DataEmbedding_inverted,
            Encoder,
            EncoderLayer,
            EnEmbedding,
            FlattenHead,
            FullAttention,
        )

        if self.context_length <= self.patch_length:
            raise ValueError(
                f"Context length ({self.context_length}) must be greater than patch"
                "length. Patches of ({self.patch_length}) will end up being longer than"
                "the sequence length."
            )

        if self.context_length % self.patch_length != 0:
            warn.warn(
                f"Context length ({self.context_length}) is not divisible by"
                " patch length. This may lead to unexpected behavior, as some"
                "time steps will not be used in the model."
            )

        self.patch_num = max(1, int(self.context_length // self.patch_length))

        if self.target_dim > 1 and self.features == "M":
            self.n_target_vars = self.target_dim
        else:
            self.n_target_vars = 1

        # currently enc_in is set only to cont_dim since
        # the data module doesn't fully support categorical
        # variables in the context length and modele expects
        # float values.
        self.enc_in = self.enc_in or self.cont_dim

        self.n_quantiles = None

        if hasattr(self.loss, "quantiles") and self.loss.quantiles is not None:
            self.n_quantiles = len(self.loss.quantiles)

        if self.hidden_size % self.n_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by n_heads ({self.n_heads}) "  # noqa: E501
                f"for multi-head attention mechanism to work properly."
            )

        self.en_embedding = EnEmbedding(
            self.n_target_vars, self.hidden_size, self.patch_length, self.dropout
        )

        self.ex_embedding = DataEmbedding_inverted(
            self.context_length, self.hidden_size, self.dropout
        )

        encoder_layers = []

        for _ in range(self.e_layers):
            encoder_layers.append(
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                            use_efficient_attention=self.use_efficient_attention,
                        ),
                        self.hidden_size,
                        self.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                            use_efficient_attention=self.use_efficient_attention,
                        ),
                        self.hidden_size,
                        self.n_heads,
                    ),
                    self.hidden_size,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
            )

        self.encoder = Encoder(
            encoder_layers, norm_layer=torch.nn.LayerNorm(self.hidden_size)
        )

        # Initialize output head
        self.head_nf = self.hidden_size * (self.patch_num + 1)
        self.head = FlattenHead(
            self.enc_in,
            self.head_nf,
            self.prediction_length,
            head_dropout=self.dropout,
            n_quantiles=self.n_quantiles,
        )

    def _forecast(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the TimeXer model.
        Args:
            x (dict[str, torch.Tensor]): Input data.
        Returns:
            dict[str, torch.Tensor]: Model predictions.
        """
        batch_size = x["history_cont"].shape[0]
        history_cont = x["history_cont"]
        history_time_idx = x.get("history_time_idx", None)

        history_target = x.get(
            "history_target",
            torch.zeros(batch_size, self.context_length, 1, device=self.device),
        )  # noqa: E501

        if history_time_idx is not None and history_time_idx.dim() == 2:
            # change [batch_size, time_steps] to [batch_size, time_steps, features]
            history_time_idx = history_time_idx.unsqueeze(-1)

        # v2 convention:
        # - endogenous information comes from the target history
        # - exogenous information comes from all continuous covariates
        endogenous_cont = history_target
        exogenous_cont = history_cont

        en_embed, n_vars = self.en_embedding(endogenous_cont)
        ex_embed = self.ex_embedding(exogenous_cont, history_time_idx)

        enc_out = self.encoder(en_embed, ex_embed)

        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )

        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)

        return dec_out

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the TimeXer model.
        Args:
            x (dict[str, torch.Tensor]): Input data.
        Returns:
            dict[str, torch.Tensor]: Model predictions.
        """

        out = self._forecast(x)
        prediction = out[:, : self.prediction_length, :]

        if "target_scale" in x:
            prediction = self.transform_output(prediction, x["target_scale"])

        return {"prediction": prediction}
