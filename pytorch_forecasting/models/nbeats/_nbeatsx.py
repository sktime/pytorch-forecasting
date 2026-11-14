"""N-BEATS with static, historical, and known future covariates."""

import torch
from torch import nn

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.layers._nbeats._nbeatsx import NBEATSxBlock
from pytorch_forecasting.metrics import (
    MASE,
    CrossEntropy,
    DistributionLoss,
    MultiLoss,
    QuantileLoss,
)
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.nbeats import _nbeatsx_pkg
from pytorch_forecasting.models.nn import MultiEmbedding


class NBeatsX(BaseModelWithCovariates):
    """NBEATSx for fixed-length, single-target point forecasting.

    Implements the generic and interpretable configurations of
    `NBEATSx <https://arxiv.org/abs/2104.05522>`_. Each block conditions on the
    target residual, historical covariates, known future covariates, and static
    features. Categorical covariates use learned embeddings. Unknown future
    covariates and future targets are never used as inputs.

    Parameters
    ----------
    stack_types : list of str, optional
        Basis for each stack: ``generic``, ``trend``, ``seasonality``, or
        ``exogenous``. Defaults to ``["generic"]``. For an interpretable model,
        use ``["trend", "seasonality", "exogenous"]``. Exogenous stacks require
        temporal covariates available in both the encoder and decoder.
    num_blocks : list of int, optional
        Blocks per stack. Defaults to [3]. A single value applies to every stack.
    num_block_layers : list of int, optional
        Hidden layers per block. Defaults to [4]; broadcasts like ``num_blocks``.
    widths : list of int, optional
        Hidden width per stack. Defaults to [256]; broadcasts like ``num_blocks``.
    sharing : list of bool, optional
        Share block weights within each stack. Defaults to [False]; broadcasts.
    expansion_coefficient_lengths : list of int, optional
        Polynomial coefficient count for trend and minimum period for
        seasonality. Defaults to [3]; broadcasts. Ignored for other stack types.
    context_length : int, default=1
        Fixed encoder length, inferred by ``from_dataset``.
    prediction_length : int, default=1
        Fixed decoder length, inferred by ``from_dataset``.
    dropout : float, default=0.1
        Dropout probability after each hidden layer.
    backcast_loss_ratio : float, default=0.1
        Weight of normalized-space mean squared backcast error in the training
        objective. Set to zero for forecast-only training; the final backcast
        head will then remain untrained unless its weights are shared.
    target : str, default="target"
        Target column in ``x_reals``, inferred by ``from_dataset``.
    loss : MultiHorizonMetric, optional
        Point forecast loss. Defaults to MASE. Distribution and quantile losses
        are not supported.
    static_categoricals, static_reals : list of str, optional
        Static feature names, inferred by ``from_dataset``.
    time_varying_categoricals_encoder, time_varying_reals_encoder : list, optional
        Historical feature names, inferred by ``from_dataset``.
    time_varying_categoricals_decoder, time_varying_reals_decoder : list, optional
        Known future feature names, inferred by ``from_dataset``.
    x_reals, x_categoricals : list of str, optional
        Column order in input tensors, inferred by ``from_dataset``.
    embedding_sizes, embedding_labels, categorical_groups : dict, optional
        Categorical embedding metadata, inferred by ``from_dataset``.
    embedding_paddings : list of str, optional
        Categoricals whose padding embedding is zero, inferred by ``from_dataset``.
    **kwargs
        Additional arguments to :class:`~pytorch_forecasting.models.BaseModel`.

    Notes
    -----
    ``prediction`` and ``backcast`` are inverse-normalized. For losses with a log
    link (Poisson or Tweedie), use the normalizer specified by that loss;
    ``predict()`` also applies the loss's inverse link to return target units.
    ``block_forecasts`` and
    ``block_backcasts`` have shape (batch, time, block) and remain in normalized
    target space so they add up exactly before inverse normalization. With a
    nonlinear target transformation, components are not additive in target units.
    Convolutional exogenous encoders and ensembles from the paper are not included.

    Examples
    --------
    Create a model from a fixed-length dataset containing covariates::

        model = NBeatsX.from_dataset(training, widths=[64])
        interpretable = NBeatsX.from_dataset(
            training, stack_types=["trend", "seasonality", "exogenous"]
        )
    """

    @classmethod
    def _pkg(cls):
        """Return the model's package metadata."""
        return _nbeatsx_pkg.NBeatsX_pkg

    def __init__(
        self,
        stack_types: list[str] | None = None,
        num_blocks: list[int] | None = None,
        num_block_layers: list[int] | None = None,
        widths: list[int] | None = None,
        sharing: list[bool] | None = None,
        expansion_coefficient_lengths: list[int] | None = None,
        context_length: int = 1,
        prediction_length: int = 1,
        dropout: float = 0.1,
        backcast_loss_ratio: float = 0.1,
        target: str = "target",
        static_categoricals: list[str] | None = None,
        static_reals: list[str] | None = None,
        time_varying_categoricals_encoder: list[str] | None = None,
        time_varying_categoricals_decoder: list[str] | None = None,
        time_varying_reals_encoder: list[str] | None = None,
        time_varying_reals_decoder: list[str] | None = None,
        x_reals: list[str] | None = None,
        x_categoricals: list[str] | None = None,
        embedding_sizes: dict[str, tuple[int, int]] | None = None,
        embedding_labels: dict | None = None,
        embedding_paddings: list[str] | None = None,
        categorical_groups: dict[str, list[str]] | None = None,
        loss=None,
        **kwargs,
    ):
        if stack_types is None:
            stack_types = ["generic"]
        if not stack_types or any(
            name not in {"generic", "trend", "seasonality", "exogenous"}
            for name in stack_types
        ):
            raise ValueError("stack_types must contain valid NBEATSx stack types")
        if context_length < 1 or prediction_length < 1:
            raise ValueError("context_length and prediction_length must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= backcast_loss_ratio < float("inf"):
            raise ValueError("backcast_loss_ratio must be finite and nonnegative")

        static_categoricals = static_categoricals or []
        static_reals = static_reals or []
        time_varying_categoricals_encoder = time_varying_categoricals_encoder or []
        time_varying_categoricals_decoder = time_varying_categoricals_decoder or []
        if time_varying_reals_encoder is None:
            time_varying_reals_encoder = [target]
        time_varying_reals_decoder = time_varying_reals_decoder or []
        if x_reals is None:
            x_reals = [target]
        x_categoricals = x_categoricals or []
        embedding_sizes = embedding_sizes or {}
        embedding_labels = embedding_labels or {}
        embedding_paddings = embedding_paddings or []
        categorical_groups = categorical_groups or {}
        if target not in x_reals or target not in time_varying_reals_encoder:
            raise ValueError("target must be an encoder real variable in x_reals")
        if target in time_varying_reals_decoder or target in static_reals:
            raise ValueError("target must not be a decoder or static variable")
        if loss is None:
            loss = MASE()
        if isinstance(loss, (CrossEntropy, DistributionLoss, QuantileLoss, MultiLoss)):
            raise ValueError("NBeatsX supports single-target point losses only")

        self.save_hyperparameters(ignore=["loss"])
        super().__init__(loss=loss, **kwargs)
        for name, default in (
            ("num_blocks", 3),
            ("num_block_layers", 4),
            ("widths", 256),
            ("sharing", False),
            ("expansion_coefficient_lengths", 3),
        ):
            values = self.hparams[name]
            if values is None:
                values = [default]
            if len(values) == 1:
                values = values * len(stack_types)
            if len(values) != len(stack_types):
                raise ValueError(f"{name} must have length 1 or len(stack_types)")
            if name != "sharing" and any(
                not isinstance(value, int) or value < 1 for value in values
            ):
                raise ValueError(f"{name} must contain positive integers")
            self.hparams[name] = values

        self.embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            embedding_paddings=embedding_paddings,
            categorical_groups=categorical_groups,
            x_categoricals=x_categoricals,
        )
        self.encoder_covariates = [
            name for name in self.encoder_variables if name != target
        ]
        feature_sizes = dict.fromkeys(x_reals, 1)
        feature_sizes.update(self.embeddings.output_size)
        input_size = (
            context_length
            + context_length * sum(feature_sizes[n] for n in self.encoder_covariates)
            + prediction_length * sum(feature_sizes[n] for n in self.decoder_variables)
            + sum(feature_sizes[n] for n in self.static_variables)
        )
        exogenous_size = sum(feature_sizes[n] for n in self.decoder_variables)
        if "exogenous" in stack_types and (
            exogenous_size == 0
            or not set(self.decoder_variables).issubset(self.encoder_covariates)
        ):
            raise ValueError(
                "exogenous stacks require known temporal covariates in both "
                "encoder and decoder"
            )

        self.net_blocks = nn.ModuleList()
        for i, stack_type in enumerate(stack_types):
            for block_id in range(self.hparams.num_blocks[i]):
                if block_id > 0 and self.hparams.sharing[i]:
                    block = self.net_blocks[-1]
                else:
                    block = NBEATSxBlock(
                        stack_type=stack_type,
                        input_size=input_size,
                        context_length=context_length,
                        prediction_length=prediction_length,
                        units=self.hparams.widths[i],
                        num_layers=self.hparams.num_block_layers[i],
                        expansion_coefficient_length=(
                            self.hparams.expansion_coefficient_lengths[i]
                        ),
                        exogenous_size=exogenous_size,
                        dropout=dropout,
                    )
                self.net_blocks.append(block)

    def _covariates(self, x, names, period):
        """Select and embed only the features available in this period."""
        continuous = x[f"{period}_cont"]
        features = []
        for name in names:
            if name in self.hparams.x_reals:
                idx = self.hparams.x_reals.index(name)
                features.append(continuous[..., idx : idx + 1])
            else:
                columns = self.hparams.categorical_groups.get(name)
                if columns is None:
                    idx = self.hparams.x_categoricals.index(name)
                else:
                    idx = [self.hparams.x_categoricals.index(c) for c in columns]
                features.append(self.embeddings[name](x[f"{period}_cat"][..., idx]))
        if not features:
            return continuous[..., :0]
        return torch.cat(features, dim=-1)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute forecasts and normalized block contributions."""
        for period, length in (
            ("encoder", self.hparams.context_length),
            ("decoder", self.hparams.prediction_length),
        ):
            if x[f"{period}_cont"].size(1) != length or torch.any(
                x[f"{period}_lengths"] != length
            ):
                raise ValueError(f"NBeatsX requires fixed {period} length {length}")
        target = x["encoder_cont"][..., self.target_positions].squeeze(-1)
        encoder = self._covariates(x, self.encoder_covariates, "encoder")
        decoder = self._covariates(x, self.decoder_variables, "decoder")
        static = self._covariates(x, self.static_variables, "encoder")[:, 0]
        covariates = torch.cat([encoder.flatten(1), decoder.flatten(1), static], dim=-1)
        encoder_basis = None
        if "exogenous" in self.hparams.stack_types:
            encoder_basis = self._covariates(x, self.decoder_variables, "encoder")

        residual = target
        backcasts, forecasts = [], []
        for block in self.net_blocks:
            backcast, forecast = block(residual, covariates, encoder_basis, decoder)
            residual = residual - backcast
            backcasts.append(backcast)
            forecasts.append(forecast)
        block_forecasts = torch.stack(forecasts, dim=-1)
        return self.to_network_output(
            prediction=self.transform_output(
                block_forecasts.sum(-1).unsqueeze(-1), target_scale=x["target_scale"]
            ),
            backcast=self.transform_output(
                (target - residual).unsqueeze(-1), target_scale=x["target_scale"]
            ),
            block_forecasts=block_forecasts.detach(),
            block_backcasts=torch.stack(backcasts, dim=-1),
        )

    def step(self, x, y, batch_idx, **kwargs):
        """Train the forecast and the normalized backcast reconstruction."""
        log, out = super().step(x, y, batch_idx, **kwargs)
        if not self.predicting and self.hparams.backcast_loss_ratio > 0:
            target = x["encoder_cont"][..., self.target_positions].squeeze(-1)
            residual = target - out.block_backcasts.sum(dim=-1)
            backcast_loss = residual.square().mean()
            log["loss"] = log["loss"] + self.hparams.backcast_loss_ratio * backcast_loss
            self.log(
                f"{self.current_stage}_backcast_loss",
                backcast_loss.detach(),
                on_step=self.training,
                on_epoch=True,
                batch_size=target.size(0),
            )
        return log, out

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        """Infer covariate metadata and fixed window lengths from a dataset."""
        if not isinstance(dataset.target, str) or isinstance(
            dataset.target_normalizer, NaNLabelEncoder
        ):
            raise ValueError("NBeatsX requires a single continuous target")
        if (
            dataset.min_encoder_length != dataset.max_encoder_length
            or dataset.min_prediction_length != dataset.max_prediction_length
            or dataset.randomize_length is not None
        ):
            raise ValueError("NBeatsX requires fixed encoder and prediction lengths")
        for name, value in (
            ("context_length", dataset.max_encoder_length),
            ("prediction_length", dataset.max_prediction_length),
            ("target", dataset.target),
        ):
            if name in kwargs and kwargs[name] != value:
                raise ValueError(f"{name} must match the dataset")
            kwargs[name] = value
        return super().from_dataset(dataset, **kwargs)
