"""
N-HiTS model for timeseries forecasting with covariates.
"""
from copy import copy
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torchmetrics import MeanSquaredError

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, QuantileLoss
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.nhits.sub_modules import NHiTS as NHiTSModule
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding
from pytorch_forecasting.utils import create_mask


class NHiTS(BaseModelWithCovariates):
    def __init__(
        self,
        output_size: Union[int, List[int]] = 1,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        context_length: int = 1,
        prediction_length: int = 1,
        static_hidden_size: int = 512,
        shared_weights: bool = True,
        activation: str = "ReLU",
        initialization: str = "lecun_normal",
        n_blocks: List[int] = 3 * [1],
        n_layers: List[int] = 9 * [2],
        n_theta_hidden: List[List[int]] = 512,
        n_pool_kernel_size: List[int] = [16, 8, 1],
        n_freq_downsample: List[int] = [60, 8, 1],
        pooling_mode: str = "max",
        interpolation_mode: str = "linear",
        batch_normalization: bool = False,
        dropout_theta: float = 0.0,
        frequency: str = "D",
        seasonality: int = 7,
        learning_rate: float = 1e-2,
        log_interval: int = -1,
        log_gradient_flow: bool = False,
        log_val_interval: int = None,
        weight_decay: float = 1e-3,
        loss: MultiHorizonMetric = None,
        reduce_on_plateau_patience: int = 1000,
        backcast_loss_ratio: float = 0.0,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """
        todo: update
        N-HiTS model.

        Parameters
        ----------
        context_length: int
            Multiplier to get encoder size.
            Insample size = context_length * output_size
        prediction_length: int
            Forecast horizon.
        shared_weights: bool
            If True, repeats first block.
        activation: str
            Activation function.
            An item from ['relu', 'softplus', 'tanh', 'selu', 'lrelu', 'prelu', 'sigmoid'].
        initialization: str
            Initialization function.
            An item from ['orthogonal', 'he_uniform', 'glorot_uniform', 'glorot_normal', 'lecun_normal'].
        n_blocks: List[int]
            Number of blocks for each stack type.
            Note that len(n_blocks) = len(stack_types).
        n_layers: List[int]
            Number of layers for each stack type.
            Note that len(n_layers) = len(stack_types).
        n_theta_hidden: List[List[int]]
            Structure of hidden layers for each stack type.
            Each internal list should contain the number of units of each hidden layer.
            Note that len(n_theta_hidden) = len(stack_types).
        n_pool_kernel_size List[int]:
            Pooling size for input for each stack.
            Note that len(n_pool_kernel_size) = len(stack_types).
        n_freq_downsample List[int]:
            Downsample multiplier of output for each stack.
            Note that len(n_freq_downsample) = len(stack_types).
        batch_normalization: bool
            Whether perform batch normalization.
        dropout_theta: float
            Float between (0, 1).
            Dropout for Nbeats basis.
        learning_rate: float
            Learning rate between (0, 1).
        lr_decay: float
            Decreasing multiplier for the learning rate.
        lr_decay_step_size: int
            Steps between each lerning rate decay.
        weight_decay: float
            L2 penalty for optimizer.
        loss_train: str
            Loss to optimize.
            An item from ['MAPE', 'MASE', 'SMAPE', 'MSE', 'MAE', 'PINBALL', 'PINBALL2'].
        loss_hypar:
            Hyperparameter for chosen loss.
        loss_valid:
            Validation loss.
            An item from ['MAPE', 'MASE', 'SMAPE', 'RMSE', 'MAE', 'PINBALL'].
        frequency: str
            Time series frequency.
        random_seed: int
            random_seed for pseudo random pytorch initializer and
            numpy random generator.
        seasonality: int
            Time series seasonality.
            Usually 7 for daily data, 12 for monthly data and 4 for weekly data.
        """
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        if loss is None:
            loss = MASE()

        if activation == "SELU":
            self.hparams.initialization = "lecun_normal"

        self.save_hyperparameters()
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        self.embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
        )

        # todo: support multi-output
        self.model = NHiTSModule(
            context_length=self.hparams.context_length,
            prediction_length=self.hparams.prediction_length,
            output_size=self.hparams.output_size,
            static_size=len(self.hparams.static_reals)
            + sum(self.embeddings.output_size[name] for name in self.hparams.static_categoricals),
            covariate_size=len(set(self.hparams.time_varying_reals_decoder) - set(self.target_names))
            + sum(self.embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_encoder),
            static_hidden_size=self.hparams.static_hidden_size,
            n_blocks=self.hparams.n_blocks,
            n_layers=self.hparams.n_layers,
            n_theta_hidden=self.n_stacks * [2 * [self.hparams.n_theta_hidden]],
            n_pool_kernel_size=self.hparams.n_pool_kernel_size,
            n_freq_downsample=self.hparams.n_freq_downsample,
            pooling_mode=self.hparams.pooling_mode,
            interpolation_mode=self.hparams.interpolation_mode,
            dropout_theta=self.hparams.dropout_theta,
            activation=self.hparams.activation,
            initialization=self.hparams.initialization,
            batch_normalization=self.hparams.batch_normalization,
            shared_weights=self.hparams.shared_weights,
        )

    @property
    def n_stacks(self) -> int:
        return len(self.hparams.n_blocks)

    def forward(self, x):
        encoder_features = self.extract_features(x, self.embeddings, period="encoder")
        encoder_x_t = torch.concat(
            [encoder_features[name] for name in self.encoder_variables if name not in self.target_names],
            dim=2,
        )
        encoder_y = torch.concat(
            [encoder_features[name] for name in self.target_names],
            dim=2,
        )
        x_s = torch.concat([encoder_features[name][:, 0] for name in self.static_variables], dim=1)
        decoder_features = self.extract_features(x, self.embeddings, period="decoder")
        decoder_x_t = torch.concat([decoder_features[name] for name in self.decoder_variables], dim=2)

        encoder_mask = create_mask(x["encoder_lengths"].max(), x["encoder_lengths"], inverse=True)

        forecast, backcast, block_forecasts, block_backcasts = self.model(
            encoder_y, encoder_mask, encoder_x_t, decoder_x_t, x_s
        )
        block_predictions = torch.cat([block_backcasts.detach(), block_forecasts.detach()], dim=1)
        encoder_target = x["encoder_cont"][..., self.target_positions]
        return self.to_network_output(
            prediction=self.transform_output(forecast, target_scale=x["target_scale"]),
            backcast=self.transform_output(prediction=encoder_target - backcast, target_scale=x["target_scale"]),
            block_predictions=tuple(
                self.transform_output(block_predictions[..., i], target_scale=x["target_scale"])
                for i in range(block_predictions.size(-1))
            ),
        )

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        """
        Convenience function to create network from :py:class`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`.

        Args:
            dataset (TimeSeriesDataSet): dataset where sole predictor is the target.
            **kwargs: additional arguments to be passed to ``__init__`` method.

        Returns:
            NBeats
        """
        # todo: assert same variables in encoder and decoder
        new_kwargs = copy(kwargs)
        new_kwargs = {"prediction_length": dataset.max_prediction_length, "context_length": dataset.max_encoder_length}
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, MeanSquaredError()))

        # initialize class
        return super().from_dataset(dataset, **new_kwargs)

    def step(self, x, y, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Take training / validation step.
        """
        log, out = super().step(x, y, batch_idx=batch_idx)

        if self.hparams.backcast_loss_ratio > 0:  # add loss from backcast
            backcast = out["backcast"]
            backcast_weight = (
                self.hparams.backcast_loss_ratio * self.hparams.prediction_length / self.hparams.context_length
            )
            backcast_weight = backcast_weight / (backcast_weight + 1)  # normalize
            forecast_weight = 1 - backcast_weight
            if isinstance(self.loss, MASE):
                backcast_loss = self.loss(backcast, x["encoder_target"], x["decoder_target"]) * backcast_weight
            else:
                backcast_loss = self.loss(backcast, x["encoder_target"]) * backcast_weight
            label = ["val", "train"][self.training]
            self.log(
                f"{label}_backcast_loss",
                backcast_loss,
                on_epoch=True,
                ostatic_sizetep=self.training,
                batch_size=len(x["decoder_target"]),
            )
            self.log(
                f"{label}_forecast_loss",
                log["loss"],
                on_epoch=True,
                ostatic_sizetep=self.training,
                batch_size=len(x["decoder_target"]),
            )
            log["loss"] = log["loss"] * forecast_weight + backcast_loss

        # self.log_interpretation(x, out, batch_idx=batch_idx)
        return log, out

    # todo: implement interpretation
