"""
N-HiTS model for timeseries forecasting with covariates.
"""
from copy import copy
from typing import Dict, List, Optional, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss
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
        static_hidden_size: Optional[int] = None,
        shared_weights: bool = True,
        activation: str = "ReLU",
        initialization: str = "lecun_normal",
        n_blocks: List[int] = [1, 1, 1],
        n_layers: Union[int, List[int]] = 2,
        hidden_size: int = 512,
        pooling_sizes: Optional[List[int]] = None,
        downsample_frequencies: Optional[List[int]] = None,
        pooling_mode: str = "max",
        interpolation_mode: str = "linear",
        batch_normalization: bool = False,
        dropout: float = 0.0,
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
        Initialize N-HiTS Model - use its :py:meth:`~from_dataset` method if possible.

        Based on the article
        `N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting <http://arxiv.org/abs/2201.12886>`_.
        The network has shown to increase accuracy by ~25% against
        :py:class:`~pytorch_forecasting.models.nbeats.NBeats` and also supports covariates.

        Args:
            hidden_size (int): size of hidden layers and can range from 8 to 1024 - use 32-128 if no
                covariates are employed. Defaults to 512.
            static_hidden_size (Optional[int], optional): size of hidden layers for static variables.
                Defaults to hidden_size.
            loss: loss to optimize. Defaults to MASE().
            shared_weights (bool, optional): if True, weights of blocks are shared in each stack. Defaults to True.
            initialization (str, optional): Initialization method. One of ['orthogonal', 'he_uniform', 'glorot_uniform',
                'glorot_normal', 'lecun_normal']. Defaults to "lecun_normal".
            n_blocks (List[int], optional): list of blocks used in each stack (i.e. length of stacks).
                Defaults to [1, 1, 1].
            n_layers (Union[int, List[int]], optional): Number of layers per block or list of number of
                layers used by blocks in each stack (i.e. length of stacks). Defaults to 2.
            pooling_sizes (Optional[List[int]], optional): List of pooling sizes for input for each stack,
                i.e. higher means more smoothing of input. Using an ordering of higher to lower in the list
                improves results.
                Defaults to a heuristic.
            pooling_mode (str, optional): Pooling mode for summarizing input. One of ['max','average'].
                Defaults to "max".
            downsample_frequencies (Optional[List[int]], optional): Downsample multiplier of output for each stack, i.e.
                higher means more interpolation at forecast time is required. Should be equal or higher
                than pooling_sizes but smaller equal prediction_length.
                Defaults to a heuristic to match pooling_sizes.
            interpolation_mode (str, optional): Interpolation mode for forecasting. One of ['linear', 'nearest',
                'cubic-x'] where 'x' is replaced by a batch size for the interpolation. Defaults to "linear".
            batch_normalization (bool, optional): Whether carry out batch normalization. Defaults to False.
            dropout (float, optional): dropout rate for hidden layers. Defaults to 0.0.
            activation (str, optional): activation function. One of ['ReLU', 'Softplus', 'Tanh', 'SELU',
                'LeakyReLU', 'PReLU', 'Sigmoid']. Defaults to "ReLU".
            output_size: number of outputs (typically number of quantiles for QuantileLoss and one target or list
                of output sizes but currently only point-forecasts allowed). Set automatically.
            static_categoricals: names of static categorical variables
            static_reals: names of static continuous variables
            time_varying_categoricals_encoder: names of categorical variables for encoder
            time_varying_categoricals_decoder: names of categorical variables for decoder
            time_varying_reals_encoder: names of continuous variables for encoder
            time_varying_reals_decoder: names of continuous variables for decoder
            categorical_groups: dictionary where values
                are list of categorical variables that are forming together a new categorical
                variable which is the key in the dictionary
            x_reals: order of continuous variables in tensor passed to forward function
            x_categoricals: order of categorical variables in tensor passed to forward function
            hidden_continuous_size: default for hidden size for processing continous variables (similar to categorical
                embedding size)
            hidden_continuous_sizes: dictionary mapping continuous input indices to sizes for variable selection
                (fallback to hidden_continuous_size if index is not in dictionary)
            embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
                embedding size
            embedding_paddings: list of indices for embeddings which transform the zero's embedding to a zero vector
            embedding_labels: dictionary mapping (string) indices to list of categorical labels
            learning_rate: learning rate
            log_interval: log predictions every x batches, do not log if 0 or less, log interpretation if > 0. If < 1.0
                , will log multiple entries per batch. Defaults to -1.
            log_val_interval: frequency with which to log validation set metrics, defaults to log_interval
            log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
                failures
            prediction_length: Length of the prediction. Also known as 'horizon'.
            context_length: Number of time units that condition the predictions. Also known as 'lookback period'.
                Should be between 1-10 times the prediction length.
            backcast_loss_ratio: weight of backcast in comparison to forecast when calculating the loss.
                A weight of 1.0 means that forecast and backcast loss is weighted the same (regardless of backcast and
                forecast lengths). Defaults to 0.0, i.e. no weight.
            log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
                failures
            reduce_on_plateau_patience (int): patience after which learning rate is reduced by a factor of 10
            logging_metrics (nn.ModuleList[MultiHorizonMetric]): list of metrics that are logged during training.
                Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
            **kwargs: additional arguments to :py:class:`~BaseModel`.
        """
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        if loss is None:
            loss = MASE()

        if activation == "SELU":
            self.hparams.initialization = "lecun_normal"

        # provide default downsampling sizes
        n_stacks = len(n_blocks)
        if pooling_sizes is None:
            pooling_sizes = np.exp2(np.round(np.linspace(0.49, np.log2(prediction_length / 2), n_stacks)))
            pooling_sizes = [int(x) for x in pooling_sizes[::-1]]
        if downsample_frequencies is None:
            downsample_frequencies = [min(prediction_length, int(np.power(x, 1.5))) for x in pooling_sizes]

        # set static hidden size
        if static_hidden_size is None:
            static_hidden_size = hidden_size

        # set layers
        if isinstance(n_layers, int):
            n_layers = [n_layers] * n_stacks

        self.save_hyperparameters()
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        self.embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
        )
        if isinstance(self.hparams.output_size, int):
            output_size = self.hparams.output_size
        else:
            output_size = sum(self.hparams.output_size)

        self.model = NHiTSModule(
            context_length=self.hparams.context_length,
            prediction_length=self.hparams.prediction_length,
            output_size=output_size,
            static_size=self.static_size,
            covariate_size=self.covariate_size,
            static_hidden_size=self.hparams.static_hidden_size,
            n_blocks=self.hparams.n_blocks,
            n_layers=self.hparams.n_layers,
            hidden_size=self.n_stacks * [2 * [self.hparams.hidden_size]],
            pooling_sizes=self.hparams.pooling_sizes,
            downsample_frequencies=self.hparams.downsample_frequencies,
            pooling_mode=self.hparams.pooling_mode,
            interpolation_mode=self.hparams.interpolation_mode,
            dropout=self.hparams.dropout,
            activation=self.hparams.activation,
            initialization=self.hparams.initialization,
            batch_normalization=self.hparams.batch_normalization,
            shared_weights=self.hparams.shared_weights,
        )

    @property
    def covariate_size(self) -> int:
        """Covariate size.

        Returns:
            int: size of time-dependent covariates
        """
        return len(set(self.hparams.time_varying_reals_decoder) - set(self.target_names)) + sum(
            self.embeddings.output_size[name] for name in self.hparams.time_varying_categoricals_encoder
        )

    @property
    def static_size(self) -> int:
        """Static covariate size.

        Returns:
            int: size of static covariates
        """
        return len(self.hparams.static_reals) + sum(
            self.embeddings.output_size[name] for name in self.hparams.static_categoricals
        )

    @property
    def n_stacks(self) -> int:
        """Number of stacks.

        Returns:
            int: number of stacks.
        """
        return len(self.hparams.n_blocks)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Pass forward of network.

        Args:
            x (Dict[str, torch.Tensor]): input from dataloader generated from
                :py:class:`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`.

        Returns:
            Dict[str, torch.Tensor]: output of model
        """
        # covariates
        if self.covariate_size > 0:
            encoder_features = self.extract_features(x, self.embeddings, period="encoder")
            encoder_x_t = torch.concat(
                [encoder_features[name] for name in self.encoder_variables if name not in self.target_names],
                dim=2,
            )
            decoder_features = self.extract_features(x, self.embeddings, period="decoder")
            decoder_x_t = torch.concat([decoder_features[name] for name in self.decoder_variables], dim=2)
        else:
            encoder_x_t = None
            decoder_x_t = None

        # statics
        if self.static_size > 0:
            x_s = torch.concat([encoder_features[name][:, 0] for name in self.static_variables], dim=1)
        else:
            x_s = None

        # target
        encoder_y = x["encoder_cont"][..., self.target_positions]
        encoder_mask = create_mask(x["encoder_lengths"].max(), x["encoder_lengths"], inverse=True)

        # run model
        forecast, backcast, block_forecasts, block_backcasts = self.model(
            encoder_y, encoder_mask, encoder_x_t, decoder_x_t, x_s
        )

        # create output
        block_predictions = torch.cat([block_backcasts.detach(), block_forecasts.detach()], dim=1)

        if forecast.size(2) > 1:  # multi-output
            n_outputs = forecast.size(2)
            forecast = [forecast[:, :, i] for i in range(n_outputs)]
            backcast = [encoder_y[:, :, i] - backcast[:, :, i] for i in range(n_outputs)]

            n_blocks = block_predictions.size(3)
            block_predictions = [block_predictions[:, :, i] for i in range(n_outputs)]
            block_predictions = tuple(
                self.transform_output([b[..., block] for b in block_predictions], target_scale=x["target_scale"])
                for block in range(n_blocks)
            )
        else:
            block_predictions = tuple(
                self.transform_output(block_predictions[..., i], target_scale=x["target_scale"])
                for i in range(block_predictions[0].size(-1))
            )
            backcast = encoder_y - backcast

        return self.to_network_output(
            prediction=self.transform_output(forecast, target_scale=x["target_scale"]),
            backcast=self.transform_output(backcast, target_scale=x["target_scale"]),
            block_predictions=block_predictions,
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
        # validate arguments
        assert not isinstance(
            dataset.target_normalizer, NaNLabelEncoder
        ), "only regression tasks are supported - target must not be categorical"
        assert (
            dataset.min_encoder_length == dataset.max_encoder_length
        ), "only fixed encoder length is allowed, but min_encoder_length != max_encoder_length"

        assert (
            dataset.max_prediction_length == dataset.min_prediction_length
        ), "only fixed prediction length is allowed, but max_prediction_length != min_prediction_length"

        assert dataset.randomize_length is None, "length has to be fixed, but randomize_length is not None"
        assert not dataset.add_relative_time_idx, "add_relative_time_idx has to be False"

        new_kwargs = copy(kwargs)
        new_kwargs.update(
            {"prediction_length": dataset.max_prediction_length, "context_length": dataset.max_encoder_length}
        )
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, MASE()))

        assert (isinstance(new_kwargs["output_size"], int) and new_kwargs["output_size"] == 1) or all(
            o == 1 for o in new_kwargs["output_size"]
        ), "output sizes can only be of size 1, i.e. point forecasts"

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
            if isinstance(self.loss, (MultiLoss, MASE)):
                backcast_loss = (
                    self.loss(
                        backcast,
                        (x["encoder_target"], None),
                        encoder_target=x["decoder_target"],
                        encoder_lengths=x["decoder_lengths"],
                    )
                    * backcast_weight
                )
            else:
                backcast_loss = self.loss(backcast, x["encoder_target"]) * backcast_weight
            label = ["val", "train"][self.training]
            self.log(
                f"{label}_backcast_loss",
                backcast_loss,
                on_epoch=True,
                on_step=self.training,
                batch_size=len(x["decoder_target"]),
            )
            self.log(
                f"{label}_forecast_loss",
                log["loss"],
                on_epoch=True,
                on_step=self.training,
                batch_size=len(x["decoder_target"]),
            )
            log["loss"] = log["loss"] * forecast_weight + backcast_loss

        # log interpretation
        self.log_interpretation(x, out, batch_idx=batch_idx)
        return log, out

    def plot_interpretation(
        self,
        x: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor],
        idx: int,
        ax=None,
    ) -> plt.Figure:
        """
        Plot interpretation.

        Plot two pannels: prediction and backcast vs actuals and
        decomposition of prediction into different block predictions which capture different frequencies.

        Args:
            x (Dict[str, torch.Tensor]): network input
            output (Dict[str, torch.Tensor]): network output
            idx (int): index of sample for which to plot the interpretation.
            ax (List[matplotlib axes], optional): list of two matplotlib axes onto which to plot the interpretation.
                Defaults to None.

        Returns:
            plt.Figure: matplotlib figure
        """
        if isinstance(x["encoder_target"], (tuple, list)):  # multi-target
            figs = []
            for i in range(len(self.target_names)):
                if ax is not None:
                    ax_i = ax[i]
                else:
                    ax_i = None

                figs.append(
                    self.plot_interpretation(
                        dict(encoder_target=x["encoder_target"][i], decoder_target=x["decoder_target"][i]),
                        dict(
                            backcast=output["backcast"][i],
                            prediction=output["prediction"][i],
                            block_predictions=output["block_predictions"][i],
                        ),
                        idx=idx,
                        ax=ax_i,
                    )
                )
            return figs

        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        else:
            fig = ax[0].get_figure()

        time = torch.arange(-self.hparams.context_length, self.hparams.prediction_length)

        # plot target vs prediction
        ax[0].plot(time, torch.cat([x["encoder_target"][idx], x["decoder_target"][idx]]).detach().cpu(), label="target")
        ax[0].plot(
            time,
            torch.cat(
                [
                    output["backcast"][idx].detach(),
                    output["prediction"][idx].detach(),
                ],
                dim=0,
            ).cpu(),
            label="prediction",
        )
        ax[0].set_xlabel("Time")

        # plot blocks
        prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
        next(prop_cycle)  # prediction
        next(prop_cycle)  # observations

        for pooling_size, block_prediction in zip(self.hparams.pooling_sizes, output["block_predictions"][1:]):
            ax[1].plot(
                time,
                block_prediction[idx].detach().cpu(),
                label=f"Pooling size: {pooling_size}",
                c=next(prop_cycle)["color"],
            )
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Decomposition")

        fig.legend()
        return fig

    def log_interpretation(self, x, out, batch_idx):
        """
        Log interpretation of network predictions in tensorboard.
        """
        label = ["val", "train"][self.training]
        if self.log_interval > 0 and batch_idx % self.log_interval == 0:
            fig = self.plot_interpretation(x, out, idx=0)
            name = f"{label.capitalize()} interpretation of item 0 in "
            if self.training:
                name += f"step {self.global_step}"
            else:
                name += f"batch {batch_idx}"
            self.logger.experiment.add_figure(name, fig, global_step=self.global_step)
            if isinstance(fig, (list, tuple)):
                for idx, f in enumerate(fig):
                    self.logger.experiment.add_figure(
                        f"{self.target_names[idx]} {name}",
                        f,
                        global_step=self.global_step,
                    )
                else:
                    self.logger.experiment.add_figure(
                        name,
                        fig,
                        global_step=self.global_step,
                    )
