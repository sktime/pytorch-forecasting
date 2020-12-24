"""
N-Beats model for timeseries forecasting without covariates.
"""
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torch import nn

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric
from pytorch_forecasting.models.base_model import BaseModel
from pytorch_forecasting.models.nbeats.sub_modules import NBEATSGenericBlock, NBEATSSeasonalBlock, NBEATSTrendBlock


class NBeats(BaseModel):
    def __init__(
        self,
        stack_types: List[str] = ["trend", "seasonality"],
        num_blocks=[3, 3],
        num_block_layers=[3, 3],
        widths=[32, 512],
        sharing: List[int] = [True, True],
        expansion_coefficient_lengths: List[int] = [3, 7],
        prediction_length: int = 1,
        context_length: int = 1,
        dropout: float = 0.1,
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
        Initialize NBeats Model - use its :py:meth:`~from_dataset` method if possible.

        Based on the article
        `N-BEATS: Neural basis expansion analysis for interpretable time series
        forecasting <http://arxiv.org/abs/1905.10437>`_. The network has (if used as ensemble) outperformed all
        other methods
        including ensembles of traditional statical methods in the M4 competition. The M4 competition is arguably
        the most
        important benchmark for univariate time series forecasting.

        Args:
            stack_types: One of the following values: “generic”, “seasonality" or “trend". A list of strings
                of length 1 or ‘num_stacks’. Default and recommended value
                for generic mode: [“generic”] Recommended value for interpretable mode: [“trend”,”seasonality”]
            num_blocks: The number of blocks per stack. A list of ints of length 1 or ‘num_stacks’.
                Default and recommended value for generic mode: [1] Recommended value for interpretable mode: [3]
            num_block_layers: Number of fully connected layers with ReLu activation per block. A list of ints of length
                1 or ‘num_stacks’.
                Default and recommended value for generic mode: [4] Recommended value for interpretable mode: [4]
            width: Widths of the fully connected layers with ReLu activation in the blocks.
                A list of ints of length 1 or ‘num_stacks’. Default and recommended value for generic mode: [512]
                Recommended value for interpretable mode: [256, 2048]
            sharing: Whether the weights are shared with the other blocks per stack.
                A list of ints of length 1 or ‘num_stacks’. Default and recommended value for generic mode: [False]
                Recommended value for interpretable mode: [True]
            expansion_coefficient_length: If the type is “G” (generic), then the length of the expansion
                coefficient.
                If type is “T” (trend), then it corresponds to the degree of the polynomial. If the type is “S”
                (seasonal) then this is the minimum period allowed, e.g. 2 for changes every timestep.
                A list of ints of length 1 or ‘num_stacks’. Default value for generic mode: [32] Recommended value for
                interpretable mode: [3]
            prediction_length: Length of the prediction. Also known as 'horizon'.
            context_length: Number of time units that condition the predictions. Also known as 'lookback period'.
                Should be between 1-10 times the prediction length.
            backcast_loss_ratio: weight of backcast in comparison to forecast when calculating the loss.
                A weight of 1.0 means that forecast and backcast loss is weighted the same (regardless of backcast and
                forecast lengths). Defaults to 0.0, i.e. no weight.
            loss: loss to optimize. Defaults to MASE().
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
        self.save_hyperparameters()
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        # setup stacks
        self.net_blocks = nn.ModuleList()
        for stack_id, stack_type in enumerate(stack_types):
            for _ in range(num_blocks[stack_id]):
                if stack_type == "generic":
                    net_block = NBEATSGenericBlock(
                        units=self.hparams.widths[stack_id],
                        thetas_dim=self.hparams.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                        dropout=self.hparams.dropout,
                    )
                elif stack_type == "seasonality":
                    net_block = NBEATSSeasonalBlock(
                        units=self.hparams.widths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                        min_period=self.hparams.expansion_coefficient_lengths[stack_id],
                        dropout=self.hparams.dropout,
                    )
                elif stack_type == "trend":
                    net_block = NBEATSTrendBlock(
                        units=self.hparams.widths[stack_id],
                        thetas_dim=self.hparams.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                        dropout=self.hparams.dropout,
                    )
                else:
                    raise ValueError(f"Unknown stack type {stack_type}")

                self.net_blocks.append(net_block)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Pass forward of network.

        Args:
            x (Dict[str, torch.Tensor]): input from dataloader generated from
                :py:class:`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`.

        Returns:
            Dict[str, torch.Tensor]: output of model
        """
        target = x["encoder_cont"][..., 0]

        timesteps = self.hparams.context_length + self.hparams.prediction_length
        generic_forecast = [torch.zeros((target.size(0), timesteps), dtype=torch.float32, device=self.device)]
        trend_forecast = [torch.zeros((target.size(0), timesteps), dtype=torch.float32, device=self.device)]
        seasonal_forecast = [torch.zeros((target.size(0), timesteps), dtype=torch.float32, device=self.device)]
        forecast = torch.zeros(
            (target.size(0), self.hparams.prediction_length), dtype=torch.float32, device=self.device
        )

        backcast = target  # initialize backcast
        for i, block in enumerate(self.net_blocks):
            # evaluate block
            backcast_block, forecast_block = block(backcast)

            # add for interpretation
            full = torch.cat([backcast_block.detach(), forecast_block.detach()], dim=1)
            if isinstance(block, NBEATSTrendBlock):
                trend_forecast.append(full)
            elif isinstance(block, NBEATSSeasonalBlock):
                seasonal_forecast.append(full)
            else:
                generic_forecast.append(full)

            # update backcast and forecast
            backcast = (
                backcast - backcast_block
            )  # do not use backcast -= backcast_block as this signifies an inline operation
            forecast = forecast + forecast_block

        return dict(
            prediction=forecast,
            target_scale=x["target_scale"],
            backcast=target - backcast,
            trend=torch.stack(trend_forecast, dim=0).sum(0),
            seasonality=torch.stack(seasonal_forecast, dim=0).sum(0),
            generic=torch.stack(generic_forecast, dim=0).sum(0),
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
        new_kwargs = {"prediction_length": dataset.max_prediction_length, "context_length": dataset.max_encoder_length}
        new_kwargs.update(kwargs)

        # validate arguments
        assert isinstance(dataset.target, str), "only one target is allowed (passed as string to dataset)"
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

        assert (
            len(dataset.flat_categoricals) == 0
            and len(dataset.reals) == 1
            and len(dataset.time_varying_unknown_reals) == 1
            and dataset.time_varying_unknown_reals[0] == dataset.target
        ), "The only variable as input should be the target which is part of time_varying_unknown_reals"

        # initialize class
        return super().from_dataset(dataset, **new_kwargs)

    def step(self, x, y, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Take training / validation step.
        """
        log, out = super().step(x, y, batch_idx=batch_idx)

        if self.hparams.backcast_loss_ratio > 0:  # add loss from backcast
            backcast = self.transform_output(dict(prediction=out["backcast"], target_scale=out["target_scale"]))
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
            self.log(f"{label}_backcast_loss", backcast_loss, on_epoch=True, on_step=self.training)
            self.log(f"{label}_forecast_loss", log["loss"], on_epoch=True, on_step=self.training)
            log["loss"] = log["loss"] * forecast_weight + backcast_loss

        self._log_interpretation(x, out, batch_idx=batch_idx)
        return log, out

    def _log_interpretation(self, x, out, batch_idx):
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

    def plot_interpretation(
        self,
        x: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor],
        idx: int,
        ax=None,
        plot_seasonality_and_generic_on_secondary_axis: bool = False,
    ) -> plt.Figure:
        """
        Plot interpretation.

        Plot two pannels: prediction and backcast vs actuals and
        decomposition of prediction into trend, seasonality and generic forecast.

        Args:
            x (Dict[str, torch.Tensor]): network input
            output (Dict[str, torch.Tensor]): network output
            idx (int): index of sample for which to plot the interpretation.
            ax (List[matplotlib axes], optional): list of two matplotlib axes onto which to plot the interpretation.
                Defaults to None.
            plot_seasonality_and_generic_on_secondary_axis (bool, optional): if to plot seasonality and
                generic forecast on secondary axis in second panel. Defaults to False.

        Returns:
            plt.Figure: matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        else:
            fig = ax.get_figure()

        time = torch.arange(-self.hparams.context_length, self.hparams.prediction_length)

        def to_prediction(y):
            return self.transform_output(dict(prediction=y[[idx]], target_scale=x["target_scale"][[idx]]))[0]

        # plot target vs prediction
        ax[0].plot(time, torch.cat([x["encoder_target"][idx], x["decoder_target"][idx]]).cpu(), label="target")
        ax[0].plot(
            time,
            torch.cat(
                [
                    to_prediction(output["backcast"].detach()),
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
        if plot_seasonality_and_generic_on_secondary_axis:
            ax2 = ax[1].twinx()
            ax2.set_ylabel("Seasonality / Generic")
        else:
            ax2 = ax[1]
        for title in ["trend", "seasonality", "generic"]:
            if title not in self.hparams.stack_types:
                continue
            if title == "trend":
                ax[1].plot(
                    time, to_prediction(output[title]).cpu(), label=title.capitalize(), c=next(prop_cycle)["color"]
                )
            else:
                ax2.plot(
                    time, to_prediction(output[title]).cpu(), label=title.capitalize(), c=next(prop_cycle)["color"]
                )
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Decomposition")

        fig.legend()
        return fig
