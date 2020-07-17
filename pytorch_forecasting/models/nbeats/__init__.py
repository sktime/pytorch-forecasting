from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from pytorch_ranger import Ranger

import torch
from torch import nn

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.models.base_model import BaseModel
from pytorch_forecasting.models.nbeats.sub_modules import NBEATSTrendBlock, NBEATSGenericBlock, NBEATSSeasonalBlock


class NBeats(BaseModel):
    def __init__(
        self,
        stack_types: List[str] = ["T", "S", "G"],
        num_blocks=[3, 3, 1],
        num_block_layers=[3, 3, 3],
        widths=[16, 128, 16],
        sharing: List[int] = [True, True, False],
        expansion_coefficient_lengths: List[int] = [5, 7, 4],
        prediction_length: int = 1,
        context_length: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-2,
        log_interval: int = -1,
        log_gradient_flow: bool = False,
        log_val_interval: int = None,
        weight_decay: float = 1e-3,
        loss=SMAPE(log_space=False),
    ):
        """
        Initialize NBeats Model

        Args:
            stack_types: One of the following values: “G” (generic), “S” (seasonal) or “T” (trend). A list of strings 
                of length 1 or ‘num_stacks’. Default and recommended value 
                for generic mode: [“G”] Recommended value for interpretable mode: [“T”,”S”]
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
            has_backcast: Only the last block of the network doesn't.
            log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
                failures
        """
        self.save_hyperparameters()
        super().__init__()
        self.loss = loss

        # setup stacks
        self.net_blocks = nn.ModuleList()
        for stack_id, stack_type in enumerate(stack_types):
            for _ in range(num_blocks[stack_id]):
                if stack_type == "G":
                    net_block = NBEATSGenericBlock(
                        units=self.hparams.widths[stack_id],
                        thetas_dim=self.hparams.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                        dropout=self.hparams.dropout,
                    )
                elif stack_type == "S":
                    net_block = NBEATSSeasonalBlock(
                        units=self.hparams.widths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                        min_period=self.hparams.expansion_coefficient_lengths[stack_id],
                        dropout=self.hparams.dropout,
                    )
                elif stack_type == "T":
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

    def forward(self, x: Dict[str, torch.Tensor]):
        target = x["encoder_target"]

        if self.loss.log_space:
            target = torch.log(target + 1e-8)

        timesteps = self.hparams.context_length + self.hparams.prediction_length
        generic_forecast = [torch.zeros((target.size(0), timesteps), dtype=torch.float32, device=self.device)]
        trend_forecast = [torch.zeros((target.size(0), timesteps), dtype=torch.float32, device=self.device)]
        seasonal_forecast = [torch.zeros((target.size(0), timesteps), dtype=torch.float32, device=self.device)]

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
            if i == 0:
                forecast = forecast_block
            else:
                forecast = forecast + forecast_block

        return dict(
            prediction=forecast,
            backcast=backcast,
            trend=torch.stack(trend_forecast, dim=0).sum(0),
            seasonality=torch.stack(seasonal_forecast, dim=0).sum(0),
            generic=torch.stack(generic_forecast, dim=0).sum(0),
        )

    def configure_optimizers(self):
        return Ranger(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        new_kwargs = {"prediction_length": dataset.max_prediction_length, "context_length": dataset.max_encoder_length}
        new_kwargs.update(kwargs)

        # validate arguments
        assert (
            dataset.min_encoder_length == dataset.max_encoder_length
        ), "only fixed encoder length is allowed, but min_encoder_length != max_encoder_length"

        assert (
            dataset.max_prediction_length == dataset.min_prediction_length
        ), "only fixed prediction length is allowed, but max_prediction_length != min_prediction_length"

        assert dataset.randomize_length is None, "length has to be fixed, but randomize_length is not None"
        assert not dataset.add_relative_time_idx, "add_relative_time_idx has to be False"

        # initialize class
        net = cls(**new_kwargs)
        net.set_dataset_parameters(dataset)
        return net

    def training_step(self, batch, batch_idx):
        x, y = batch
        log, out = self._step(x, y, batch_idx=batch_idx, label="train")
        self._log_interpretation(x, out, batch_idx=batch_idx, label="train")
        return log

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        x, y = batch
        log, out = self._step(x, y, batch_idx=batch_idx, label="val")
        self._log_interpretation(x, out, batch_idx=batch_idx, label="val")
        return log

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def _log_interpretation(self, x, out, batch_idx, label="train"):
        if self.log_interval(label == "train") > 0 and batch_idx % self.log_interval(label == "train") == 0:
            fig = self.plot_interpretation(
                {name: value[0] for name, value in x.items()}, {name: value[0] for name, value in out.items()},
            )
            name = f"{label.capitalize()} interpretation of item 0 in "
            if label == "train":
                name += f"step {self.global_step}"
            else:
                name += f"batch {batch_idx}"
            self.logger.experiment.add_figure(name, fig, global_step=self.global_step)

    def plot_interpretation(self, x, output):
        fig, ax = plt.subplots(2, 1, figsize=(6, 8))

        time = torch.arange(-self.hparams.context_length, self.hparams.prediction_length)

        # plot target vs prediction
        ax[0].plot(time, torch.cat([x["encoder_target"], x["decoder_target"]]), label="target")
        ax[0].plot(
            time,
            torch.cat(
                [
                    x["encoder_target"] - self.loss.to_prediction(output["backcast"].detach()),
                    self.loss.to_prediction(output["prediction"].detach()),
                ],
                dim=0,
            ),
            label="prediction",
        )
        ax[0].set_xlabel("Time")

        # plot blocks
        if self.loss.log_space:
            ax2 = ax[1].twinx()
            ax2.set_ylabel("Seasonality / Generic")
        else:
            ax2 = ax[1]
        for title in ["trend", "seasonality", "generic"]:
            if title == "trend":
                ax[1].plot(time, self.loss.to_prediction(output[title]), label=title.capitalize())
            else:
                ax2.plot(time, self.loss.to_prediction(output[title]), label=title.capitalize())
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Trend")

        fig.legend()
        return fig

