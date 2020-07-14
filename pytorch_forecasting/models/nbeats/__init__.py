from typing import Dict, List
import numpy as np
import matplotlib as plt

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.models.base_model import BaseModel
from pytorch_forecasting.models.nbeats.sub_modules import NBEATSTrendBlock, NBEATSGenericBlock, NBEATSSeasonalBlock


class NBeats(BaseModel):
    def __init__(
        self,
        stack_types: List[str] = ["T", "S", "G"],
        num_blocks=[3, 3, 1],
        num_block_layers=[4, 4, 4],
        widths=[256, 2048, 512],
        sharing: List[int] = [True, True, False],
        expansion_coefficient_lengths: List[int] = [4, 4, 32],
        prediction_length: int = 1,
        context_length: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-2,
        log_interval: int = -1,
        log_val_interval: int = None,
        loss=SMAPE(),
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
                (seasonal) then its not used. 
                A list of ints of length 1 or ‘num_stacks’. Default value for generic mode: [32] Recommended value for 
                interpretable mode: [3]
            prediction_length: Length of the prediction. Also known as 'horizon'.
            context_length: Number of time units that condition the predictions. Also known as 'lookback period'.
            has_backcast: Only the last block of the network doesn't.
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
        timesteps = self.hparams.context_length + self.hparams.prediction_length
        generic_forecast = [torch.zeros((target.size(0), timesteps), dtype=torch.float32, device=self.device)]
        trend_forecast = [torch.zeros((target.size(0), timesteps), dtype=torch.float32, device=self.device)]
        seasonal_forecast = [torch.zeros((target.size(0), timesteps), dtype=torch.float32, device=self.device)]

        backcast = target  # initialize backcast
        forecast = torch.zeros(
            (target.size(0), self.hparams.prediction_length), dtype=torch.float32, device=self.device
        )
        for block in self.net_blocks:
            # evaluate block
            backcast_block, forecast_block = block(backcast)

            # add for interpretation
            full = torch.cat([backcast_block, forecast_block], dim=1)
            if isinstance(block, NBEATSTrendBlock):
                trend_forecast.append(full)
            elif isinstance(block, NBEATSSeasonalBlock):
                seasonal_forecast.append(full)
            else:
                generic_forecast.append(full)

            # update backcast and forecast
            backcast -= backcast_block
            forecast += forecast_block

        return dict(
            prediction=forecast,
            backcast=backcast,
            trend=torch.stack(trend_forecast, dim=0).sum(0),
            seasonality=torch.stack(seasonal_forecast, dim=0).sum(0),
            generic=torch.stack(generic_forecast, dim=0).sum(0),
        )

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
        return cls(**new_kwargs)

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
                {name: value[0] for name, value in x.items()}, {name: value[0].detach() for name, value in out.items()},
            )
            self.logger.experiment.add_figure(f"{label.capitalize()} interpretation", fig, global_step=self.global_step)

    def plot_interpretation(self, x, output):
        fig, ax = plt.subplots(4, 1, figsize=(15, 4))

        x = torch.arange(-self.hparams.context_length, self.hparams.prediction_length)

        # plot target vs prediction
        ax[0].plot(x, torch.cat([x["decoder_target"], x["encoder_target"]], label="target"))
        ax[0].plot(x, torch.cat([x["decoder_target"] - output["backcast"], output["prediction"]]), label="prediction")
        ax[0].set_xlabel("Time")

        # plot blocks
        for idx, title in enumerate(["trend", "seasonality", "generic"]):
            ax[idx + 1].plot(x, output[title])
            ax[idx + 1].set_xlabel("Time")
            ax[idx + 1].set_ylabel(title.capitalize())
        fig.legend()

        return fig

