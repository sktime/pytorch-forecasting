"""
N-Beats model for timeseries forecasting without covariates.
"""

from typing import Dict, List, Optional

import torch
from torch import nn

from pytorch_forecasting.data import TimeSeriesDataSet

# from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric
from pytorch_forecasting.models.base_model import BaseModel
from pytorch_forecasting.models.nbeats.sub_modules import NBEATSGenericBlock, NBEATSSeasonalBlock, NBEATSTrendBlock
from pytorch_forecasting.utils._dependencies import _check_matplotlib


class NBeats(BaseModel):
    def __init__(
        self,
        stack_types: Optional[List[str]] = None,
        num_blocks: Optional[List[int]] = None,
        num_block_layers: Optional[List[int]] = None,
        widths: Optional[List[int]] = None,
        sharing: Optional[List[bool]] = None,
        expansion_coefficient_lengths: Optional[List[int]] = None,
        prediction_length: int = 1,
        context_length: int = 1,
        use_kan: bool = False,
        num_grids: int = 5,
        k: int = 3,
        noise_scale: float = 0.5,
        scale_base_mu: float = 0.0,
        scale_base_sigma: float = 1.0,
        scale_sp: float = 1.0,
        base_fun: callable = torch.nn.SiLU(),
        grid_eps: float = 0.02,
        grid_range: List[int] = [-1, 1],
        sp_trainable: bool = True,
        sb_trainable: bool = True,
        sparse_init: bool = False,
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

        The :py:class:`~pytorch_forecasting.models.nhits.NHiTS` network has recently shown to consistently outperform
        N-BEATS.

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
            num_grids :  Parameter for KAN layer. the number of grid intervals = G. Default: 5.
            k : Parameter for KAN layer. the order of piecewise polynomial. Default: 3.
            noise_scale : Parameter for KAN layer. the scale of noise injected at initialization. Default: 0.1.
            scale_base_mu : Parameter for KAN layer. the scale of the residual function b(x) is intialized to be
                N(scale_base_mu, scale_base_sigma^2). Deafult: 0.0
            scale_base_sigma : Parameter for KAN layer. the scale of the residual function b(x) is intialized to be
                N(scale_base_mu, scale_base_sigma^2). Deafult: 1.0
            scale_sp : Parameter for KAN layer. the scale of the base function spline(x). Deafult: 1.0
            base_fun : Parameter for KAN layer. residual function b(x). Default: torch.nn.SiLU()
            grid_eps : Parameter for KAN layer. When grid_eps = 1, the grid is uniform; when grid_eps = 0,
                the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the
                two extremes. Deafult: 0.02
            grid_range : Parameter for KAN layer. list/np.array of shape (2,). setting the range of grids.
                Default: [-1,1].
            sp_trainable : Parameter for KAN layer. If true, scale_sp is trainable. Default: True.
            sb_trainable : Parameter for KAN layer. If true, scale_base is trainable. Default: True.
            sparse_init : Parameter for KAN layer. if sparse_init = True, sparse initialization is applied.
                Default: False.
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
        if expansion_coefficient_lengths is None:
            expansion_coefficient_lengths = [3, 7]
        if sharing is None:
            sharing = [True, True]
        if widths is None:
            widths = [32, 512]
        if num_block_layers is None:
            num_block_layers = [3, 3]
        if num_blocks is None:
            num_blocks = [3, 3]
        if stack_types is None:
            stack_types = ["trend", "seasonality"]
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        if loss is None:
            loss = MASE()
        # Bundle KAN parameters into a dictionary
        self.kan_params = {
            "use_kan": use_kan,
            "num_grids": num_grids,
            "k": k,
            "noise_scale": noise_scale,
            "scale_base_mu": scale_base_mu,
            "scale_base_sigma": scale_base_sigma,
            "scale_sp": scale_sp,
            "base_fun": base_fun,
            "grid_eps": grid_eps,
            "grid_range": grid_range,
            "sp_trainable": sp_trainable,
            "sb_trainable": sb_trainable,
            "sparse_init": sparse_init,
        }

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
                        kan_params=self.hparams.kan_params,
                    )
                elif stack_type == "seasonality":
                    net_block = NBEATSSeasonalBlock(
                        units=self.hparams.widths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                        min_period=self.hparams.expansion_coefficient_lengths[stack_id],
                        dropout=self.hparams.dropout,
                        kan_params=self.hparams.kan_params,
                    )
                elif stack_type == "trend":
                    net_block = NBEATSTrendBlock(
                        units=self.hparams.widths[stack_id],
                        thetas_dim=self.hparams.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                        dropout=self.hparams.dropout,
                        kan_params=self.hparams.kan_params,
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

        return self.to_network_output(
            prediction=self.transform_output(forecast, target_scale=x["target_scale"]),
            backcast=self.transform_output(prediction=target - backcast, target_scale=x["target_scale"]),
            trend=self.transform_output(torch.stack(trend_forecast, dim=0).sum(0), target_scale=x["target_scale"]),
            seasonality=self.transform_output(
                torch.stack(seasonal_forecast, dim=0).sum(0), target_scale=x["target_scale"]
            ),
            generic=self.transform_output(torch.stack(generic_forecast, dim=0).sum(0), target_scale=x["target_scale"]),
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
            and len(dataset._time_varying_unknown_reals) == 1
            and dataset._time_varying_unknown_reals[0] == dataset.target
        ), "The only variable as input should be the target which is part of time_varying_unknown_reals"

        # initialize class
        return super().from_dataset(dataset, **new_kwargs)

    def step(self, x, y, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Take training / validation step.
        """
        log, out = super().step(x, y, batch_idx=batch_idx)

        if self.hparams.backcast_loss_ratio > 0 and not self.predicting:  # add loss from backcast
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

        self.log_interpretation(x, out, batch_idx=batch_idx)
        return log, out

    def log_interpretation(self, x, out, batch_idx):
        """
        Log interpretation of network predictions in tensorboard.
        """
        mpl_available = _check_matplotlib("log_interpretation", raise_error=False)

        # Don't log figures if matplotlib or add_figure is not available
        if not mpl_available or not self._logger_supports("add_figure"):
            return None

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
    ):
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
        _check_matplotlib("plot_interpretation")

        import matplotlib.pyplot as plt

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
                    time,
                    output[title][idx].detach().cpu(),
                    label=title.capitalize(),
                    c=next(prop_cycle)["color"],
                )
            else:
                ax2.plot(
                    time,
                    output[title][idx].detach().cpu(),
                    label=title.capitalize(),
                    c=next(prop_cycle)["color"],
                )
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Decomposition")

        fig.legend()
        return fig


# from sktime.datasets import load_airline
# import pandas as pd
# from pytorch_forecasting.data import TimeSeriesDataSet
# import lightning.pytorch as pl
# from lightning.pytorch.callbacks import EarlyStopping

# # Load the airline dataset
# y = load_airline()

# # Convert to DataFrame and reset index for clarity
# df = y.reset_index()

# # Add a 'time_idx' column with values same as the index of rows
# df["time_idx"] = df.index

# # Display the DataFrame
# data = df.drop(columns=["Period"])
# data["series"] = 0
# # data["value"] = data["Number of airline passengers"]+20


# # create dataset and dataloaders
# max_encoder_length = 60
# max_prediction_length = 20

# training_cutoff = data["time_idx"].max() - max_prediction_length

# context_length = max_encoder_length
# prediction_length = max_prediction_length

# training = TimeSeriesDataSet(
#     data[lambda x: x.time_idx <= training_cutoff],
#     time_idx="time_idx",
#     target="Number of airline passengers",
#     categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
#     group_ids=["series"],
#     # only unknown variable is "value" - and N-Beats can also not take any additional variables
#     time_varying_unknown_reals=["Number of airline passengers"],
#     max_encoder_length=context_length,
#     max_prediction_length=prediction_length,
# )
# print("hazrat")
# validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)
# batch_size = 2
# train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
# val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

# pl.seed_everything(42)
# trainer = pl.Trainer(accelerator="auto", gradient_clip_val=0.01)
# net = NBeats.from_dataset(
#     training,
#     learning_rate=1e-3,
#     log_interval=10,
#     log_val_interval=1,
#     weight_decay=1e-2,
#     widths=[32, 512],
#     backcast_loss_ratio=1.0,
# )

# early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
# trainer = pl.Trainer(
#     max_epochs=2,
#     accelerator="auto",
#     enable_model_summary=True,
#     gradient_clip_val=0.1,
#     callbacks=[early_stop_callback],
#     limit_train_batches=150,
# )

# trainer.fit(
#     net,
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader,
# )

# best_model_path = trainer.checkpoint_callback.best_model_path
# best_model = NBeats.load_from_checkpoint(best_model_path)

# # for x, y in iter(val_dataloader):
# #     for y in y:
# #         print(y,type(y))
# # actuals = torch.cat([y for x, y in iter(val_dataloader)]).to("cpu")
# # actuals = [y_tensors[0]  for _, y_tensors in iter(val_dataloader)][0]

# # print(actuals)

# # predictions = best_model.predict(val_dataloader, trainer_kwargs=dict(accelerator="cpu"))
# # print(predictions)
# # predictions_tensor = torch.cat(predictions)
# # actuals_tensor = torch.cat(actuals)

# # # Calculate the absolute error and mean
# # error = (actuals_tensor - predictions_tensor).abs().mean()

# # print(f"Mean absolute error: {error}")
# import matplotlib.pyplot as plt

# raw_predictions = best_model.predict(val_dataloader, mode="raw", return_x=True)

# for idx in range(10):  # plot 10 examples
#     figure = best_model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
#     plt.show()


import warnings

warnings.filterwarnings("ignore")
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import pandas as pd
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data


data = generate_ar_data(seasonality=10.0, timesteps=400, n_series=100, seed=42)
data["static"] = 2
data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
data.head()

# create dataset and dataloaders
max_encoder_length = 60
max_prediction_length = 20

training_cutoff = data["time_idx"].max() - max_prediction_length

context_length = max_encoder_length
prediction_length = max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="value",
    categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
    group_ids=["series"],
    # only unknown variable is "value" - and N-Beats can also not take any additional variables
    time_varying_unknown_reals=["value"],
    max_encoder_length=context_length,
    max_prediction_length=prediction_length,
)

validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

pl.seed_everything(42)
trainer = pl.Trainer(accelerator="auto", gradient_clip_val=0.01)
# net = NBeats.from_dataset(training, learning_rate=3e-2, weight_decay=1e-2, widths=[32, 512], backcast_loss_ratio=0.1)
net = NBeats.from_dataset(
    training,
    learning_rate=1e-3,
    log_interval=10,
    log_val_interval=1,
    weight_decay=1e-2,
    widths=[32, 512],
    backcast_loss_ratio=1.0,
    num_block_layers=[3, 3],
)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
trainer = pl.Trainer(
    max_epochs=1,
    accelerator="auto",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    limit_train_batches=150,
)

trainer.fit(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

best_model_path = trainer.checkpoint_callback.best_model_path
best_model = NBeats.load_from_checkpoint(best_model_path)

raw_predictions = best_model.predict(val_dataloader, mode="raw", return_x=True)
print(best_model)
import matplotlib.pyplot as plt

raw_predictions = best_model.predict(val_dataloader, mode="raw", return_x=True)

for idx in range(10):  # plot 10 examples
    figure = best_model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
    plt.show()
