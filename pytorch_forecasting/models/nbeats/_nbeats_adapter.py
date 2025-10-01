"""
N-Beats model adapter for timeseries forecasting without covariates.
"""

from typing import Optional

import torch

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.layers._nbeats._blocks import (
    NBEATSSeasonalBlock,
    NBEATSTrendBlock,
)
from pytorch_forecasting.metrics import MASE
from pytorch_forecasting.models.base_model import BaseModel
from pytorch_forecasting.utils._dependencies import _check_matplotlib


class NBeatsAdapter(BaseModel):
    """
    Initialize NBeats Adapter.

    Parameters
    ----------
    **kwargs
        additional arguments to :py:class:`~BaseModel`.
    """  # noqa: E501

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Pass forward of network.

        Parameters
        ----------
        x : dict of str to torch.Tensor
            input from dataloader generated from
            :py:class:`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`.

        Returns
        -------
        dict of str to torch.Tensor
            output of model
        """
        target = x["encoder_cont"][..., 0]

        timesteps = self.hparams.context_length + self.hparams.prediction_length
        generic_forecast = [
            torch.zeros(
                (target.size(0), timesteps), dtype=torch.float32, device=self.device
            )
        ]
        trend_forecast = [
            torch.zeros(
                (target.size(0), timesteps), dtype=torch.float32, device=self.device
            )
        ]
        seasonal_forecast = [
            torch.zeros(
                (target.size(0), timesteps), dtype=torch.float32, device=self.device
            )
        ]
        forecast = torch.zeros(
            (target.size(0), self.hparams.prediction_length),
            dtype=torch.float32,
            device=self.device,
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
            )  # do not use backcast -= backcast_block as this signifies an inline operation # noqa : E501
            forecast = forecast + forecast_block

        return self.to_network_output(
            prediction=self.transform_output(forecast.unsqueeze(-1), target_scale=x["target_scale"]),
            backcast=self.transform_output(
                prediction=(target - backcast).unsqueeze(-1), target_scale=x["target_scale"]
            ),
            trend=self.transform_output(
                torch.stack(trend_forecast, dim=0).sum(0).unsqueeze(-1),
                target_scale=x["target_scale"],
            ),
            seasonality=self.transform_output(
                torch.stack(seasonal_forecast, dim=0).sum(0).unsqueeze(-1),
                target_scale=x["target_scale"],
            ),
            generic=self.transform_output(
                torch.stack(generic_forecast, dim=0).sum(0).unsqueeze(-1),
                target_scale=x["target_scale"],
            ),
        )

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        """
        Convenience function to create network from :py:class
        `~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`.

        Parameters
        ----------
        dataset : TimeSeriesDataSet
            dataset where sole predictor is the target.
        **kwargs
            additional arguments to be passed to ``__init__`` method.

        Returns
        -------
        NBeats
        """  # noqa: E501
        new_kwargs = {
            "prediction_length": dataset.max_prediction_length,
            "context_length": dataset.max_encoder_length,
        }
        new_kwargs.update(kwargs)

        # validate arguments
        assert isinstance(
            dataset.target, str
        ), "only one target is allowed (passed as string to dataset)"
        assert not isinstance(
            dataset.target_normalizer, NaNLabelEncoder
        ), "only regression tasks are supported - target must not be categorical"
        assert dataset.min_encoder_length == dataset.max_encoder_length, (
            "only fixed encoder length is allowed,"
            " but min_encoder_length != max_encoder_length"
        )

        assert dataset.max_prediction_length == dataset.min_prediction_length, (
            "only fixed prediction length is allowed,"
            " but max_prediction_length != min_prediction_length"
        )

        assert (
            dataset.randomize_length is None
        ), "length has to be fixed, but randomize_length is not None"
        assert (
            not dataset.add_relative_time_idx
        ), "add_relative_time_idx has to be False"

        assert (
            len(dataset.flat_categoricals) == 0
            and len(dataset.reals) == 1
            and len(dataset._time_varying_unknown_reals) == 1
            and dataset._time_varying_unknown_reals[0] == dataset.target
        ), (
            "The only variable as input should be the"
            " target which is part of time_varying_unknown_reals"
        )

        # initialize class
        return super().from_dataset(dataset, **new_kwargs)

    def step(self, x, y, batch_idx) -> dict[str, torch.Tensor]:
        """
        Take training / validation step.
        """
        log, out = super().step(x, y, batch_idx=batch_idx)

        if (
            self.hparams.backcast_loss_ratio > 0 and not self.predicting
        ):  # add loss from backcast
            backcast = out["backcast"]
            backcast_weight = (
                self.hparams.backcast_loss_ratio
                * self.hparams.prediction_length
                / self.hparams.context_length
            )
            backcast_weight = backcast_weight / (backcast_weight + 1)  # normalize
            forecast_weight = 1 - backcast_weight
            if isinstance(self.loss, MASE):
                backcast_loss = (
                    self.loss(backcast, x["encoder_target"], x["decoder_target"])
                    * backcast_weight
                )
            else:
                backcast_loss = (
                    self.loss(backcast, x["encoder_target"]) * backcast_weight
                )
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
        x: dict[str, torch.Tensor],
        output: dict[str, torch.Tensor],
        idx: int,
        ax=None,
        plot_seasonality_and_generic_on_secondary_axis: bool = False,
    ):
        """
        Plot interpretation.

        Plot two panels: prediction and backcast vs actuals and
        decomposition of prediction into trend, seasonality and generic forecast.

        Parameters
        ----------
        x : dict of str to torch.Tensor
            network input
        output : dict of str to torch.Tensor
            network output
        idx : int
            index of sample for which to plot the interpretation.
        ax : list of matplotlib.axes
            list of two matplotlib axes onto which to plot the interpretation. Defaults to None.
        plot_seasonality_and_generic_on_secondary_axis : bool
            if to plot seasonality and generic forecast on secondary axis in second panel.
            Defaults to False.

        Returns
        -------
        matplotlib.figure.Figure
            matplotlib figure
        """  # noqa: E501
        _check_matplotlib("plot_interpretation")

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        else:
            fig = ax[0].get_figure()

        time = torch.arange(
            -self.hparams.context_length, self.hparams.prediction_length
        )

        # plot target vs prediction
        ax[0].plot(
            time,
            torch.cat([x["encoder_target"][idx], x["decoder_target"][idx]])
            .detach()
            .cpu(),
            label="target",
        )
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
