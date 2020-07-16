from pytorch_forecasting.metrics import SMAPE
from typing import Dict
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.metric import TensorMetric
from pytorch_ranger import Ranger
import torch
import numpy as np

import matplotlib.pyplot as plt


class BaseModel(LightningModule):
    """
    BaseModel

    
   `` __init__``

    

    ``forward()`` method should

    Returns:
        Dict[str]: at least should contain ``prediction`` key

    """

    def __init__(
        self,
        log_interval=-1,
        log_val_interval: int = None,
        learning_rate: float = 1e-3,
        log_gradient_flow: bool = False,
        loss: TensorMetric = SMAPE(),
    ):
        """
        BaseModel for timeseries forecasting from which to inherit from

        Args:
            log_interval (int, optional): batches after which predictions are logged. Defaults to -1.
            log_val_interval (int, optional): batches after which predictions for validation are logged. Defaults to None/log_interval.
            learning_rate (float, optional): Learning rate. Defaults to 1e-3.
            log_gradient_flow (bool): If to log gradient flow, this takes time and should be only done to diagnose training
                failures. Defaults to False.
            loss (TensorMetric, optional): metric to optimize. Defaults to SMAPE().
        """
        super().__init__()
        if self.hparams.log_val_interval is None:
            self.hparams.log_val_interval = self.hparams.log_interval

    def size(self) -> int:
        """
        get number of parameters in model
        """
        return sum(p.numel() for p in self.parameters())

    def _step(self, x: Dict[str, torch.Tensor], y: torch.Tensor, batch_idx: int, label="train"):
        out = self(x)
        y_hat = out["prediction"]
        # calculate loss and log it
        loss = self.loss(y_hat, y)
        tensorboard_logs = {f"{label}_loss": loss}
        log = {
            f"{label}_loss": loss,
            "log": tensorboard_logs,
        }
        if label == "train":
            log["loss"] = loss
        if self.log_interval(label == "train") > 0:
            self._log_prediction(x, y_hat, batch_idx, label=label)
        return log, out

    def log_interval(self, train: bool):
        """
        Log interval depending if training or validating
        """
        if train:
            return self.hparams.log_interval
        else:
            return self.hparams.log_val_interval

    def _log_prediction(self, x, y_hat, batch_idx, label="train"):
        # log single prediction figure
        if batch_idx % self.log_interval(label == "train") == 0 and self.log_interval(label == "train") > 0:
            y_all = torch.cat(
                [x["encoder_target"][0], x["decoder_target"][0]]
            )  # all true values for y of the first sample in batch
            if y_all.ndim == 2:  # timesteps, (target, weight), i.e. weight is included
                y_all = y_all[:, 0]
            max_encoder_length = x["encoder_lengths"].max()
            fig = self.plot_prediction(
                torch.cat(
                    (
                        y_all[: x["encoder_lengths"][0]],
                        y_all[max_encoder_length : (max_encoder_length + x["decoder_lengths"][0])],
                    ),
                ),
                y_hat[0, : x["decoder_lengths"][0]].detach(),
            )  # first in batch
            tag = f"{label.capitalize()} prediction"
            if label == "train":
                tag += f" of item 0 in global batch {self.global_step}"
            else:
                tag += f" of item 0 in batch {batch_idx}"
            self.logger.experiment.add_figure(
                tag, fig, global_step=self.global_step,
            )

    def plot_prediction(self, y: torch.Tensor, y_hat: torch.Tensor) -> plt.Figure:
        """
        plot prediction of prediction vs actuals

        Args:
            y: all actual values
            y_hat: predictions

        Returns:
            matplotlib figure
        """
        # move to cpu
        y = y.detach().cpu()
        y_hat = y_hat.cpu()
        # create figure
        fig, ax = plt.subplots()
        n_pred = y_hat.shape[0]
        x_obs = np.arange(y.shape[0] - n_pred)
        x_pred = np.arange(y.shape[0] - n_pred, y.shape[0])
        prop_cycle = iter(plt.rcParams["axes.prop_cycle"])
        obs_color = next(prop_cycle)["color"]
        pred_color = next(prop_cycle)["color"]
        # plot observed history
        if len(x_obs) > 0:
            if len(x_obs) > 1:
                plotter = ax.plot
            else:
                plotter = ax.scatter
            plotter(x_obs, y[:-n_pred], label="observed", c=obs_color)
        if len(x_pred) > 1:
            plotter = ax.plot
        else:
            plotter = ax.scatter
        # plot observed prediction
        plotter(x_pred, y[-n_pred:], label=None, c=obs_color)

        # plot prediction
        plotter(x_pred, self.loss.to_prediction(y_hat), label=f"predicted", c=pred_color)

        # plot predicted quantiles
        y_quantiles = self.loss.to_quantiles(y_hat)
        plotter(x_pred, y_quantiles[:, y_quantiles.shape[1] // 2], c=pred_color, alpha=0.15)
        for i in range(y_quantiles.shape[1] // 2):
            if len(x_pred) > 1:
                ax.fill_between(x_pred, y_quantiles[:, i], y_quantiles[:, -i - 1], alpha=0.15, fc=pred_color)
            else:
                ax.errorbar(
                    x_pred, torch.tensor([[y_quantiles[0, i]], [y_quantiles[0, -i - 1]]]), c=pred_color, capsize=1.0,
                )
        loss = self.loss(y_hat[None], y[-n_pred:][None])
        ax.set_title(f"Loss {loss:.3g}")
        ax.set_xlabel("Time index")
        fig.legend()
        return fig

    def _log_gradient_flow(self, named_parameters):
        """
        log distribution of gradients to identify exploding / vanishing gradients
        """
        ave_grads = []
        layers = []
        for name, p in named_parameters:
            if p.grad is not None and p.requires_grad and "bias" not in name:
                layers.append(name)
                ave_grads.append(p.grad.abs().mean())
                self.logger.experiment.add_histogram(tag=name, values=p.grad, global_step=self.global_step)
        fig, ax = plt.subplots()
        ax.plot(ave_grads)
        ax.set_xlabel("Layers")
        ax.set_ylabel("Average gradient")
        ax.set_yscale("log")
        ax.set_title("Gradient flow")
        self.logger.experiment.add_figure(f"Gradient flow", fig, global_step=self.global_step)

    def on_after_backward(self):
        if (
            self.global_step % self.hparams.log_interval == 0
            and self.hparams.log_interval > 0
            and self.hparams.log_gradient_flow
        ):
            self._log_gradient_flow(self.named_parameters())

    def configure_optimizers(self):
        return Ranger(self.parameters(), lr=self.hparams.learning_rate)
