"""
Timeseries models share a number of common characteristics. This module implements these in a common base class.
"""
from copy import deepcopy
import inspect
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.parsing import get_init_args
import scipy.stats
import torch
import torch.nn as nn
from torch.nn.utils import rnn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer, GroupNormalizer, MultiNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import MASE, SMAPE, DistributionLoss, Metric, MultiLoss
from pytorch_forecasting.optim import Ranger
from pytorch_forecasting.utils import apply_to_list, create_mask, get_embedding_size, groupby_apply, to_list


class BaseModel(LightningModule):
    """
    BaseModel from which new timeseries models should inherit from.
    The ``hparams`` of the created object will default to the parameters indicated in :py:meth:`~__init__`.

    The :py:meth:`~BaseModel.forward` method should return a dictionary with at least the entry ``prediction`` and
    ``target_scale`` that contains the network's output. See the function's documentation for more details.

    The idea of the base model is that common methods do not have to be re-implemented for every new architecture.
    The class is a [LightningModule](https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html)
    and follows its conventions. However, there are important additions:

        * You need to specify a ``loss`` attribute that stores the function to calculate the
          :py:class:`~pytorch_forecasting.metrics.MultiHorizonLoss` for backpropagation.
        * The :py:meth:`~BaseModel.from_dataset` method can be used to initialize a network using the specifications
          of a dataset. Often, parameters such as the number of features can be easily deduced from the dataset.
          Further, the method will also store how to rescale normalized predictions into the unnormalized prediction
          space. Override it to pass additional arguments to the __init__ method of your network that depend on your
          dataset.
        * The :py:meth:`~BaseModel.transform_output` method rescales the network output using the target normalizer
          from thedataset.
        * The :py:meth:`~BaseModel.step` method takes care of calculating the loss, logging additional metrics defined
          in the ``logging_metrics`` attribute and plots of sample predictions. You can override this method to add
          custom interpretations or pass extra arguments to the networks forward method.
        * The :py:meth:`~BaseModel.epoch_end` method can be used to calculate summaries of each epoch such as
          statistics on the encoder length, etc.
        * The :py:meth:`~BaseModel.predict` method makes predictions using a dataloader or dataset. Override it if you
          need to pass additional arguments to ``forward`` by default.

    To implement your own architecture, it is best to
    go throught the :ref:`Using custom data and implementing custom models <new-model-tutorial>` and
    to look at existing ones to understand what might be a good approach.

    Example:

        .. code-block:: python

            class Network(BaseModel):

                def __init__(self, my_first_parameter: int=2, loss=SMAPE()):
                    self.save_hyperparameters()
                    super().__init__()
                    self.loss = loss

                def forward(self, x):
                    encoding_target = x["encoder_target"]
                    return dict(prediction=..., target_scale=x["target_scale"])

    """

    def __init__(
        self,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        learning_rate: Union[float, List[float]] = 1e-3,
        log_gradient_flow: bool = False,
        loss: Metric = SMAPE(),
        logging_metrics: nn.ModuleList = nn.ModuleList([]),
        reduce_on_plateau_patience: int = 1000,
        reduce_on_plateau_min_lr: float = 1e-5,
        weight_decay: float = 0.0,
        monotone_constaints: Dict[str, int] = {},
        output_transformer: Callable = None,
        optimizer="ranger",
    ):
        """
        BaseModel for timeseries forecasting from which to inherit from

        Args:
            log_interval (Union[int, float], optional): Batches after which predictions are logged. If < 1.0, will log
                multiple entries per batch. Defaults to -1.
            log_val_interval (Union[int, float], optional): batches after which predictions for validation are
                logged. Defaults to None/log_interval.
            learning_rate (float, optional): Learning rate. Defaults to 1e-3.
            log_gradient_flow (bool): If to log gradient flow, this takes time and should be only done to diagnose
                training failures. Defaults to False.
            loss (Metric, optional): metric to optimize. Defaults to SMAPE().
            logging_metrics (nn.ModuleList[MultiHorizonMetric]): list of metrics that are logged during training.
                Defaults to [].
            reduce_on_plateau_patience (int): patience after which learning rate is reduced by a factor of 10. Defaults
                to 1000
            reduce_on_plateau_min_lr (float): minimum learning rate for reduce on plateua learning rate scheduler.
                Defaults to 1e-5
            weight_decay (float): weight decay. Defaults to 0.0.
            monotone_constaints (Dict[str, int]): dictionary of monotonicity constraints for continuous decoder
                variables mapping
                position (e.g. ``"0"`` for first position) to constraint (``-1`` for negative and ``+1`` for positive,
                larger numbers add more weight to the constraint vs. the loss but are usually not necessary).
                This constraint significantly slows down training. Defaults to {}.
            output_transformer (Callable): transformer that takes network output and transforms it to prediction space.
                Defaults to None which is equivalent to ``lambda out: out["prediction"]``.
            optimizer (str): Optimizer, "ranger", "adam" or "adamw". Defaults to "ranger".
        """
        super().__init__()
        # update hparams
        frame = inspect.currentframe()
        init_args = get_init_args(frame)
        self.save_hyperparameters({name: val for name, val in init_args.items() if name not in self.hparams})

        # update log interval if not defined
        if self.hparams.log_val_interval is None:
            self.hparams.log_val_interval = self.hparams.log_interval

        if not hasattr(self, "loss"):
            self.loss = loss
        if not hasattr(self, "logging_metrics"):
            self.logging_metrics = nn.ModuleList([l for l in logging_metrics])
        if not hasattr(self, "output_transformer"):
            self.output_transformer = output_transformer

    def transform_output(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract prediction from network output and rescale it to real space / de-normalize it.

        Args:
            out (Dict[str, torch.Tensor]): Network output with "prediction" and "target_scale" entries.
                the output will be either

                * The input if the input is a tensor
                * ``out["prediction"]`` if there is no ``output_transformer``
                  (which is a :py:class:`~pytorch_forecasting.data.encoders.TorchNormalizer`) defined or
                  ``out["output_transformation"] is None``
                * transformed output - this is either the ``output_transformer`` applied to the input
                  or, in case of :py:class:`~pytorch_forecasting.metrics.DistributionLoss` as loss module,
                  the loss module is used together with the ``output_transformer``

        Returns:
            torch.Tensor: rescaled prediction
        """
        # no transformation possible
        if isinstance(out, torch.Tensor):
            return out

        # no transformation logic
        elif self.output_transformer is None or out.get("output_transformation", True) is None:
            out = out["prediction"]

        # distribution transformation
        elif isinstance(self.loss, DistributionLoss) or (
            isinstance(self.loss, MultiLoss) and isinstance(self.loss[0], DistributionLoss)
        ):
            if isinstance(self.loss, MultiLoss):
                # todo: handle mixed losses
                out = self.loss.rescale_parameters(
                    out["prediction"],
                    target_scale=out["target_scale"],
                    encoder=self.output_transformer.normalizers,  # need to use normalizer per encoder
                )
            else:
                out = self.loss.rescale_parameters(
                    out["prediction"], target_scale=out["target_scale"], encoder=self.output_transformer
                )

        # normal output transformation
        else:
            out = self.output_transformer(out)
        return out

    def size(self) -> int:
        """
        get number of parameters in model
        """
        return sum(p.numel() for p in self.parameters())

    def training_step(self, batch, batch_idx):
        """
        Train on batch.
        """
        x, y = batch
        log, _ = self.step(x, y, batch_idx)
        # log loss
        self.log("train_loss", log["loss"], on_step=True, on_epoch=True, prog_bar=True)
        return log

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        log, _ = self.step(x, y, batch_idx)  # log loss
        self.log("val_loss", log["loss"], on_step=False, on_epoch=True, prog_bar=True)
        return log

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs)

    def step(
        self, x: Dict[str, torch.Tensor], y: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Run for each train/val step.

        Args:
            x (Dict[str, torch.Tensor]): x as passed to the network by the dataloader
            y (Tuple[torch.Tensor, torch.Tensor]): y as passed to the loss function by the dataloader
            batch_idx (int): batch number
            **kwargs: additional arguments to pass to the network apart from ``x``

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: tuple where the first
                entry is a dictionary to which additional logging results can be added for consumption in the
                ``epoch_end`` hook and the second entry is the model's output.
        """
        # pack y sequence if different encoder lengths exist
        if (x["decoder_lengths"] < x["decoder_lengths"].max()).any():
            if isinstance(y[0], (list, tuple)):
                y = (
                    [
                        rnn.pack_padded_sequence(
                            y_part, lengths=x["decoder_lengths"].cpu(), batch_first=True, enforce_sorted=False
                        )
                        for y_part in y[0]
                    ],
                    y[1],
                )
            else:
                y = (
                    rnn.pack_padded_sequence(
                        y[0], lengths=x["decoder_lengths"].cpu(), batch_first=True, enforce_sorted=False
                    ),
                    y[1],
                )

        if self.training and len(self.hparams.monotone_constaints) > 0:
            # calculate gradient with respect to continous decoder features
            x["decoder_cont"].requires_grad_(True)
            assert not torch._C._get_cudnn_enabled(), (
                "To use monotone constraints, wrap model and training in context "
                "`torch.backends.cudnn.flags(enable=False)`"
            )
            out = self(x, **kwargs)
            out["prediction"] = self.transform_output(out)
            prediction = out["prediction"]

            # handle multiple targets
            prediction_list = to_list(prediction)
            gradient = 0
            # todo: should monotone constrains be applicable to certain targets?
            for pred in prediction_list:
                gradient = gradient + torch.autograd.grad(
                    outputs=pred,
                    inputs=x["decoder_cont"],
                    grad_outputs=torch.ones_like(pred),  # t
                    create_graph=True,  # allows usage in graph
                    allow_unused=True,
                )[0]

            # select relevant features
            indices = torch.tensor(
                [self.hparams.x_reals.index(name) for name in self.hparams.monotone_constaints.keys()]
            )
            monotonicity = torch.tensor(
                [val for val in self.hparams.monotone_constaints.values()], dtype=gradient.dtype, device=gradient.device
            )
            # add additionl loss if gradient points in wrong direction
            gradient = gradient[..., indices] * monotonicity[None, None]
            monotinicity_loss = gradient.clamp_max(0).mean()
            # multiply monotinicity loss by large number to ensure relevance and take to the power of 2
            # for smoothness of loss function
            monotinicity_loss = 10 * torch.pow(monotinicity_loss, 2)
            if isinstance(self.loss, MASE):
                loss = self.loss(
                    prediction, y, encoder_target=x["encoder_target"], encoder_lengths=x["encoder_lengths"]
                )
            else:
                loss = self.loss(prediction, y)

            loss = loss * (1 + monotinicity_loss)
        else:
            out = self(x, **kwargs)
            out["prediction"] = self.transform_output(out)

            # calculate loss
            prediction = out["prediction"]
            if isinstance(self.loss, MASE):
                loss = self.loss(
                    prediction, y, encoder_target=x["encoder_target"], encoder_lengths=x["encoder_lengths"]
                )
            else:
                loss = self.loss(prediction, y)

        # log
        self.log_metrics(x, y, out)
        if self.log_interval > 0:
            self.log_prediction(x, out, batch_idx)
        log = {"loss": loss, "n_samples": x["decoder_lengths"].size(0)}

        return log, out

    def log_metrics(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        out: Dict[str, torch.Tensor],
    ) -> None:
        """
        Log metrics every training/validation step.

        Args:
            x (Dict[str, torch.Tensor]): x as passed to the network by the dataloader
            y (torch.Tensor): y as passed to the loss function by the dataloader
            out (Dict[str, torch.Tensor]): output of the network
        """
        # logging losses - for each target
        if isinstance(self.loss, MultiLoss):
            y_hat_detached = [p.detach() for p in out["prediction"]]
            y_hat_point_detached = self.loss.to_prediction(y_hat_detached)
        else:
            y_hat_point_detached = [self.loss.to_prediction(out["prediction"].detach())]

        for metric in self.logging_metrics:
            for idx, y_point, y_part, encoder_target in zip(
                list(range(len(y_hat_point_detached))),
                y_hat_point_detached,
                to_list(y[0]),
                to_list(x["encoder_target"]),
            ):
                y_true = (y_part, y[1])
                if isinstance(metric, MASE):
                    loss_value = metric(
                        y_point, y_true, encoder_target=encoder_target, encoder_lengths=x["encoder_lengths"]
                    )
                else:
                    loss_value = metric(y_point, y_true)
                if len(y_hat_point_detached) > 1:
                    target_tag = f"Target {idx}"
                else:
                    target_tag = ""
                self.log(
                    f"{target_tag}_{['val', 'train'][self.training]}_{metric.name}",
                    loss_value,
                    on_step=self.training,
                    on_epoch=True,
                )

    def forward(
        self, x: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Network forward pass.

        Args:
            x (Dict[str, Union[torch.Tensor, List[torch.Tensor]]]): network input (x as returned by the dataloader).
                See :py:meth:`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet.to_dataloader` method that
                returns a tuple of ``x`` and ``y``. This function expects ``x``.

        Returns:
            Dict[str, Union[torch.Tensor, List[torch.Tensor]]]: network outputs / dictionary of tensors or list
                of tensors. The minimal required entries in the dictionary are (and shapes in brackets):

                * ``prediction`` (batch_size x n_decoder_time_steps x n_outputs or list thereof with each
                  entry for a different target): unscaled predictions that can be fed to metric. List of tensors
                  if multiple targets are predicted at the same time.
                * ``target_scale`` (batch_size x scale_size or list thereof with each entry for a different target):
                  target scales that allow rescaling the predictions into the real space.
                  The scale can mostly be directly taken from ``x``, i.e. ``target_scale=x["target_scale"]``
        """
        raise NotImplementedError()

    def epoch_end(self, outputs):
        """
        Run at epoch end for training or validation. Can be overriden in models.
        """
        pass

    @property
    def log_interval(self) -> float:
        """
        Log interval depending if training or validating
        """
        if self.training:
            return self.hparams.log_interval
        else:
            return self.hparams.log_val_interval

    def log_prediction(self, x: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Log metrics every training/validation step.

        Args:
            x (Dict[str, torch.Tensor]): x as passed to the network by the dataloader
            out (Dict[str, torch.Tensor]): output of the network
            batch_idx (int): current batch index
        """
        # log single prediction figure
        if (batch_idx % self.log_interval == 0 or self.log_interval < 1.0) and self.log_interval > 0:
            if self.log_interval < 1.0:  # log multiple steps
                log_indices = torch.arange(
                    0, len(x["encoder_lengths"]), max(1, round(self.log_interval * len(x["encoder_lengths"])))
                )
            else:
                log_indices = [0]
            for idx in log_indices:
                fig = self.plot_prediction(x, out, idx=idx, add_loss_to_title=True)
                tag = f"{['Val', 'Train'][self.training]} prediction"
                if self.training:
                    tag += f" of item {idx} in global batch {self.global_step}"
                else:
                    tag += f" of item {idx} in batch {batch_idx}"
                if isinstance(fig, (list, tuple)):
                    for idx, f in enumerate(fig):
                        self.logger.experiment.add_figure(
                            f"Target {idx} {tag}",
                            f,
                            global_step=self.global_step,
                        )
                else:
                    self.logger.experiment.add_figure(
                        tag,
                        fig,
                        global_step=self.global_step,
                    )

    def plot_prediction(
        self,
        x: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
        idx: int = 0,
        add_loss_to_title: Union[Metric, torch.Tensor, bool] = False,
        show_future_observed: bool = True,
        ax=None,
    ) -> plt.Figure:
        """
        Plot prediction of prediction vs actuals

        Args:
            x: network input
            out: network output
            idx: index of prediction to plot
            add_loss_to_title: if to add loss to title or loss function to calculate. Can be either metrics,
                bool indicating if to use loss metric or tensor which contains losses for all samples.
                Calcualted losses are determined without weights. Default to False.
            show_future_observed: if to show actuals for future. Defaults to True.
            ax: matplotlib axes to plot on

        Returns:
            matplotlib figure
        """
        # all true values for y of the first sample in batch
        encoder_targets = to_list(x["encoder_target"])
        decoder_targets = to_list(x["decoder_target"])

        # get predictions
        y_raws = to_list(out["prediction"])
        y_hats = to_list(self.loss.to_prediction(out["prediction"]))
        y_quantiles = to_list(self.loss.to_quantiles(out["prediction"]))

        # for each target, plot
        figs = []
        for y_raw, y_hat, y_quantile, encoder_target, decoder_target in zip(
            y_raws, y_hats, y_quantiles, encoder_targets, decoder_targets
        ):

            y_all = torch.cat([encoder_target[idx], decoder_target[idx]])
            max_encoder_length = x["encoder_lengths"].max()
            y = torch.cat(
                (
                    y_all[: x["encoder_lengths"][idx]],
                    y_all[max_encoder_length : (max_encoder_length + x["decoder_lengths"][idx])],
                ),
            )
            # move predictions to cpu
            y_hat = y_hat.detach().cpu()[idx, : x["decoder_lengths"][idx]]
            y_quantile = y_quantile.detach().cpu()[idx, : x["decoder_lengths"][idx]]
            y_raw = y_raw.detach().cpu()[idx, : x["decoder_lengths"][idx]]

            # move to cpu
            y = y.detach().cpu()
            # create figure
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
            n_pred = y_hat.shape[0]
            x_obs = np.arange(-(y.shape[0] - n_pred), 0)
            x_pred = np.arange(n_pred)
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
            if show_future_observed:
                plotter(x_pred, y[-n_pred:], label=None, c=obs_color)

            # plot prediction
            plotter(x_pred, y_hat, label="predicted", c=pred_color)

            # plot predicted quantiles
            plotter(x_pred, y_quantile[:, y_quantile.shape[1] // 2], c=pred_color, alpha=0.15)
            for i in range(y_quantile.shape[1] // 2):
                if len(x_pred) > 1:
                    ax.fill_between(x_pred, y_quantile[:, i], y_quantile[:, -i - 1], alpha=0.15, fc=pred_color)
                else:
                    quantiles = torch.tensor([[y_quantile[0, i]], [y_quantile[0, -i - 1]]])
                    ax.errorbar(
                        x_pred,
                        y[[-n_pred]],
                        yerr=quantiles - y[-n_pred],
                        c=pred_color,
                        capsize=1.0,
                    )
            if add_loss_to_title is not False:
                if isinstance(add_loss_to_title, bool):
                    loss = self.loss
                elif isinstance(add_loss_to_title, torch.Tensor):
                    loss = add_loss_to_title.detach()[idx].item()
                elif isinstance(add_loss_to_title, Metric):
                    loss = add_loss_to_title
                    loss.quantiles = self.loss.quantiles
                else:
                    raise ValueError(f"add_loss_to_title '{add_loss_to_title}'' is unkown")
                if isinstance(loss, MASE):
                    loss_value = loss(y_raw[None], (y[-n_pred:][None], None), y[:n_pred][None])
                elif isinstance(loss, Metric):
                    loss_value = loss(y_raw[None], (y[-n_pred:][None], None))
                else:
                    loss_value = loss
                ax.set_title(f"Loss {loss_value}")
            ax.set_xlabel("Time index")
            fig.legend()
            figs.append(fig)

        # return multiple of target is a list, otherwise return single figure
        if isinstance(x["encoder_target"], (tuple, list)):
            return figs
        else:
            return fig

    def log_gradient_flow(self, named_parameters: Dict[str, torch.Tensor]) -> None:
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
        self.logger.experiment.add_figure("Gradient flow", fig, global_step=self.global_step)

    def on_after_backward(self):
        """
        Log gradient flow for debugging.
        """
        if (
            self.hparams.log_interval > 0
            and self.global_step % self.hparams.log_interval == 0
            and self.hparams.log_gradient_flow
        ):
            self.log_gradient_flow(self.named_parameters())

    def configure_optimizers(self):
        """
        Configure optimizers.

        Uses single Ranger optimizer. Depending if learning rate is a list or a single float, implement dynamic
        learning rate scheduler or deterministic version

        Returns:
            Tuple[List]: first entry is list of optimizers and second is list of schedulers
        """
        # either set a schedule of lrs or find it dynamically
        if isinstance(self.hparams.learning_rate, (list, tuple)):  # set schedule
            lrs = self.hparams.learning_rate
            if self.hparams.optimizer == "adam":
                optimizer = torch.optim.Adam(self.parameters(), lr=lrs[0])
            elif self.hparams.optimizer == "adamw":
                optimizer = torch.optim.AdamW(self.parameters(), lr=lrs[0])
            elif self.hparams.optimizer == "ranger":
                optimizer = Ranger(self.parameters(), lr=lrs[0], weight_decay=self.hparams.weight_decay)
            else:
                raise ValueError(f"Optimizer of self.hparams.optimizer={self.hparams.optimizer} unknown")
            # normalize lrs
            lrs = np.array(lrs) / lrs[0]
            schedulers = [
                {
                    "scheduler": LambdaLR(optimizer, lambda epoch: lrs[min(epoch, len(lrs) - 1)]),
                    "interval": "epoch",
                    "reduce_on_plateau": False,
                    "frequency": 1,
                }
            ]
        else:  # find schedule based on validation loss
            if self.hparams.optimizer == "adam":
                optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
            elif self.hparams.optimizer == "ranger":
                optimizer = Ranger(
                    self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
                )
            elif self.hparams.optimizer == "adamw":
                optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
            else:
                raise ValueError(f"Optimizer of self.hparams.optimizer={self.hparams.optimizer} unknown")
            schedulers = [
                {
                    "scheduler": ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=0.1,
                        patience=self.hparams.reduce_on_plateau_patience,
                        cooldown=self.hparams.reduce_on_plateau_patience,
                        min_lr=self.hparams.reduce_on_plateau_min_lr,
                    ),
                    "monitor": "val_loss",  # Default: val_loss
                    "interval": "epoch",
                    "reduce_on_plateau": True,
                    "frequency": 1,
                }
            ]
        return [optimizer], schedulers

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs) -> LightningModule:
        """
        Create model from dataset, i.e. save dataset parameters in model

        This function should be called as ``super().from_dataset()`` in a derived models that implement it

        Args:
            dataset (TimeSeriesDataSet): timeseries dataset

        Returns:
            BaseModel: Model that can be trained
        """
        if "output_transformer" not in kwargs:
            kwargs["output_transformer"] = dataset.target_normalizer
        net = cls(**kwargs)
        net.dataset_parameters = dataset.get_parameters()
        return net

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["dataset_parameters"] = getattr(
            self, "dataset_parameters", None
        )  # add dataset parameters for making fast predictions
        checkpoint["loss"] = cloudpickle.dumps(self.loss)  # restore loss
        checkpoint["output_transformer"] = cloudpickle.dumps(self.output_transformer)  # restore output transformer
        # hyper parameters are passed as arguments directly and not as single dictionary
        checkpoint["hparams_name"] = "kwargs"

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.dataset_parameters = checkpoint.get("dataset_parameters", None)
        self.loss = cloudpickle.loads(checkpoint["loss"])
        self.output_transformer = cloudpickle.loads(checkpoint["output_transformer"])

    def predict(
        self,
        data: Union[DataLoader, pd.DataFrame, TimeSeriesDataSet],
        mode: Union[str, Tuple[str, str]] = "prediction",
        return_index: bool = False,
        return_decoder_lengths: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        fast_dev_run: bool = False,
        show_progress_bar: bool = False,
        return_x: bool = False,
        **kwargs,
    ):
        """
        Run inference / prediction.

        Args:
            dataloader: dataloader, dataframe or dataset
            mode: one of "prediction", "quantiles" or "raw", or tuple ``("raw", output_name)`` where output_name is
                a name in the dictionary returned by ``forward()``
            return_index: if to return the prediction index
            return_decoder_lengths: if to return decoder_lengths
            batch_size: batch size for dataloader - only used if data is not a dataloader is passed
            num_workers: number of workers for dataloader - only used if data is not a dataloader is passed
            fast_dev_run: if to only return results of first batch
            show_progress_bar: if to show progress bar. Defaults to False.
            return_x: if to return network inputs
            **kwargs: additional arguments to network's forward method

        Returns:
            output, x, index, decoder_lengths: some elements might not be present depending on what is configured
                to be returned
        """
        # convert to dataloader
        if isinstance(data, pd.DataFrame):
            data = TimeSeriesDataSet.from_parameters(self.dataset_parameters, data, predict=True)
        if isinstance(data, TimeSeriesDataSet):
            dataloader = data.to_dataloader(batch_size=batch_size, train=False, num_workers=num_workers)
        else:
            dataloader = data

        # ensure passed dataloader is correct
        assert isinstance(dataloader.dataset, TimeSeriesDataSet), "dataset behind dataloader mut be TimeSeriesDataSet"

        # prepare model
        self.eval()  # no dropout, etc. no gradients

        # run predictions
        output = []
        decode_lenghts = []
        x_list = []
        index = []
        progress_bar = tqdm(desc="Predict", unit=" batches", total=len(dataloader), disable=not show_progress_bar)
        with torch.no_grad():
            for x, _ in dataloader:
                # move data to appropriate device
                data_device = x["encoder_cont"].device
                if data_device != self.device:
                    for name in x.keys():
                        if isinstance(x[name], (tuple, list)):
                            x[name] = [xi.to(self.device) for xi in x[name]]
                        else:
                            x[name] = x[name].to(self.device)

                # make prediction
                out = self(x, **kwargs)  # raw output is dictionary
                out["prediction"] = self.transform_output(out)

                lengths = x["decoder_lengths"]
                if return_decoder_lengths:
                    decode_lenghts.append(lengths)
                nan_mask = create_mask(lengths.max(), lengths)
                if isinstance(mode, (tuple, list)):
                    if mode[0] == "raw":
                        out = out[mode[1]]
                    else:
                        raise ValueError(
                            f"If a tuple is specified, the first element must be 'raw' - got {mode[0]} instead"
                        )
                elif mode == "prediction":
                    out = self.loss.to_prediction(out["prediction"])
                    # mask non-predictions
                    if isinstance(out, (list, tuple)):
                        out = [
                            o.masked_fill(nan_mask, torch.tensor(float("nan"))) if o.dtype == torch.float else o
                            for o in out
                        ]
                    elif out.dtype == torch.float:  # only floats can be filled with nans
                        out = out.masked_fill(nan_mask, torch.tensor(float("nan")))
                elif mode == "quantiles":
                    out = self.loss.to_quantiles(out["prediction"])
                    # mask non-predictions
                    if isinstance(out, (list, tuple)):
                        out = [
                            o.masked_fill(nan_mask.unsqueeze(-1), torch.tensor(float("nan")))
                            if o.dtype == torch.float
                            else o
                            for o in out
                        ]
                    elif out.dtype == torch.float:
                        out = out.masked_fill(nan_mask.unsqueeze(-1), torch.tensor(float("nan")))
                elif mode == "raw":
                    pass
                else:
                    raise ValueError(f"Unknown mode {mode} - see docs for valid arguments")

                output.append(out)
                if return_x:
                    x_list.append(x)
                if return_index:
                    index.append(dataloader.dataset.x_to_index(x))
                progress_bar.update()
                if fast_dev_run:
                    break

        # concatenate output (of different batches)
        if isinstance(mode, (tuple, list)) or mode != "raw":
            if isinstance(output[0], (tuple, list)) and len(output[0]) > 0 and isinstance(output[0][0], torch.Tensor):
                output = [torch.cat(out, dim=0) for out in output]
            else:
                output = torch.cat(output, dim=0)
        elif mode == "raw":
            output_cat = {}
            for name in output[0].keys():
                v0 = output[0][name]
                if isinstance(v0, torch.Tensor):
                    output_cat[name] = torch.cat([out[name] for out in output], dim=0)
                elif isinstance(v0, (tuple, list)) and len(v0) > 0 and isinstance(v0[0], torch.Tensor):
                    output_cat[name] = [torch.cat(out[name], dim=0) for out in output]
                else:
                    try:
                        output_cat[name] = np.concatenate([out[name] for out in output], axis=0)
                    except ValueError:
                        output_cat[name] = [out[name] for out in output]
            output = output_cat

        # generate output
        if return_x or return_index or return_decoder_lengths:
            output = [output]
        if return_x:
            x_cat = {}
            for name in x_list[0].keys():
                x_cat[name] = torch.cat([x[name] for x in x_list], dim=0)
            x_cat = x_cat
            output.append(x_cat)
        if return_index:
            output.append(pd.concat(index, axis=0, ignore_index=True))
        if return_decoder_lengths:
            output.append(torch.cat(decode_lenghts, dim=0))
        return output

    def predict_dependency(
        self,
        data: Union[DataLoader, pd.DataFrame, TimeSeriesDataSet],
        variable: str,
        values: Iterable,
        mode: str = "dataframe",
        target="decoder",
        show_progress_bar: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, torch.Tensor, pd.Series, pd.DataFrame]:
        """
        Predict partial dependency.


        Args:
            data (Union[DataLoader, pd.DataFrame, TimeSeriesDataSet]): data
            variable (str): variable which to modify
            values (Iterable): array of values to probe
            mode (str, optional): Output mode. Defaults to "dataframe". Either

                * "series": values are average prediction and index are probed values
                * "dataframe": columns are as obtained by the `dataset.x_to_index()` method,
                    prediction (which is the mean prediction over the time horizon),
                    normalized_prediction (which are predictions devided by the prediction for the first probed value)
                    the variable name for the probed values
                * "raw": outputs a tensor of shape len(values) x prediction_shape

            target: Defines which values are overwritten for making a prediction.
                Same as in :py:meth:`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet.set_overwrite_values`.
                Defaults to "decoder".
            show_progress_bar: if to show progress bar. Defaults to False.
            **kwargs: additional kwargs to :py:meth:`~predict` method

        Returns:
            Union[np.ndarray, torch.Tensor, pd.Series, pd.DataFrame]: output
        """
        values = np.asarray(values)
        if isinstance(data, pd.DataFrame):  # convert to dataframe
            data = TimeSeriesDataSet.from_parameters(self.dataset_parameters, data, predict=True)
        elif isinstance(data, DataLoader):
            data = data.dataset

        results = []
        progress_bar = tqdm(desc="Predict", unit=" batches", total=len(values), disable=not show_progress_bar)
        for idx, value in enumerate(values):
            # set values
            data.set_overwrite_values(variable=variable, values=value, target=target)
            # predict
            kwargs.setdefault("mode", "prediction")

            if idx == 0 and mode == "dataframe":  # need index for returning as dataframe
                res, index = self.predict(data, return_index=True, **kwargs)
                results.append(res)
            else:
                results.append(self.predict(data, **kwargs))
            # increment progress
            progress_bar.update()

        data.reset_overwrite_values()  # reset overwrite values to avoid side-effect

        # results to one tensor
        results = torch.stack(results, dim=0)

        # convert results to requested output format
        if mode == "series":
            results = results[:, ~torch.isnan(results[0])].mean(1)  # average samples and prediction horizon
            results = pd.Series(results, index=values)

        elif mode == "dataframe":
            # take mean over time
            is_nan = torch.isnan(results)
            results[is_nan] = 0
            results = results.sum(-1) / (~is_nan).float().sum(-1)

            # create dataframe
            dependencies = (
                index.iloc[np.tile(np.arange(len(index)), len(values))]
                .reset_index(drop=True)
                .assign(prediction=results.flatten())
            )
            dependencies[variable] = values.repeat(len(data))
            first_prediction = dependencies.groupby(data.group_ids, observed=True).prediction.transform("first")
            dependencies["normalized_prediction"] = dependencies["prediction"] / first_prediction
            dependencies["id"] = dependencies.groupby(data.group_ids, observed=True).ngroup()
            results = dependencies

        elif mode == "raw":
            pass

        else:
            raise ValueError(f"mode {mode} is unknown - see documentation for available modes")

        return results


class BaseModelWithCovariates(BaseModel):
    """
    Model with additional methods using covariates.

    Assumes the following hyperparameters:

    Args:
        static_categoricals (List[str]): names of static categorical variables
        static_reals (List[str]): names of static continuous variables
        time_varying_categoricals_encoder (List[str]): names of categorical variables for encoder
        time_varying_categoricals_decoder (List[str]): names of categorical variables for decoder
        time_varying_reals_encoder (List[str]): names of continuous variables for encoder
        time_varying_reals_decoder (List[str]): names of continuous variables for decoder
        x_reals (List[str]): order of continuous variables in tensor passed to forward function
        x_categoricals (List[str]): order of categorical variables in tensor passed to forward function
        embedding_sizes (Dict[str, Tuple[int, int]]): dictionary mapping categorical variables to tuple of integers
            where the first integer denotes the number of categorical classes and the second the embedding size
        embedding_labels (Dict[str, List[str]]): dictionary mapping (string) indices to list of categorical labels
        embedding_paddings (List[str]): names of categorical variables for which label 0 is always mapped to an
             embedding vector filled with zeros
        categorical_groups (Dict[str, List[str]]): dictionary of categorical variables that are grouped together and
            can also take multiple values simultaneously (e.g. holiday during octoberfest). They should be implemented
            as bag of embeddings
    """

    @property
    def reals(self) -> List[str]:
        """List of all continuous variables in model"""
        return list(
            dict.fromkeys(
                self.hparams.static_reals
                + self.hparams.time_varying_reals_encoder
                + self.hparams.time_varying_reals_decoder
            )
        )

    @property
    def categoricals(self) -> List[str]:
        """List of all categorical variables in model"""
        return list(
            dict.fromkeys(
                self.hparams.static_categoricals
                + self.hparams.time_varying_categoricals_encoder
                + self.hparams.time_varying_categoricals_decoder
            )
        )

    @property
    def static_variables(self) -> List[str]:
        """List of all static variables in model"""
        return self.hparams.static_categoricals + self.hparams.static_reals

    @property
    def encoder_variables(self) -> List[str]:
        """List of all encoder variables in model (excluding static variables)"""
        return self.hparams.time_varying_categoricals_encoder + self.hparams.time_varying_reals_encoder

    @property
    def decoder_variables(self) -> List[str]:
        """List of all decoder variables in model (excluding static variables)"""
        return self.hparams.time_varying_categoricals_decoder + self.hparams.time_varying_reals_decoder

    @property
    def categorical_groups_mapping(self) -> Dict[str, str]:
        """Mapping of categorical variables to categorical groups"""
        groups = {}
        for group_name, sublist in self.hparams.categorical_groups.items():
            groups.update({name: group_name for name in sublist})
        return groups

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: List[str] = None,
        **kwargs,
    ) -> LightningModule:
        """
        Create model from dataset and set parameters related to covariates.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: List of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            LightningModule
        """
        # assert fixed encoder and decoder length for the moment
        if allowed_encoder_known_variable_names is None:
            allowed_encoder_known_variable_names = (
                dataset.time_varying_known_categoricals + dataset.time_varying_known_reals
            )

        # embeddings
        embedding_labels = {
            name: encoder.classes_
            for name, encoder in dataset.categorical_encoders.items()
            if name in dataset.categoricals
        }
        embedding_paddings = dataset.dropout_categoricals
        # determine embedding sizes based on heuristic
        embedding_sizes = {
            name: (len(encoder.classes_), get_embedding_size(len(encoder.classes_)))
            for name, encoder in dataset.categorical_encoders.items()
            if name in dataset.categoricals
        }
        embedding_sizes.update(kwargs.get("embedding_sizes", {}))
        kwargs.setdefault("embedding_sizes", embedding_sizes)

        new_kwargs = dict(
            static_categoricals=dataset.static_categoricals,
            time_varying_categoricals_encoder=[
                name for name in dataset.time_varying_known_categoricals if name in allowed_encoder_known_variable_names
            ]
            + dataset.time_varying_unknown_categoricals,
            time_varying_categoricals_decoder=dataset.time_varying_known_categoricals,
            static_reals=dataset.static_reals,
            time_varying_reals_encoder=[
                name for name in dataset.time_varying_known_reals if name in allowed_encoder_known_variable_names
            ]
            + dataset.time_varying_unknown_reals,
            time_varying_reals_decoder=dataset.time_varying_known_reals,
            x_reals=dataset.reals,
            x_categoricals=dataset.flat_categoricals,
            embedding_labels=embedding_labels,
            embedding_paddings=embedding_paddings,
            categorical_groups=dataset.variable_groups,
        )
        new_kwargs.update(kwargs)
        return super().from_dataset(dataset, **new_kwargs)

    def calculate_prediction_actual_by_variable(
        self,
        x: Dict[str, torch.Tensor],
        y_pred: torch.Tensor,
        normalize: bool = True,
        bins: int = 95,
        std: float = 2.0,
        log_scale: bool = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Calculate predictions and actuals by variable averaged by ``bins`` bins spanning from ``-std`` to ``+std``

        Args:
            x: input as ``forward()``
            y_pred: predictions obtained by ``self.transform_output(self(x, **kwargs))``
            normalize: if to return normalized averages, i.e. mean or sum of ``y``
            bins: number of bins to calculate
            std: number of standard deviations for standard scaled continuous variables
            log_scale (str, optional): if to plot in log space. If None, determined based on skew of values.
                Defaults to None.

        Returns:
            dictionary that can be used to plot averages with :py:meth:`~plot_prediction_actual_by_variable`
        """
        support = {}  # histogram
        # averages
        averages_actual = {}
        averages_prediction = {}

        # mask values and transform to log space
        max_encoder_length = x["decoder_lengths"].max()
        mask = create_mask(max_encoder_length, x["decoder_lengths"], inverse=True)
        # select valid y values
        y_flat = x["decoder_target"][mask]
        y_pred_flat = y_pred[mask]

        # determine in which average in log-space to transform data
        if log_scale is None:
            skew = torch.mean(((y_flat - torch.mean(y_flat)) / torch.std(y_flat)) ** 3)
            log_scale = skew > 1.6

        if log_scale:
            y_flat = torch.log(y_flat + 1e-8)
            y_pred_flat = torch.log(y_pred_flat + 1e-8)

        # real bins
        positive_bins = (bins - 1) // 2

        # if to normalize
        if normalize:
            reduction = "mean"
        else:
            reduction = "sum"
        # continuous variables
        reals = x["decoder_cont"]
        for idx, name in enumerate(self.hparams.x_reals):
            averages_actual[name], support[name] = groupby_apply(
                (reals[..., idx][mask] * positive_bins / std).round().clamp(-positive_bins, positive_bins).long()
                + positive_bins,
                y_flat,
                bins=bins,
                reduction=reduction,
                return_histogram=True,
            )
            averages_prediction[name], _ = groupby_apply(
                (reals[..., idx][mask] * positive_bins / std).round().clamp(-positive_bins, positive_bins).long()
                + positive_bins,
                y_pred_flat,
                bins=bins,
                reduction=reduction,
                return_histogram=True,
            )

        # categorical_variables
        cats = x["decoder_cat"]
        for idx, name in enumerate(self.hparams.x_categoricals):  # todo: make it work for grouped categoricals
            reduction = "sum"
            name = self.categorical_groups_mapping.get(name, name)
            averages_actual_cat, support_cat = groupby_apply(
                cats[..., idx][mask],
                y_flat,
                bins=self.hparams.embedding_sizes[name][0],
                reduction=reduction,
                return_histogram=True,
            )
            averages_prediction_cat, _ = groupby_apply(
                cats[..., idx][mask],
                y_pred_flat,
                bins=self.hparams.embedding_sizes[name][0],
                reduction=reduction,
                return_histogram=True,
            )

            # add either to existing calculations or
            if name in averages_actual:
                averages_actual[name] += averages_actual_cat
                support[name] += support_cat
                averages_prediction[name] += averages_prediction_cat
            else:
                averages_actual[name] = averages_actual_cat
                support[name] = support_cat
                averages_prediction[name] = averages_prediction_cat

        if normalize:  # run reduction for categoricals
            for name in self.hparams.embedding_sizes.keys():
                averages_actual[name] /= support[name].clamp(min=1)
                averages_prediction[name] /= support[name].clamp(min=1)

        if log_scale:
            for name in support.keys():
                averages_actual[name] = torch.exp(averages_actual[name])
                averages_prediction[name] = torch.exp(averages_prediction[name])

        return {
            "support": support,
            "average": {"actual": averages_actual, "prediction": averages_prediction},
            "std": std,
        }

    def plot_prediction_actual_by_variable(
        self, data: Dict[str, Dict[str, torch.Tensor]], name: str = None, ax=None, log_scale: bool = None
    ) -> Union[Dict[str, plt.Figure], plt.Figure]:
        """
        Plot predicions and actual averages by variables

        Args:
            data (Dict[str, Dict[str, torch.Tensor]]): data obtained from
                :py:meth:`~calculate_prediction_actual_by_variable`
            name (str, optional): name of variable for which to plot actuals vs predictions. Defaults to None which
                means returning a dictionary of plots for all variables.
            log_scale (str, optional): if to plot in log space. If None, determined based on skew of values.
                Defaults to None.

        Raises:
            ValueError: if the variable name is unkown

        Returns:
            Union[Dict[str, plt.Figure], plt.Figure]: matplotlib figure
        """
        if name is None:  # run recursion for figures
            figs = {name: self.plot_prediction_actual_by_variable(data, name) for name in data["support"].keys()}
            return figs
        else:
            # create figure
            kwargs = {}
            # adjust figure size for figures with many labels
            if self.hparams.embedding_sizes.get(name, [1e9])[0] > 10:
                kwargs = dict(figsize=(10, 5))
            if ax is None:
                fig, ax = plt.subplots(**kwargs)
            else:
                fig = ax.get_figure()
            ax.set_title(f"{name} averages")
            ax.set_xlabel(name)
            ax.set_ylabel("Prediction")

            ax2 = ax.twinx()  # second axis for histogram
            ax2.set_ylabel("Frequency")

            # get values for average plot and histogram
            values_actual = data["average"]["actual"][name].cpu().numpy()
            values_prediction = data["average"]["prediction"][name].cpu().numpy()
            bins = values_actual.size
            support = data["support"][name].cpu().numpy()

            # only display values where samples were observed
            support_non_zero = support > 0
            support = support[support_non_zero]
            values_actual = values_actual[support_non_zero]
            values_prediction = values_prediction[support_non_zero]

            # determine if to display results in log space
            if log_scale is None:
                log_scale = scipy.stats.skew(values_actual) > 1.6

            if log_scale:
                ax.set_yscale("log")

            # plot averages
            if name in self.hparams.x_reals:
                # create x
                if name in to_list(self.dataset_parameters["target"]):
                    if isinstance(self.output_transformer, MultiNormalizer):
                        scaler = self.output_transformer.normalizers[self.dataset_parameters["target"].index(name)]
                    else:
                        scaler = self.output_transformer
                else:
                    scaler = self.dataset_parameters["scalers"][name]
                x = np.linspace(-data["std"], data["std"], bins)
                # reversing normalization for group normalizer is not possible without sample level information
                if not isinstance(scaler, (GroupNormalizer, EncoderNormalizer)):
                    x = scaler.inverse_transform(x)
                    ax.set_xlabel(f"Normalized {name}")

                if len(x) > 0:
                    x_step = x[1] - x[0]
                else:
                    x_step = 1
                x = x[support_non_zero]
                ax.plot(x, values_actual, label="Actual")
                ax.plot(x, values_prediction, label="Prediction")

            elif name in self.hparams.embedding_labels:
                # sort values from lowest to highest
                sorting = values_actual.argsort()
                labels = np.asarray(list(self.hparams.embedding_labels[name].keys()))[support_non_zero][sorting]
                values_actual = values_actual[sorting]
                values_prediction = values_prediction[sorting]
                support = support[sorting]
                # cut entries if there are too many categories to fit nicely on the plot
                maxsize = 50
                if values_actual.size > maxsize:
                    values_actual = np.concatenate([values_actual[: maxsize // 2], values_actual[-maxsize // 2 :]])
                    values_prediction = np.concatenate(
                        [values_prediction[: maxsize // 2], values_prediction[-maxsize // 2 :]]
                    )
                    labels = np.concatenate([labels[: maxsize // 2], labels[-maxsize // 2 :]])
                    support = np.concatenate([support[: maxsize // 2], support[-maxsize // 2 :]])
                # plot for each category
                x = np.arange(values_actual.size)
                x_step = 1
                ax.scatter(x, values_actual, label="Actual")
                ax.scatter(x, values_prediction, label="Prediction")
                # set labels at x axis
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=90)
            else:
                raise ValueError(f"Unknown name {name}")
            # plot support histogram
            if len(support) > 1 and np.median(support) < support.max() / 10:
                ax2.set_yscale("log")
            ax2.bar(x, support, width=x_step, linewidth=0, alpha=0.2, color="k")
            # adjust layout and legend
            fig.tight_layout()
            fig.legend()
            return fig


class AutoRegressiveBaseModel(BaseModel):
    """
    Model with additional methods for autoregressive models.

    Adds in particular the :py:meth:`~decode_autoregressive` method for making auto-regressive predictions.

    Assumes the following hyperparameters:

    Args:
        target (str): name of target variable
        target_lags (Dict[str, Dict[str, int]]): dictionary of target names mapped each to a dictionary of corresponding
            lagged variables and their lags.
            Lags can be useful to indicate seasonality to the models. If you know the seasonalit(ies) of your data,
            add at least the target variables with the corresponding lags to improve performance.
            Defaults to no lags, i.e. an empty dictionary.
    """

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        **kwargs,
    ) -> LightningModule:
        """
        Create model from dataset.

        Args:
            dataset: timeseries dataset
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            LightningModule
        """
        kwargs.setdefault("target", dataset.target)
        # check that lags for targets are the same
        lags = {name: lag for name, lag in dataset.lags.items() if name in dataset.target_names}  # filter for targets
        target0 = dataset.target_names[0]
        lag = set(lags.get(target0, []))
        for target in dataset.target_names:
            assert lag == set(lags.get(target, [])), f"all target lags in dataset must be the same but found {lags}"

        kwargs.setdefault("target_lags", {name: dataset._get_lagged_names(name) for name in lags})
        return super().from_dataset(dataset, **kwargs)

    def output_to_prediction(
        self,
        normalized_prediction_parameters: torch.Tensor,
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        **kwargs,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        """
        Convert network output to rescaled and normalized prediction.

        Function is typically not called directly but via :py:meth:`~decode_autoregressive`.

        Args:
            normalized_prediction_parameters (torch.Tensor): network prediction output
            target_scale (Union[List[torch.Tensor], torch.Tensor]): target scale to rescale network output
            **kwargs: extra arguments for dictionary passed to :py:meth:`~transform_output` method.

        Returns:
            Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]: tuple of rescaled prediction and
                normalized prediction (e.g. for input into next auto-regressive step)
        """
        single_prediction = to_list(normalized_prediction_parameters)[0].ndim == 2
        if single_prediction:  # add time dimension as it is expected
            normalized_prediction_parameters = apply_to_list(normalized_prediction_parameters, lambda x: x.unsqueeze(1))
        # transform into real space
        prediction_parameters = self.transform_output(
            dict(prediction=normalized_prediction_parameters, target_scale=target_scale, **kwargs)
        )
        # todo: handle classification
        # sample value(s) from distribution and  select first sample
        if isinstance(self.loss, DistributionLoss) or (
            isinstance(self.loss, MultiLoss) and isinstance(self.loss[0], DistributionLoss)
        ):
            # todo: handle mixed losses
            prediction = self.loss.sample(prediction_parameters, 1)
        else:
            prediction = prediction_parameters
        # normalize prediction prediction
        normalized_prediction = self.output_transformer.transform(prediction, target_scale=target_scale)
        if isinstance(normalized_prediction, list):
            input_target = torch.cat(normalized_prediction, dim=-1)
        else:
            input_target = normalized_prediction  # set next input target to normalized prediction

        # remove time dimension
        if single_prediction:
            prediction = apply_to_list(prediction, lambda x: x.squeeze(1))
            input_target = input_target.squeeze(1)
        return prediction, input_target

    def decode_autoregressive(
        self,
        decode_one: Callable,
        first_target: Union[List[torch.Tensor], torch.Tensor],
        first_hidden_state: Any,
        target_scale: Union[List[torch.Tensor], torch.Tensor],
        n_decoder_steps: int,
        **kwargs,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Make predictions in auto-regressive manner.

        Supports only continuous targets.

        Args:
            decode_one (Callable): function that takes at least the following arguments:

                * ``idx`` (int): index of decoding step (from 0 to n_decoder_steps-1)
                * ``lagged_targets`` (List[torch.Tensor]): list of normalized targets.
                  List is ``idx + 1`` elements long with the most recent entry at the end, i.e.
                  ``previous_target = lagged_targets[-1]`` and in general ``lagged_targets[-lag]``.
                * ``hidden_state`` (Any): Current hidden state required for prediction.
                  Keys are variable names. Only lags that are greater than ``idx`` are included.
                * additional arguments are not dynamic but can be passed via the ``**kwargs`` argument

                And returns tuple of (not rescaled) network prediction output and hidden state for next
                auto-regressive step.

            first_target (Union[List[torch.Tensor], torch.Tensor]): first target value to use for decoding
            first_hidden_state (Any): first hidden state used for decoding
            target_scale (Union[List[torch.Tensor], torch.Tensor]): target scale as in ``x``
            n_decoder_steps (int): number of decoding/prediction steps
            **kwargs: additional arguments that are passed to the decode_one function.

        Returns:
            Union[List[torch.Tensor], torch.Tensor]: re-scaled prediction (i.e. use ``output_transformation = None``
                when passing on - see :py:meth:`~pytorch_forecasting.models.base_model.BaseModel.transform_output`)

        Example:

            LSTM/GRU decoder

            .. code-block:: python

                def decode(self, x, hidden_state):
                    # create input vector
                    input_vector = x["decoder_cont"].clone()
                    input_vector[..., self.target_positions] = torch.roll(
                        input_vector[..., self.target_positions],
                        shifts=1,
                        dims=1,
                    )
                    # but this time fill in missing target from encoder_cont at the first time step instead of
                    # throwing it away
                    last_encoder_target = x["encoder_cont"][
                        torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
                        x["encoder_lengths"] - 1,
                        self.target_positions.unsqueeze(-1)
                    ].T
                    input_vector[:, 0, self.target_positions] = last_encoder_target

                    if self.training:  # training mode
                        decoder_output, _ = self.rnn(
                            x,
                            hidden_state,
                            lengths=x["decoder_lengths"],
                            enforce_sorted=False,
                        )

                        # from hidden state size to outputs
                        if isinstance(self.hparams.target, str):  # single target
                            output = self.distribution_projector(decoder_output)
                        else:
                            output = [projector(decoder_output) for projector in self.distribution_projector]

                        # predictions are not yet rescaled
                        return return dict(prediction=output, target_scale=x["target_scale"])

                    else:  # prediction mode
                        target_pos = self.target_positions

                        def decode_one(idx, lagged_targets, hidden_state):
                            x = input_vector[:, [idx]]
                            x[:, 0, target_pos] = lagged_targets[-1]  # overwrite at target positions

                            # overwrite at lagged targets positions
                            for lag, lag_positions in lagged_target_positions.items():
                                if idx > lag:  # only overwrite if target has been generated
                                    x[:, 0, lag_positions] = lagged_targets[-lag]

                            decoder_output, hidden_state = self.rnn(x, hidden_state)
                            decoder_output = decoder_output[:, 0]  # take first timestep
                            # from hidden state size to outputs
                            if isinstance(self.hparams.target, str):  # single target
                                output = self.distribution_projector(decoder_output)
                            else:
                                output = [projector(decoder_output) for projector in self.distribution_projector]
                            return output, hidden_state

                        # make predictions which are fed into next step
                        output = self.decode_autoregressive(
                            decode_one,
                            first_target=input_vector[:, 0, target_pos],
                            first_hidden_state=hidden_state,
                            target_scale=x["target_scale"],
                            n_decoder_steps=input_vector.size(1),
                        )

                        # predictions are already rescaled
                        return dict(prediction=output, output_transformation=None, target_scale=x["target_scale"])

        """
        # make predictions which are fed into next step
        output = []
        current_target = first_target
        current_hidden_state = first_hidden_state

        normalized_output = [first_target]

        for idx in range(n_decoder_steps):
            # get lagged targets
            current_target, current_hidden_state = decode_one(
                idx, lagged_targets=normalized_output, hidden_state=current_hidden_state, **kwargs
            )

            # get prediction and its normalized version for the next step
            prediction, current_target = self.output_to_prediction(current_target, target_scale=target_scale)
            # save normalized output for lagged targets
            normalized_output.append(current_target)
            # set output to unnormalized samples, append each target as n_batch_samples x n_random_samples

            output.append(prediction)
        if isinstance(self.hparams.target, str):
            output = torch.stack(output, dim=1)
        else:
            # for multi-targets
            output = [torch.stack([out[idx] for out in output], dim=1) for idx in range(len(self.target_positions))]
        return output

    @property
    def target_positions(self) -> torch.LongTensor:
        """
        Positions of target variable(s) in covariates.

        Returns:
            torch.LongTensor: tensor of positions.
        """
        # todo: expand for categorical targets
        return torch.tensor(
            [0],
            device=self.device,
            dtype=torch.long,
        )

    @property
    def lagged_target_positions(self) -> Dict[int, torch.LongTensor]:
        """
        Positions of lagged target variable(s) in covariates.

        Returns:
            Dict[int, torch.LongTensor]: dictionary mapping integer lags to tensor of variable positions.
        """
        raise Exception(
            "lagged targets can only be used with class inheriting "
            "from AutoRegressiveBaseModelWithCovariates but not from AutoRegressiveBaseModel"
        )


class AutoRegressiveBaseModelWithCovariates(BaseModelWithCovariates, AutoRegressiveBaseModel):
    """
    Model with additional methods for autoregressive models with covariates.

    Assumes the following hyperparameters:

    Args:
        target (str): name of target variable
        target_lags (Dict[str, Dict[str, int]]): dictionary of target names mapped each to a dictionary of corresponding
            lagged variables and their lags.
            Lags can be useful to indicate seasonality to the models. If you know the seasonalit(ies) of your data,
            add at least the target variables with the corresponding lags to improve performance.
            Defaults to no lags, i.e. an empty dictionary.
        static_categoricals (List[str]): names of static categorical variables
        static_reals (List[str]): names of static continuous variables
        time_varying_categoricals_encoder (List[str]): names of categorical variables for encoder
        time_varying_categoricals_decoder (List[str]): names of categorical variables for decoder
        time_varying_reals_encoder (List[str]): names of continuous variables for encoder
        time_varying_reals_decoder (List[str]): names of continuous variables for decoder
        x_reals (List[str]): order of continuous variables in tensor passed to forward function
        x_categoricals (List[str]): order of categorical variables in tensor passed to forward function
        embedding_sizes (Dict[str, Tuple[int, int]]): dictionary mapping categorical variables to tuple of integers
            where the first integer denotes the number of categorical classes and the second the embedding size
        embedding_labels (Dict[str, List[str]]): dictionary mapping (string) indices to list of categorical labels
        embedding_paddings (List[str]): names of categorical variables for which label 0 is always mapped to an
             embedding vector filled with zeros
        categorical_groups (Dict[str, List[str]]): dictionary of categorical variables that are grouped together and
            can also take multiple values simultaneously (e.g. holiday during octoberfest). They should be implemented
            as bag of embeddings
    """

    @property
    def target_positions(self) -> torch.LongTensor:
        """
        Positions of target variable(s) in covariates.

        Returns:
            torch.LongTensor: tensor of positions.
        """
        # todo: expand for categorical targets
        return torch.tensor(
            [self.hparams.x_reals.index(name) for name in to_list(self.hparams.target)],
            device=self.device,
            dtype=torch.long,
        )

    @property
    def lagged_target_positions(self) -> Dict[int, torch.LongTensor]:
        """
        Positions of lagged target variable(s) in covariates.

        Returns:
            Dict[int, torch.LongTensor]: dictionary mapping integer lags to tensor of variable positions.
        """
        # todo: expand for categorical targets
        if len(self.hparams.target_lags) == 0:
            return {}
        else:
            # extract lags which are the same across all targets
            lags = list(next(iter(self.hparams.target_lags.values())).values())
            lag_names = {l: [] for l in lags}
            for targeti_lags in self.hparams.target_lags.values():
                for name, l in targeti_lags.items():
                    lag_names[l].append(name)

            lag_pos = {
                lag: torch.tensor(
                    [self.hparams.x_reals.index(name) for name in to_list(names)],
                    device=self.device,
                    dtype=torch.long,
                )
                for lag, names in lag_names.items()
            }
            return lag_pos
