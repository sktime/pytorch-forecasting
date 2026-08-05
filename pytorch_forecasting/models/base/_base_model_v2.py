########################################################################################
# Disclaimer: This baseclass is still work in progress and experimental, please
# use with care. This class is a basic skeleton of how the base classes may look like
# in the version-2.
########################################################################################


from typing import Any, Optional, Union
from warnings import warn

from lightning import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pytorch_forecasting.callbacks.predict import PredictCallback
from pytorch_forecasting.metrics import Metric, MultiLoss
from pytorch_forecasting.utils._classproperty import classproperty


class BaseModel(LightningModule):
    """Base model for time series forecasting.

    Parameters
    ----------
    loss : Descendants of ``pytorch_forecasting.metrics.Metric`` class
        Loss function to use for training.
    logging_metrics : Optional[List[nn.Module]], optional
        List of metrics to log during training, validation, and testing.
    optimizer : Optional[Union[Optimizer, str, callable]], optional
        Optimizer to use for training.
        Can be a string ("adam", "adamw", "adagrad", "sgd", or any
        ``torch.optim`` class name), a callable returning an optimizer,
        or an instance of ``torch.optim.Optimizer``.
    optimizer_params : Optional[Dict], optional
        Parameters for the optimizer.
    lr_scheduler : Optional[str], optional
        Learning rate scheduler to use.
        Supported values: "reduce_lr_on_plateau", "step_lr",
        "cosine_annealing", "cosine_annealing_warm_restarts".
    lr_scheduler_params : Optional[Dict], optional
        Parameters for the learning rate scheduler.
    """

    _OPTIMIZER_REGISTRY = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "adagrad": torch.optim.Adagrad,
        "sgd": torch.optim.SGD,
    }

    _SCHEDULER_REGISTRY = {
        "reduce_lr_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "step_lr": torch.optim.lr_scheduler.StepLR,
        "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR,
        "cosine_annealing_warm_restarts": (
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        ),
    }

    def __init__(
        self,
        loss: Metric,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
    ):
        super().__init__()
        self.loss = loss
        self.logging_metrics = nn.ModuleList(
            logging_metrics if logging_metrics is not None else []
        )
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = (
            lr_scheduler_params if lr_scheduler_params is not None else {}
        )
        self.model_name = self.__class__.__name__
        warn(
            f"The Model '{self.model_name}' is part of an experimental rework"
            "of the pytorch-forecasting model layer, scheduled for release with v2.0.0."
            " The API is not stable and may change without prior warning. "
            "This class is intended for beta testing and as a basic skeleton, "
            "but not for stable production use. "
            "Feedback and suggestions are very welcome in "
            "pytorch-forecasting issue 1736, "
            "https://github.com/sktime/pytorch-forecasting/issues/1736",
            UserWarning,
        )

    @classproperty
    def pkg(cls):
        """Package class for the model."""
        return cls._pkg()

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : Dict[str, torch.Tensor]
            Dictionary containing input tensors

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing output tensors
        """
        raise NotImplementedError("Forward method must be implemented by subclass.")

    def predict(
        self,
        dataloader: DataLoader,
        mode: str = "prediction",
        return_info: list[str] | None = None,
        mode_kwargs: dict[str, Any] = None,
        trainer_kwargs: dict[str, Any] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Generate predictions for new data using the `lightning.Trainer`.

        Parameters
        ----------
        dataloader : DataLoader
            The dataloader containing the data to predict on.
        mode : str
            The prediction mode ("prediction", "quantiles", or "raw").
        return_info : list[str], optional
            A list of additional information to return.
        mode_kwargs : dict[str, Any]
            Additional arguments for `to_prediction`/`to_quantiles`.
        trainer_kwargs: dict[str, Any]
            Additional arguments for `Trainer`.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary of prediction results.
        """
        trainer_kwargs = trainer_kwargs or {}
        predict_callback = PredictCallback(
            mode=mode, return_info=return_info, mode_kwargs=mode_kwargs
        )

        callbacks = trainer_kwargs.get("callbacks", [])
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        callbacks.append(predict_callback)
        trainer_kwargs["callbacks"] = callbacks

        trainer = Trainer(**trainer_kwargs)
        trainer.predict(self, dataloaders=dataloader)

        return predict_callback.result

    def _convert_output(
        self,
        out: dict[str, Any],
        metric_fn_name: str,
        use_metric: bool = False,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Convert inputs for prediction/quantiles.

        Parameters
        ----------
        out : dict
            Network output dict with key "prediction".
        metric_fn_name : str
            Name of the Metric method to invoke: "to_prediction" or "to_quantiles".
        use_metric : bool, default = False
            If True, use loss metric for conversion.
            If False, take mean over prediction directly.
        """
        pred = self._coerce_y_hat_for_loss(out["prediction"])

        if not use_metric:
            if isinstance(self.loss, MultiLoss):
                return [
                    getattr(Metric, metric_fn_name)(sub_loss, pred[idx], **kwargs)
                    for idx, sub_loss in enumerate(self.loss)
                ]
            pred_out = getattr(Metric, metric_fn_name)(self.loss, pred, **kwargs)
            return pred_out if isinstance(pred_out, (list, tuple)) else [pred_out]

        bound_fn = getattr(self.loss, metric_fn_name)
        pred_out = bound_fn(pred, **kwargs) if kwargs else bound_fn(pred)
        return pred_out if isinstance(pred_out, (list, tuple)) else [pred_out]

    def to_prediction(
        self, out: dict[str, Any], use_metric: bool = False, **kwargs
    ) -> list[torch.Tensor] | torch.Tensor:
        """Converts raw model output to point forecasts.

        Parameters
        ----------
        out : dict
            Network output dict with key ``"prediction"``.
        use_metric : bool, default = False
            If True, use loss metric for conversion.
            If False, take mean over prediction directly.
        **kwargs
            Passed on to the metric's ``to_prediction``.
        """
        return self._convert_output(out, "to_prediction", use_metric, **kwargs)

    def to_quantiles(
        self, out: dict[str, Any], use_metric: bool = False, **kwargs
    ) -> list[torch.Tensor] | torch.Tensor:
        """Converts raw model output to quantile forecasts.

        Parameters
        ----------
        out : dict
            Network output dict.
        use_metric : bool, default = False
            If True, use loss metric for conversion.
            If False, take mean over prediction directly.
        **kwargs
            Passed on to the metric's ``to_quantiles``.
        """
        return self._convert_output(out, "to_quantiles", use_metric, **kwargs)

    def _coerce_targets_for_loss(self, y):
        """Coerce target outputs to match loss function expectations."""
        y_targets, y_weights = y
        if not isinstance(y_targets, (list, tuple)):
            y_targets = [y_targets]
        n_targets = len(y_targets)

        if isinstance(self.loss, MultiLoss):
            if len(self.loss) != n_targets:
                raise ValueError(
                    f"MultiLoss holds {len(self.loss)} metrics but the data "
                    f"provides {n_targets} target(s) - these have to match."
                )
            return list(y_targets), y_weights

        if n_targets > 1:
            raise ValueError(
                f"The data provides {n_targets} targets, which requires the loss "
                f"to be a MultiLoss, but found {self.loss.__class__.__name__}. "
                f"Use MultiLoss([...]) with one metric per target."
            )
        return y_targets[0], y_weights

    def _coerce_y_hat_for_loss(self, y_hat):
        """Coerce the network output to match loss function expectations."""
        if not isinstance(self.loss, MultiLoss):
            if isinstance(y_hat, (list, tuple)):
                if len(y_hat) != 1:
                    raise ValueError(
                        f"The model returned {len(y_hat)} predictions, which "
                        f"requires the loss to be a MultiLoss, but found "
                        f"{self.loss.__class__.__name__}."
                    )
                return y_hat[0]
            return y_hat

        n_metrics = len(self.loss)
        if isinstance(y_hat, (list, tuple)):
            if len(y_hat) != n_metrics:
                raise ValueError(
                    f"MultiLoss holds {n_metrics} metrics but the model returned "
                    f"{len(y_hat)} predictions - these have to match."
                )
            return list(y_hat)

        if n_metrics == 1:
            return [y_hat]
        if y_hat.size(-1) != n_metrics:
            raise ValueError(
                f"MultiLoss holds {n_metrics} metrics, so the model has to return "
                f"one prediction per target, either as a list or as a tensor whose "
                f"last dimension is {n_metrics}, but got shape "
                f"{tuple(y_hat.shape)}."
            )
        return list(torch.split(y_hat, 1, dim=-1))

    def _step(self, batch, batch_idx):
        """Shared step logic for train, val, and test."""
        x, y = batch
        y = self._coerce_targets_for_loss(y)
        y_hat_dict = self(x)
        y_hat = self._coerce_y_hat_for_loss(y_hat_dict["prediction"])
        loss = self.loss(y_hat, y)
        return {"loss": loss, "y_hat": y_hat, "y": y}

    def training_step(self, batch, batch_idx):
        step_out = self._step(batch, batch_idx)

        self.log(
            "train_loss", step_out["loss"], on_step=True, on_epoch=True, prog_bar=True
        )
        self.log_metrics(step_out["y_hat"], step_out["y"], prefix="train")
        return step_out["loss"]

    def validation_step(self, batch, batch_idx):
        step_out = self._step(batch, batch_idx)

        self.log(
            "val_loss", step_out["loss"], on_step=False, on_epoch=True, prog_bar=True
        )
        self.log_metrics(step_out["y_hat"], step_out["y"], prefix="val")
        return step_out

    def test_step(self, batch, batch_idx):
        step_out = self._step(batch, batch_idx)

        self.log("test_loss", step_out["loss"], on_step=False, on_epoch=True)
        self.log_metrics(step_out["y_hat"], step_out["y"], prefix="test")
        return step_out

    def predict_step(
        self,
        batch: tuple[dict[str, torch.Tensor]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """
        Prediction step for the model.

        Parameters
        ----------
        batch : Tuple[Dict[str, torch.Tensor]]
            Batch of data containing input tensors.
        batch_idx : int
            Index of the batch.
        dataloader_idx : int
            Index of the dataloader.

        Returns
        -------
        torch.Tensor
            Predicted output tensor.
        """
        x, _ = batch
        return self(x)

    def configure_optimizers(self) -> dict:
        """
        Configure the optimizer and learning rate scheduler.

        Returns
        -------
        Dict
            Dictionary containing the optimizer and scheduler configuration.
        """
        optimizer = self._get_optimizer()
        if self.lr_scheduler is not None:
            scheduler = self._get_scheduler(optimizer)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                    },
                }
            else:
                return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}

    def _get_optimizer(self) -> Optimizer:
        """
        Get the optimizer based on the specified optimizer name and parameters.

        Returns
        -------
        Optimizer
            The optimizer instance.
        """
        if callable(self.optimizer) and not isinstance(self.optimizer, str):
            return self.optimizer(self.parameters(), **self.optimizer_params)
        elif isinstance(self.optimizer, str):
            name = self.optimizer.lower()
            if name in self._OPTIMIZER_REGISTRY:
                opt_cls = self._OPTIMIZER_REGISTRY[name]
            elif hasattr(torch.optim, self.optimizer):
                opt_cls = getattr(torch.optim, self.optimizer)
            else:
                raise ValueError(f"Optimizer {self.optimizer} not supported.")
            return opt_cls(self.parameters(), **self.optimizer_params)
        elif isinstance(self.optimizer, Optimizer):
            return self.optimizer
        else:
            raise ValueError(
                "Optimizer must be a string, a callable, or "
                "an instance of torch.optim.Optimizer."
            )

    def _get_scheduler(
        self, optimizer: Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Get the lr scheduler based on the specified scheduler name and params.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer instance.

        Returns
        -------
        torch.optim.lr_scheduler._LRScheduler
            The learning rate scheduler instance.
        """
        name = self.lr_scheduler.lower()
        if name in self._SCHEDULER_REGISTRY:
            sched_cls = self._SCHEDULER_REGISTRY[name]
            return sched_cls(optimizer, **self.lr_scheduler_params)
        else:
            raise ValueError(f"Scheduler {self.lr_scheduler} not supported.")

    def log_metrics(
        self, y_hat: torch.Tensor, y: torch.Tensor, prefix: str = "val"
    ) -> None:
        """
        Log additional metrics during training, validation, or testing.

        Parameters
        ----------
        y_hat : torch.Tensor
            Predicted output tensor.
        y : torch.Tensor
            Target output tensor.
        prefix : str
            Prefix for the logged metrics (e.g., "train", "val", "test").
        """
        if not self.logging_metrics:
            return

        target, weight = y

        is_multi = isinstance(self.loss, MultiLoss)
        y_hat_list = y_hat if is_multi else [y_hat]
        target_list = target if is_multi else [target]

        for metric in self.logging_metrics:
            for idx, (yh, yt) in enumerate(zip(y_hat_list, target_list)):
                metric_value = metric(yh, (yt, weight))
                tag = f"target{idx}_" if is_multi else ""
                self.log(
                    f"{tag}{prefix}_{metric.__class__.__name__}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
