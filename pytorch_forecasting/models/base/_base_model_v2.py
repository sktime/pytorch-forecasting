########################################################################################
# Disclaimer: This baseclass is still work in progress and experimental, please
# use with care. This class is a basic skeleton of how the base classes may look like
# in the version-2.
########################################################################################


from typing import Optional, Union
from warnings import warn

from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torch.optim import Optimizer


class BaseModel(LightningModule):
    def __init__(
        self,
        loss: nn.Module,
        logging_metrics: Optional[list[nn.Module]] = None,
        optimizer: Optional[Union[Optimizer, str]] = "adam",
        optimizer_params: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[dict] = None,
    ):
        """
        Base model for time series forecasting.

        Parameters
        ----------
        loss : nn.Module
            Loss function to use for training.
        logging_metrics : Optional[List[nn.Module]], optional
            List of metrics to log during training, validation, and testing.
        optimizer : Optional[Union[Optimizer, str]], optional
            Optimizer to use for training.
            Can be a string ("adam", "sgd") or an instance of `torch.optim.Optimizer`.
        optimizer_params : Optional[Dict], optional
            Parameters for the optimizer.
        lr_scheduler : Optional[str], optional
            Learning rate scheduler to use.
            Supported values: "reduce_lr_on_plateau", "step_lr".
        lr_scheduler_params : Optional[Dict], optional
            Parameters for the learning rate scheduler.
        """
        super().__init__()
        self.loss = loss
        self.logging_metrics = logging_metrics if logging_metrics is not None else []
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

    def training_step(
        self, batch: tuple[dict[str, torch.Tensor]], batch_idx: int
    ) -> STEP_OUTPUT:
        """
        Training step for the model.

        Parameters
        ----------
        batch : Tuple[Dict[str, torch.Tensor]]
            Batch of data containing input and target tensors.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        STEP_OUTPUT
            Dictionary containing the loss and other metrics.
        """
        x, y = batch
        y_hat_dict = self(x)
        y_hat = y_hat_dict["prediction"]
        loss = self.loss(y_hat, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_metrics(y_hat, y, prefix="train")
        return {"loss": loss}

    def validation_step(
        self, batch: tuple[dict[str, torch.Tensor]], batch_idx: int
    ) -> STEP_OUTPUT:
        """
        Validation step for the model.

        Parameters
        ----------
        batch : Tuple[Dict[str, torch.Tensor]]
            Batch of data containing input and target tensors.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        STEP_OUTPUT
            Dictionary containing the loss and other metrics.
        """
        x, y = batch
        y_hat_dict = self(x)
        y_hat = y_hat_dict["prediction"]
        loss = self.loss(y_hat, y)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_metrics(y_hat, y, prefix="val")
        return {"val_loss": loss}

    def test_step(
        self, batch: tuple[dict[str, torch.Tensor]], batch_idx: int
    ) -> STEP_OUTPUT:
        """
        Test step for the model.

        Parameters
        ----------
        batch : Tuple[Dict[str, torch.Tensor]]
            Batch of data containing input and target tensors.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        STEP_OUTPUT
            Dictionary containing the loss and other metrics.
        """
        x, y = batch
        y_hat_dict = self(x)
        y_hat = y_hat_dict["prediction"]
        loss = self.loss(y_hat, y)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_metrics(y_hat, y, prefix="test")
        return {"test_loss": loss}

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
        y_hat = self(x)
        return y_hat

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
        if isinstance(self.optimizer, str):
            if self.optimizer.lower() == "adam":
                return torch.optim.Adam(self.parameters(), **self.optimizer_params)
            elif self.optimizer.lower() == "sgd":
                return torch.optim.SGD(self.parameters(), **self.optimizer_params)
            else:
                raise ValueError(f"Optimizer {self.optimizer} not supported.")
        elif isinstance(self.optimizer, Optimizer):
            return self.optimizer
        else:
            raise ValueError(
                "Optimizer must be either a string or "
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
        if self.lr_scheduler.lower() == "reduce_lr_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **self.lr_scheduler_params
            )
        elif self.lr_scheduler.lower() == "step_lr":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, **self.lr_scheduler_params
            )
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
        for metric in self.logging_metrics:
            metric_value = metric(y_hat, y)
            self.log(
                f"{prefix}_{metric.__class__.__name__}",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
