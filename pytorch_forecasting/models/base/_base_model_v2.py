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

    def standardize_model_output(
        self,
        prediction: torch.Tensor,
        expected_dims: tuple[int] = None,  # noqa: E501
    ) -> torch.Tensor:
        """
        Standardize model outputs to a 4-dimensional tensor, with shape
        (batch_size, timesteps, num_features, last_dim).

        Parameters
        ----------
        prediction : torch.Tensor
            The raw prediction tensor from the model.
            - Must be a torch.Tensor (in the future, also accept a list of tensors for
              multi-target forecasting).
            - Supported dims: 2D, 3D or 4D tensors.
            - if 2D: (batch_size, timesteps) - univariate forecasting
            - if 3D:
                a) (batch_size, timesteps, n_targets) - multivariate forecasting
                b) (batch_size, timesteps, last_dim) - univariate forecasting with quantiles or distribution.
                c) (batch_size, timesteps, n_targets * last_dim) - multivariate
              forecasting with quantiles, where features and quantiles are flattened in dim 2.
            - if 4D: (batch_size, timesteps, n_targets, last_dim) - multivariate
              forecasting with quantiles or distribution parameters.
            - In the future, once multi-target forecasting with ``MultiLoss`` is supported, this
              will also accept a list of tensors, where each tensor inside the list
              is treated as above. Note: In this case, each tensor in the list
              will have n_targets = 1, as each tensor corresponds to a single target.
            - If anything apart from the above dimensions is provided, an error is raised.

        expected_dims : tuple[int], default= None
        A tuple specifying the dimensions: (batch_size, timesteps, n_targets, last_dim).

            batch_size : Optional[int], default=None
                - Position 1: Expected batch size
                - When specified: Validates prediction.shape[0]
                - When None: Uses actual tensor dimension

            timesteps : Optional[int], default=None
                - Position 2: Expected number of timesteps
                - When specified: Validates prediction.shape[1]
                - When None: Uses actual tensor dimension

            n_targets : int
                - Position 0: Number of target features
                - Must be provided explicitly (cannot be None)
                - Used for reshaping 2D and 3D tensors to 4D.

            last_dim : Optional[int], default=None
                - Position 3: Size of the last dimension.
                - Common use case - quantile, sample, distribution params.
                - When it is specified, it is used to directly reshape.
                - When None and model uses QuantileLoss: It is set to the number of quantiles
                - When None and no quantile information is available: It defaults to 1.
                - If required, this can be extended to handle other cases where the last_dim is None
                but its value can be inferred from the loss function or model configuration (apart from
                the existing QuantileLoss case, of course).
        Returns
        -------
        torch.Tensor
            The standardized prediction tensor with shape (batch_size, timesteps, n_targets, last_dim).
            The prediction tensor is obtained by reshaping the input tensor. There are
            several cases to consider:

            - If the input tensor is 2D, it is reshaped to (batch_size, timesteps, 1, 1).
            - If the input tensor is 3D, it is reshaped to (batch_size, timesteps, 1, 1) for a
            multivariate single-target forecast, or (batch_size, timesteps, 1, last_dim) for a univariate quantile forecast.
            - If the input tensor is 4D, it is assumed to be in the shape
            (batch_size, timesteps, n_targets, last_dim) or (batch_size, timesteps, last_dim, n_targets).
        Notes
        -----
        [1] The fourth dimension (last_dim) commonly represents:

            * Quantiles: For quantile regression (e.g., 0.1, 0.5, 0.9)
            * Distribution parameters: For parametric forecasts (e.g., mean, variance)
            * Samples: For sample-based uncertainty estimates

            The current implementation assumes the most common case of quantile forecasts
            when automatically inferring this dimension from the loss function,
            but any value can be explicitly provided. A fallback of 1 is used in case where
            no information is available on ``last_dim``.

        [2] This can currently handle situations where a single target is used
            either in a univariate or multivariate situation and multiple-targets using the
            same loss function.

            In case of multi-target forecasting with separate loss functions for each target,
            the input tensor is expected to be a list of tensors. This is not yet supported
            in this function, but it is planned for the future.
        """  # noqa: E501

        n_targets, batch_size, timesteps, last_dim = expected_dims

        if not isinstance(prediction, torch.Tensor):
            raise TypeError(
                f"Expected prediction to be a torch.Tensor, but got {type(prediction)}"
            )

        if n_targets is None:
            raise ValueError(
                "Expected n_targets to be a positive integer, but got `None`."
            )

        if last_dim is None:
            if hasattr(self.loss, "quantiles") and self.loss.quantiles is not None:
                last_dim = len(self.loss.quantiles)  # Quantile regression case
            # we can add more cases here in the future, where we refer to the specific
            # loss function to determine the last dimension. For now we are sticking
            # to the quantile regression case.
            else:
                last_dim = 1

        if batch_size is not None:
            if prediction.shape[0] != batch_size:
                raise ValueError(
                    f"Expected batch size {batch_size}, but got {prediction.shape[0]}."
                )

        if timesteps is not None:
            if prediction.shape[1] != timesteps:
                raise ValueError(
                    f"Expected timesteps {timesteps}, but got {prediction.shape[1]}."
                )

        if prediction.ndim == 2:
            # reshape to (batch_size, timsteps, 1, 1)
            prediction = prediction.unsqueeze(-1).unsqueeze(-1)

        elif prediction.ndim == 3:
            if prediction.shape[2] == n_targets:
                # reshape to (batch_size, timesteps, n_targets, 1)
                prediction = prediction.unsqueeze(-1)
            elif prediction.shape[2] == last_dim:
                # reshape to (batch_size, timesteps, 1, last_dim)
                prediction = prediction.unsqueeze(2)
            elif prediction.shape[2] == n_targets * last_dim:
                # multivariate forecast with quantiles
                # where features and quantiles are flattened in dim 2.
                # reshape to (batch_size, timesteps, n_targets, last_dim)
                prediction = prediction.reshape(
                    prediction.shape[0], prediction.shape[1], n_targets, last_dim
                )
            else:
                # reshape to (batch_size, timesteps, n_targets, last_dim)
                prediction = prediction.unsqueeze(-1)

        elif prediction.ndim == 4:
            # assuming only a single case where n_targets and last_dim are swapped.
            if prediction.shape[2] == last_dim and prediction.shape[3] == n_targets:
                # reshape to (batch_size, timesteps, n_targets, last_dim)
                warn(
                    "Prediction tensor has shape (batch_size, timesteps, last_dim, n_targets). "  # noqa: E501
                    "This is not the expected shape. Transposing the last two dimensions."  # noqa: E501
                )
                prediction = prediction.permute(0, 1, 3, 2)

        else:
            raise ValueError(
                f"Expected prediction tensor to have 2, 3, or 4 dimensions, "
                f"but got {prediction.ndim} dimensions."
            )

        # final check to ensure the output is 4D
        if prediction.ndim != 4:
            raise ValueError(
                f"Failed to standardize output to 4D tensor. Current shape: {prediction.shape}"  # noqa: E501
            )

        return prediction
