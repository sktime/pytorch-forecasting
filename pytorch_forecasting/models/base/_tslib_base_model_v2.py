"""
Experimental implementation of a base class for `tslib` models.
NOTE: This PR is stacked on the PR #1812(phoeenniixx).
"""

from typing import Dict, List, Optional, Tuple, Union
from warnings import warn

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_forecasting.models.base import BaseModel


class TslibBaseModel(BaseModel):
    """
    Base class for `tslib` models.

    Parameters
    ----------
    loss : nn.Module
        Loss function to use for training.
    logging_metrics : Optional[List[nn.Module]], optional
        List of metrics to log during training, validation, and testing.
    optimizer : Optional[Union[Optimizer, str]], optional
        Optimizer to use for training.
    optimizer_params : Optional[Dict], optional
        Parameters for the optimizer.
    lr_scheduler : Optional[str], optional
        Learning rate scheduler to use.
    lr_scheduler_params : Optional[Dict], optional
        Parameters for the learning rate scheduler.
    metadata : Optional[Dict], default=None
        Metadata for the model from TslibDataModule.
    """

    def __init__(
        self,
        loss: nn.Module,
        logging_metrics: Optional[List[nn.Module]] = None,
        optimizer: Optional[Union[Optimizer, str]] = "adam",
        optimizer_params: Optional[Dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        self.metadata = metadata or {}
        self.model_name = self.__class__.__name__

        warn(
            f"The Model '{self.model_name}' is part of an experimental implementation"
            "of the pytorch-forecasting model layer for Time Series Library, scheduled"
            "for release with v2.0.0. The API is not stable"
            "and may change without prior warning. This class is intended for beta"
            "testing, not for stable production use.",
            UserWarning,
        )

        self.context_length = self.metadata.get("context_length", 0)
        self.prediction_length = self.metadata.get("prediction_length", 0)

        feature_indices = metadata.get("feature_indices", {})
        self.cont_indices = feature_indices.get("continuous", [])
        self.cat_indices = feature_indices.get("categorical", [])
        self.known_indices = feature_indices.get("known", [])
        self.unknown_indices = feature_indices.get("unknown", [])
        self.target_indices = feature_indices.get("target", [])

        feature_dims = metadata.get("n_features", {})
        self.cont_dim = feature_dims.get("continuous", 0)
        self.cat_dim = feature_dims.get("categorical", 0)
        self.static_cat_dim = feature_dims.get("static_categorical", 0)
        self.static_cont_dim = feature_dims.get("static_continuous", 0)
        self.target_dim = feature_dims.get("target", 1)

        self.feature_names = metadata.get("feature_names", {})

    def _init_network(self):
        """
        Initialize the network architecture.
        This method should be implemented in subclasses to define the specific layers
        and sub_modules of the model.
        """
        raise NotImplementedError("Subclasses must implement _init_network method.")

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Parameters
        ----------
        x: Dict[str, torch.Tensor]
            Dictionary containing input tensors.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing output tensors. These can include
            - predictions:
                Prediction_output of shape (batch_size, prediction_length, target_dim)
            - attention_weights: Optionally, output attention weights
        """

        raise NotImplementedError("Subclasses must implement forward method.")

    def predict_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor]],
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

        if "target" in x:
            y_hat["target"] = x["target"]

        return y_hat

    def transform_output(
        self,
        y_hat: Union[
            torch.Tensor, List[torch.Tensor]
        ],  # evidenced from TimeXer implementation - in PR #1797  # noqa: E501
        target_scale: Optional[Dict[str, torch.Tensor]],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Transform the output of the model to the original scale.

        Parameters
        ----------
        y_hat : Union[torch.Tensor, List[torch.Tensor]]
            Dictionary containing the model output.
        target_scale : Optional[Dict[str, torch.Tensor]]
            Dictionary containing the target scale for inverse transformation.

        Returns
        -------
        Union[torch.Tensor, List[torch.Tensor]]
            Dictionary containing the transformed output.

        Notes
        -----
        WARNING! : This is a temporary implementation and is meant to be replaced with
        a more robust scaling and normalization module for v2 of PTF.
        """

        scale = None
        center = None

        if "scale" in target_scale and "center" in target_scale:
            scale = target_scale["scale"]
            center = target_scale["center"]
        else:
            raise ValueError("Cannot transform output without scale and center.")

        while scale.dim() < y_hat.dim():
            scale = scale.unsqueeze(0)
            center = center.unsqueeze(0)

        return y_hat * scale + center
