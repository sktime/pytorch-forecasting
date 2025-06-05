"""
LTSF-DLinear model for Pytorch Forecasting.
-------------------------------------------
"""

################################################
# NOTE: This is an experimental implementation #
# of LTSF-DLinear model. It is an unstable API #
################################################

from typing import Any, Optional, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from pytorch_forecasting.models.base._tslib_base_model_v2 import TslibBaseModel


class DLinearModel(TslibBaseModel):
    """
    DLinear: Decomposition Linear Model for Long-Term Time Series Forecasting.

    DLinear decomposes time series into trend and seasonal components and applies
    separate linear layers to each component. The final prediction is the sum of
    both components.

    Parameters
    ----------
    loss: nn.Module
        Loss function for training step.
    enc_in: int, optional
        Number of input features for the encoder. If not provided,
        it is initialised to the number of continuous features in the dataset.
    moving_avg: int , default=25
        Kernel size for moving average decomposition.
    individual: bool, default=False
        Whether to use individual linear layers for each variate (True) or
        shared layers across all variates (False).
    logging_metrics: Optional[list[nn.Module]], default=None
        List of metrics to log during training, validation, and testing.
    optimizer: Optional[Union[Optimizer, str]], default='adam'
        Optimizer to use for training.
    optimizer_params: Optional[dict], default=None
        Parameters for the optimizer.
    lr_scheduler: Optional[str], default=None
        Learning rate scheduler to use.
    lr_scheduler_params: Optional[dict], default=None
        Parameters for the learning rate scheduler.
    metadata: Optional[dict], default=None
        Metadata for the model from TslibDataModule.

    References
    ----------
    [1] https://arxiv.org/pdf/2205.13504
    [2] https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py

    Notes
    -----
    [1] This implementation supports only continuous features. Categorical variables
        will be accomodated in future versions.
    """

    def __init__(
        self,
        loss: nn.Module,
        enc_in: Optional[int] = None,
        moving_avg: int = 25,
        individual: bool = False,
        logging_metrics: Optional[list[nn.Module]] = None,
        optimizer: Optional[Union[Optimizer, str]] = "adam",
        optimizer_params: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            metadata=metadata,
        )

        warnings.warn(
            "DLinearModel is an experimental model implemented on TslibBaseModelV2. "
            "It is an unstable version and may be subject to unannounced changes. "
            "Please use with caution."
        )

        self.enc_in = enc_in
        self.moving_avg = moving_avg
        self.individual = individual

        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])

        def _init_network(self):
            """
            Initialise the DLinear model network layer components.
            """
