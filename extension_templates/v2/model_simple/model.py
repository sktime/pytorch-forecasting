"""Template for the neural-network model used by a v2 forecaster."""

import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import Metric
from pytorch_forecasting.models.base._base_model_v2 import BaseModel

# todo: add any necessary imports here
# import soft dependencies only inside methods of the class, not at the top of the file


# todo: change class name and write docstring
class MyModel(BaseModel):
    """Custom Model.
    todo: write docstring.

    todo: describe your custom model here

    Parameters
    ----------
    parama : anytype
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default=MyOtherEstimator(foo=42))
        descriptive explanation of paramc
    and so on
    """

    def __init__(
        self,
        loss: Metric | nn.Module,
        parama=None,
        paramb="default",
        paramc=None,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: dict | None = None,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )
        self.save_hyperparameters(
            ignore=["loss", "logging_metrics", "optimizer", "metadata"]
        )
        self.metadata = metadata if metadata is not None else {}
        self.parama = parama
        self.paramb = paramb
        self.paramc = paramc

    @classmethod
    def _pkg(cls):
        """Return the forecaster linked to this model."""
        from pytorch_forecasting.models.my_model.model_forecaster import (
            MyModelForecaster,
        )

        return MyModelForecaster

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the model's forward pass.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Batch inputs produced by the selected data module.
        """
        # TODO: implement the forward pass and return the model outputs.
        raise NotImplementedError("Implement the model forward pass.")
