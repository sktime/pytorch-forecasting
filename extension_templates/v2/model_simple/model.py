"""Extension template for a custom v2 forecasting model.

Purpose:
    Provide a fill-in template for implementing a custom forecasting model.
    This is not a concrete model or a base class to import.

How to use this template:
- Copy this file into an importable package and rename the class.
- Replace the example parameters with typed, documented options for your
  architecture.
- Complete the TODOs and document the implemented model and its purpose.
- Initialize architecture-specific layers after reading required dimensions from
  ``metadata``.
- Add helper methods for architecture-specific logic. Do not override private
  ``BaseModel`` methods other than the required ``_pkg`` linkage; prefix custom
  helper methods with ``_custom`` where practical.
- Import optional dependencies inside methods rather than at module level.
- Run the v2 forecaster checks after implementing the linked forecaster.

Required interfaces:
    forward - return a dict containing the "prediction" tensor in the shape and
        target layout expected by the selected datamodule and loss.
    _pkg - return the linked BaseForecaster subclass. BaseModel currently exposes
        this through its legacy ``pkg`` class property; the name does not mean that
        this method should return a model-package class. Do not add ``model_pkg.py``
        or a separate package wrapper.

The linked forecaster template is in ``model_forecaster.py``. It replaces the
legacy ``model_pkg.py`` template and defines the estimator and data interfaces.
"""

from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import Metric
from pytorch_forecasting.models.base._base_model_v2 import BaseModel

# TODO: Add imports required by your architecture.
# Import optional dependencies inside methods rather than at module level.


# TODO: Rename the class and document the implemented model.
class MyModel(BaseModel):
    """Custom model template.
    TODO: Replace this placeholder with an informative docstring.

    TODO: Describe the architecture, its forecasting purpose, and any important
    constraints on its inputs or predictions.

    Parameters
    ----------
    loss : Metric or nn.Module
        Training loss passed to ``BaseModel``. The forecaster should provide a
        default when appropriate.
    parama, paramb, paramc
        Placeholder architecture options. Replace these with documented,
        appropriately typed parameters for the implemented model.
    metadata : Any, optional
        Data-derived metadata supplied by ``BaseForecaster`` after the
        datamodule is attached to data.
    """

    # TODO: Replace the example parameters with typed, documented model options.
    # Keep the shared loss, logging, optimizer, and scheduler options aligned
    # with BaseModel.__init__.
    def __init__(
        self,
        loss: Metric | nn.Module,
        parama: Any = None,
        paramb: str = "default",
        paramc: Any = None,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        metadata: Any | None = None,
    ):
        # Forward the shared training configuration to BaseModel.
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )
        # Exclude module-valued settings and runtime metadata, as in the TFT v2
        # implementation. Update this list if the constructor changes.
        self.save_hyperparameters(ignore=["loss", "logging_metrics", "metadata"])
        # BaseForecaster derives metadata from the attached datamodule and passes
        # it here. Use required entries to size architecture-specific layers; for
        # direct model construction, pass data_module.metadata explicitly.
        self.metadata = metadata
        self.parama = parama
        self.paramb = paramb
        self.paramc = paramc

        # TODO: Initialize architecture-specific layers using model options and
        # the required dimensions from self.metadata.

    @classmethod
    def _pkg(cls):
        """Return the BaseForecaster subclass linked to this model.

        BaseModel exposes this class through its legacy ``pkg`` linkage; the
        linked class is now a forecaster, not a model-package class.
        """
        # TODO: Update the module path and class name when copying the template.
        from pytorch_forecasting.models.my_model.model_forecaster import (
            MyModelForecaster,
        )

        return MyModelForecaster

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the model forward pass.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Batch inputs produced by the selected datamodule. Keys and tensor
            formats depend on that datamodule.

        Returns
        -------
        dict[str, torch.Tensor]
            Model outputs. The base model and its loss require a ``"prediction"``
            tensor; additional output keys may be used for prediction modes.
        """
        # TODO: Implement the forward pass for the selected datamodule's inputs.
        # Return {"prediction": tensor} for the training loss.
        raise NotImplementedError("Implement the model forward pass.")

    # TODO: Add documented helper methods for architecture-specific logic.
