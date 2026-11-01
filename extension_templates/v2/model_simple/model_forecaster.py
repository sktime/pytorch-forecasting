"""Extension template for the forecaster linked to a v2 model.

Purpose of this implementation template:
    quick implementation of new forecasters following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new forecaster:
- make a copy alongside the model implementation and give it a descriptive name.
    For Temporal Fusion Transformer, use a name such as tft_forecaster.py.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- update the tags, parameter descriptions, and module and class docstrings
- import the model and data-module classes inside their respective class methods
- link the model to this forecaster through the model's existing _pkg method
- once complete: use as a local library, or contribute to pytorch-forecasting via PR

Mandatory methods to implement:
    get_cls - return the corresponding model class.
    get_datamodule_cls - return a compatible concrete data-module class.
    get_model_params - map forecaster options to model constructor arguments.
    get_test_train_params - define lightweight configurations for the test fixtures.

BaseForecaster owns fit, predict, data attachment, and model construction. The
model receives data-derived metadata separately; do not put it in model params.
"""

from pathlib import Path
from typing import Any

from lightning import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import Metric
from pytorch_forecasting.models.base._base_forecaster import BaseForecaster


# TODO: Rename the class and replace the placeholder docstring below.
class MyModelForecaster(BaseForecaster):
    """Configure a model and its compatible data module for forecasting.

    TODO: Describe the model's purpose, inputs, and supported prediction types.
    Replace the example architecture parameters with those of the model.

    Parameters
    ----------
    parama : Any, optional
        Example model-specific option.
    paramb : str, default="default"
        Example string option.
    paramc : Any, optional
        Example component or additional model configuration.
    loss : Metric or nn.Module, optional
        Training loss. This example resolves None to MAE in get_model_params.
    logging_metrics : list[nn.Module], optional
        Metrics forwarded to the model for logging.
    optimizer : Optimizer or str, optional, default="adam"
        Optimizer setting forwarded to the model.
    optimizer_params : dict, optional
        Keyword arguments for the optimizer.
    lr_scheduler : str, optional
        Scheduler setting forwarded to the model.
    lr_scheduler_params : dict, optional
        Keyword arguments for the scheduler.
    trainer : Trainer, optional
        Trainer configuration used by BaseForecaster for fitting.
    datamodule : BaseTimeSeriesDataModule, optional
        Configured data module without attached data. If omitted, the base uses
        get_datamodule_cls. Data is attached when fit or predict is called.
    ckpt_path : str or Path, optional
        Checkpoint path handled by BaseForecaster.
    """

    # TODO: Fill out estimator tags for the implemented model.
    # Tags are inherited when not set here; declare only supported capabilities.
    _tags = {
        # Human-readable name that must match the model class name (str).
        "info:name": "MyModel",
        # Approximate compute cost: 1 = lightweight, 3 = medium, 5 = very heavy.
        "info:compute": 2,
        # Prediction types: "point", "quantile", and/or "distr" (list[str]).
        "info:pred_type": ["point"],
        # Supported targets: "numeric" and/or "category" (list[str]).
        "info:y_type": ["numeric"],
        # TODO: Replace with contributor GitHub handles (list[str]).
        "authors": ["your-github-handle"],
        # Whether the model uses exogenous covariates in a non-trivial way.
        "capability:exogenous": True,
        # Whether the model supports multiple target variables.
        "capability:multivariate": True,
        # Whether the model supports probabilistic prediction intervals.
        "capability:pred_int": False,
        # Whether variable-length encoder histories are supported.
        "capability:flexible_history_length": True,
        # Whether predictions can be made without a long observed history.
        "capability:cold_start": False,
        # Optional external Python dependencies; keep empty if none are needed.
        "python_dependencies": [],
    }

    # TODO: Add architecture options with suitable defaults and document them above.
    # Store constructor options as attributes before delegating to the base.
    def __init__(
        self,
        parama: Any = None,
        paramb: str = "default",
        paramc: Any = None,
        loss: Metric | nn.Module | None = None,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        trainer: Trainer | None = None,
        datamodule: Any = None,
        ckpt_path: str | Path | None = None,
    ):
        self.parama = parama
        self.paramb = paramb
        self.paramc = paramc
        self.loss = loss
        self.logging_metrics = logging_metrics
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        super().__init__(trainer=trainer, datamodule=datamodule, ckpt_path=ckpt_path)

    @classmethod
    def get_cls(cls):
        """Return the corresponding neural-network model class.

        TODO: Update the import path and class name for the implemented model.
        Keep the import inside this method to avoid eager model loading.
        """
        from pytorch_forecasting.models.my_model.model import MyModel

        return MyModel

    @classmethod
    def get_datamodule_cls(cls):
        """Return a concrete data-module class compatible with the model.

        TODO: Select a module from pytorch_forecasting.data.data_module whose
        batch keys, target layout, and metadata match the model implementation.
        If none is compatible, adapt data_module.py and _dataset.py first.
        """
        from pytorch_forecasting.data.data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

    def get_model_params(self) -> dict[str, Any]:
        """Map forecaster options to model constructor arguments.

        TODO: Replace the placeholder parameters with the model's actual options.
        Resolve defaults here without overwriting the constructor attributes.

        Returns
        -------
        dict[str, Any]
            Keyword arguments for get_cls(). Do not include trainer, datamodule,
            checkpoint settings, or metadata. BaseForecaster obtains metadata
            from the attached data module and passes it separately to the model.
        """
        from pytorch_forecasting.metrics import MAE

        return {
            "parama": self.parama,
            "paramb": self.paramb,
            "paramc": self.paramc,
            "loss": MAE() if self.loss is None else self.loss,
            "logging_metrics": self.logging_metrics,
            "optimizer": self.optimizer,
            "optimizer_params": self.optimizer_params,
            "lr_scheduler": self.lr_scheduler,
            "lr_scheduler_params": self.lr_scheduler_params,
        }

    @classmethod
    def get_test_train_params(cls) -> list[dict[str, Any]]:
        """Return lightweight forecaster configurations for the v2 test suite.

        TODO: Choose settings that construct valid, small models after adaptation.
        The first dictionary must stay empty so the default configuration is
        tested. Include an alternative configuration to exercise another case.
        Keep training time low so CI does not time out or exhaust worker memory.

        Returns
        -------
        list[dict[str, Any]]
            Constructor arguments used by the v2 test fixtures. Keep imports of
            components needed only for testing inside this method.
        """
        return [{}, {"paramb": "alternate"}]
