"""NBEATSx forecaster."""

from pathlib import Path

from lightning.pytorch import Trainer
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.data.data_module import EncoderDecoderTimeSeriesDataModule
from pytorch_forecasting.models.base._base_forecaster import BaseForecaster
from pytorch_forecasting.models.nbeats._nbeatsx_v2 import NBeatsX_v2


class NBeatsXForecaster(BaseForecaster):
    """NBEATSx point forecasting with static and temporal covariates.

    Fit and predict with a :class:`~pytorch_forecasting.data.TimeSeries` and fixed
    encoder and prediction lengths. With the v2 data pipeline, categorical
    covariates must be numeric codes.

    Parameters
    ----------
    stack_types : list of str, optional
        Stack bases: generic, trend, seasonality, or exogenous. Defaults to
        ``["generic"]``. Exogenous stacks require known temporal covariates.
    num_blocks, num_block_layers, widths : list of int, optional
        Blocks per stack, hidden layers per block, and hidden widths. Defaults
        are 3, 4, and 256. A one-element list applies to every stack.
    sharing : list of bool, optional
        Share weights within each stack. Defaults to false for all stacks.
    expansion_coefficient_lengths : list of int, optional
        Polynomial coefficient count for trend and minimum period for
        seasonality. Defaults to 3 for every stack.
    dropout : float, default=0.1
        Dropout probability.
    backcast_loss_ratio : float, default=0.1
        Weight of the normalized backcast reconstruction error.
    loss : nn.Module, optional
        Single-target point forecasting loss. Defaults to MASE.
    logging_metrics : list of nn.Module, optional
        Additional metrics to log.
    optimizer : Optimizer or str, default="adam"
        Optimizer to use for training.
    optimizer_params : dict, optional
        Arguments for the optimizer.
    lr_scheduler : str, optional
        Learning rate scheduler.
    lr_scheduler_params : dict, optional
        Arguments for the scheduler.
    trainer, datamodule, ckpt_path : optional
        See :class:`~pytorch_forecasting.models.base.BaseForecaster`.

    Examples
    --------
    Fit on a ``TimeSeries`` with known temporal covariates::

        forecaster = NBeatsXForecaster(
            stack_types=["trend", "seasonality", "exogenous"],
            widths=[32],
            datamodule=EncoderDecoderTimeSeriesDataModule(
                max_encoder_length=24,
                max_prediction_length=6,
            ),
            trainer=Trainer(max_epochs=3),
        )
        forecaster.fit(training, save_ckpt=False)
        predictions = forecaster.predict(validation)
    """

    _tags = {
        "info:name": "NBeatsX_v2",
        "info:compute": 1,
        "capability:exogenous": True,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    def __init__(
        self,
        stack_types: list[str] | None = None,
        num_blocks: list[int] | None = None,
        num_block_layers: list[int] | None = None,
        widths: list[int] | None = None,
        sharing: list[bool] | None = None,
        expansion_coefficient_lengths: list[int] | None = None,
        dropout: float = 0.1,
        backcast_loss_ratio: float = 0.1,
        loss: nn.Module | None = None,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        trainer: Trainer | None = None,
        datamodule: EncoderDecoderTimeSeriesDataModule | None = None,
        ckpt_path: str | Path | None = None,
    ):
        self.stack_types = stack_types
        self.num_blocks = num_blocks
        self.num_block_layers = num_block_layers
        self.widths = widths
        self.sharing = sharing
        self.expansion_coefficient_lengths = expansion_coefficient_lengths
        self.dropout = dropout
        self.backcast_loss_ratio = backcast_loss_ratio
        self.loss = loss
        self.logging_metrics = logging_metrics
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        super().__init__(trainer=trainer, datamodule=datamodule, ckpt_path=ckpt_path)

    @classmethod
    def get_cls(cls):
        """Return the Lightning model."""
        return NBeatsX_v2

    @classmethod
    def get_datamodule_cls(cls):
        """Return the encoder-decoder datamodule."""
        return EncoderDecoderTimeSeriesDataModule

    def get_model_params(self):
        """Return parameters for the Lightning model."""
        return {
            name: value
            for name, value in self.get_params(deep=False).items()
            if name not in {"trainer", "datamodule", "ckpt_path"}
        }

    @classmethod
    def get_test_train_params(cls):
        """Return a small configuration for the forecaster test suite."""
        return [
            dict(
                widths=[8],
                num_blocks=[1],
                num_block_layers=[1],
                datamodule_cfg=dict(max_encoder_length=6, max_prediction_length=3),
            )
        ]
