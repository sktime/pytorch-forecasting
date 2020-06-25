"""
Hyperparameters can be efficiently tuned with `optuna <https://optuna.readthedocs.io/>`_.
"""
import os
from typing import Dict, Tuple, Any

import optuna
import torch
from pytorch_lightning import Callback
from torch.utils.data import DataLoader

from temporal_fusion_transformer_pytorch.model import TemporalFusionTransformer
from temporal_fusion_transformer_pytorch.data import TimeSeriesDataSet
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def optimize_hyperparameters(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model_path: str,
    max_epochs: int = 20,
    n_trials: int = 100,
    timeout: float = 3600 * 8.0,  # 8 hours
    gradient_clip_val_range: Tuple[float, float] = (0.01, 100.0),
    hidden_size_range: Tuple[int, int] = (16, 265),
    hidden_continuous_size_range: Tuple[int, int] = (8, 64),
    attention_head_size_range: Tuple[int, int] = (1, 4),
    dropout_range: Tuple[float, float] = (0.1, 0.3),
    learning_rate_range: Tuple[float, float] = (0.001, 0.1),
    trainer_kwargs: Dict[str, Any] = {},
    **kwargs,
) -> optuna.Study:
    assert isinstance(train_dataloader.dataset, TimeSeriesDataSet) and isinstance(
        val_dataloader.dataset, TimeSeriesDataSet
    ), "dataloaders must be built from timeseriesdataset"

    # create objective function
    def objective(trial: optuna.Trial) -> float:
        # Filenames for each trial must be made unique in order to access each checkpoint.
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            os.path.join(model_path, "trial_{}".format(trial.number), "{epoch}"), monitor="val_loss"
        )

        # The default logger in PyTorch Lightning writes to event files to be consumed by
        # TensorBoard. We don't use any logger here as it requires us to implement several abstract
        # methods. Instead we setup a simple callback, that saves metrics from each validation step.
        metrics_callback = MetricsCallback()
        trainer = pl.Trainer(
            logger=False,
            checkpoint_callback=checkpoint_callback,
            max_epochs=max_epochs,
            gradient_clip_val=trial.suggest_loguniform("gradient_clip_val", *gradient_clip_val_range),
            gpus=0 if torch.cuda.is_available() else None,
            callbacks=[metrics_callback],
            early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            **trainer_kwargs,
        )

        # create model
        model = TemporalFusionTransformer.from_dataset(
            train_dataloader.dataset,
            dropout=trial.suggest_uniform("dropout", *dropout_range),
            hidden_size=trial.suggest_int("hidden_size", *hidden_size_range, log=True),
            hidden_continuous_size=trial.suggest_int("hidden_continuous_size", *hidden_continuous_size_range, log=True),
            attention_head_size=trial.suggest_int("attention_head_size", *attention_head_size_range),
            learning_rate=trial.suggest_loguniform("learning_rate", *learning_rate_range),
            log_interval=-1,
            **kwargs,
        )
        # fit
        trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

        # report result
        return metrics_callback.metrics[-1]["val_loss"].item()

    # setup optuna and run
    pruner = optuna.pruners.SuccessiveHalvingPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study
