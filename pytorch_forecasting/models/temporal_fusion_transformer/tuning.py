"""
Hyperparameters can be efficiently tuned with `optuna <https://optuna.readthedocs.io/>`_.
"""
import os
from typing import Dict, Tuple, Any

import optuna
import torch
import numpy as np
import statsmodels.api as sm
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback, TensorBoardCallback


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
    trainer_kwargs: Dict[str, Any] = {},
    log_dir: str = "lightning_logs",
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
        learning_rate_callback = LearningRateLogger()
        logger = TensorBoardLogger(log_dir, name="optuna", version=trial.number)
        gradient_clip_val = trial.suggest_loguniform("gradient_clip_val", *gradient_clip_val_range)
        trainer = pl.Trainer(
            checkpoint_callback=checkpoint_callback,
            max_epochs=max_epochs,
            gradient_clip_val=gradient_clip_val,
            gpus=[0] if torch.cuda.is_available() else None,
            callbacks=[metrics_callback, learning_rate_callback],
            early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            logger=logger,
            **trainer_kwargs,
        )

        # create model
        hidden_size = trial.suggest_int("hidden_size", *hidden_size_range, log=True)
        model = TemporalFusionTransformer.from_dataset(
            train_dataloader.dataset,
            dropout=trial.suggest_uniform("dropout", *dropout_range),
            hidden_size=hidden_size,
            hidden_continuous_size=trial.suggest_int(
                "hidden_continuous_size",
                hidden_continuous_size_range[0],
                min(hidden_continuous_size_range[1], hidden_size),
                log=True,
            ),
            attention_head_size=trial.suggest_int("attention_head_size", *attention_head_size_range),
            log_interval=-1,
            **kwargs,
        )
        # find good learning rate
        if "learning_rate" not in kwargs or isinstance(kwargs["learning_rate"], (tuple, list)):
            lr_trainer = pl.Trainer(
                gradient_clip_val=gradient_clip_val,
                gpus=[0] if torch.cuda.is_available() else None,
                logger=False,
            )
            res = lr_trainer.lr_find(
                model,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloader,
                early_stop_threshold=10000.0,
                min_lr=kwargs.get("learning_rate", [1e-5, 1.0])[0],
                num_training=100,
                max_lr=kwargs.get("learning_rate", [1e-5, 1.0])[1],
            )

            loss_finite = np.isfinite(res.results["loss"])
            lr_smoothed, loss_smoothed = sm.nonparametric.lowess(
                np.asarray(res.results["loss"])[loss_finite],
                np.asarray(res.results["lr"])[loss_finite],
                frac=1.0 / 10.0,
            )[10:-1].T
            optimal_idx = np.gradient(loss_smoothed).argmin()
            optimal_lr = lr_smoothed[optimal_idx]
            print(f"using learning rate of {optimal_lr:.3g}")
            model.hparams.learning_rate = optimal_lr
        # fit
        trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

        # report result
        return metrics_callback.metrics[-1]["val_loss"].item()

    # setup optuna and run
    pruner = optuna.pruners.SuccessiveHalvingPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study
