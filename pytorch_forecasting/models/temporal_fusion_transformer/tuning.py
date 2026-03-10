"""
Hyperparameters can be efficiently tuned with `optuna <https://optuna.readthedocs.io/>`_.
"""

import copy
import logging
import os
from typing import Any, Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import scipy._lib._util
from skbase.utils.dependencies import _check_soft_dependencies
from torch.utils.data import DataLoader

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.tuning.tuner import Tuner

optuna_logger = logging.getLogger("optuna")


# ToDo: remove this once statsmodels release a version compatible with latest
# scipy version
def _lazywhere(cond, arrays, f, fillvalue=np.nan, f2=None):
    """
    Backported lazywhere implementation (basic version).
    """
    arrays = np.broadcast_arrays(*arrays)
    cond = np.array(cond, dtype=bool, copy=False)
    out = np.full(cond.shape, fillvalue)
    if f2 is None:
        out[cond] = f(*[a[cond] for a in arrays])
    else:
        out[cond] = f(*[a[cond] for a in arrays])
        out[~cond] = f2(*[a[~cond] for a in arrays])
    return out


scipy._lib._util._lazywhere = _lazywhere


def optimize_hyperparameters(
    train_dataloaders: DataLoader,
    val_dataloaders: DataLoader,
    model_path: str,
    max_epochs: int = 20,
    n_trials: int = 100,
    timeout: float = 3600 * 8.0,  # 8 hours
    gradient_clip_val_range: tuple[float, float] = (0.01, 100.0),
    hidden_size_range: tuple[int, int] = (16, 265),
    hidden_continuous_size_range: tuple[int, int] = (8, 64),
    attention_head_size_range: tuple[int, int] = (1, 4),
    dropout_range: tuple[float, float] = (0.1, 0.3),
    learning_rate_range: tuple[float, float] = (1e-5, 1.0),
    use_learning_rate_finder: bool = True,
    trainer_kwargs: dict[str, Any] = {},
    log_dir: str = "lightning_logs",
    study=None,
    verbose: int | bool = None,
    pruner=None,
    **kwargs,
):
    """
    Optimize hyperparameters of a Temporal Fusion Transformer model.

    Runs hyperparameter optimization using Optuna. The learning rate
    can optionally be determined using the PyTorch Lightning learning
    rate finder.

    Parameters
    ----------
    train_dataloaders : DataLoader
        Dataloader for training.
    val_dataloaders : DataLoader
        Dataloader for validation.
    model_path : str
        Directory where model checkpoints are saved.
    max_epochs : int, optional
        Maximum number of training epochs. Default is 20.
    n_trials : int, optional
        Number of hyperparameter trials. Default is 100.
    timeout : float, optional
        Maximum time in seconds for optimization. Default is 8 hours.
    gradient_clip_val_range : tuple of float, optional
        Range for gradient clipping values.
    hidden_size_range : tuple of int, optional
        Range for hidden size.
    hidden_continuous_size_range : tuple of int, optional
        Range for hidden continuous size.
    attention_head_size_range : tuple of int, optional
        Range for attention head size.
    dropout_range : tuple of float, optional
        Range for dropout values.
    learning_rate_range : tuple of float, optional
        Range for learning rate.
    use_learning_rate_finder : bool, optional
        Whether to use the Lightning learning rate finder.
    trainer_kwargs : dict of str to Any, optional
        Additional arguments passed to the PyTorch Lightning Trainer.
    log_dir : str, optional
        Directory for TensorBoard logs.
    study : optuna.Study, optional
        Existing Optuna study to resume.
    verbose : int or bool, optional
        Verbosity level.
    pruner : optuna.pruners.BasePruner, optional
        Optuna pruner to use.
    **kwargs
        Additional keyword arguments passed to
        :class:`~pytorch_forecasting.TemporalFusionTransformer`.

    Returns
    -------
    optuna.Study
        The resulting Optuna study.

    Raises
    ------
    ImportError
        If required optional dependencies are not installed.
    """  # noqa : E501
    if not _check_soft_dependencies(["optuna", "statsmodels"], severity="none"):
        raise ImportError(
            "optimize_hyperparameters requires optuna and statsmodels. "
            "Please install these packages with `pip install optuna statsmodels`. "
            "From optuna 3.3.0, optuna-integration is also required."
        )

    import optuna
    from optuna.integration import PyTorchLightningPruningCallback
    import optuna.logging
    import statsmodels.api as sm

    # need to inherit from callback for this to work
    class PyTorchLightningPruningCallbackAdjusted(
        PyTorchLightningPruningCallback, pl.Callback
    ):  # noqa: E501
        pass

    if pruner is None:
        pruner = optuna.pruners.SuccessiveHalvingPruner()

    assert isinstance(train_dataloaders.dataset, TimeSeriesDataSet) and isinstance(
        val_dataloaders.dataset, TimeSeriesDataSet
    ), "dataloaders must be built from timeseriesdataset"

    logging_level = {
        None: optuna.logging.get_verbosity(),
        0: optuna.logging.WARNING,
        1: optuna.logging.INFO,
        2: optuna.logging.DEBUG,
    }
    optuna_verbose = logging_level[verbose]
    optuna.logging.set_verbosity(optuna_verbose)

    loss = kwargs.get(
        "loss", QuantileLoss()
    )  # need a deepcopy of loss as it will otherwise propagate from one trial to the next # noqa : E501

    # create objective function
    def objective(trial: optuna.Trial) -> float:
        # Filenames for each trial must be made unique
        # in order to access each checkpoint.
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(model_path, f"trial_{trial.number}"),
            filename="{epoch}",
            monitor="val_loss",
        )

        learning_rate_callback = LearningRateMonitor()
        logger = TensorBoardLogger(log_dir, name="optuna", version=trial.number)
        gradient_clip_val = trial.suggest_loguniform(
            "gradient_clip_val", *gradient_clip_val_range
        )
        default_trainer_kwargs = dict(
            accelerator="auto",
            max_epochs=max_epochs,
            gradient_clip_val=gradient_clip_val,
            callbacks=[
                learning_rate_callback,
                checkpoint_callback,
                PyTorchLightningPruningCallbackAdjusted(trial, monitor="val_loss"),
            ],
            logger=logger,
            enable_progress_bar=optuna_verbose < optuna.logging.INFO,
            enable_model_summary=[False, True][optuna_verbose < optuna.logging.INFO],
        )
        default_trainer_kwargs.update(trainer_kwargs)
        trainer = pl.Trainer(
            **default_trainer_kwargs,
        )

        # create model
        hidden_size = trial.suggest_int("hidden_size", *hidden_size_range, log=True)
        kwargs["loss"] = copy.deepcopy(loss)
        model = TemporalFusionTransformer.from_dataset(
            train_dataloaders.dataset,
            dropout=trial.suggest_uniform("dropout", *dropout_range),
            hidden_size=hidden_size,
            hidden_continuous_size=trial.suggest_int(
                "hidden_continuous_size",
                hidden_continuous_size_range[0],
                min(hidden_continuous_size_range[1], hidden_size),
                log=True,
            ),
            attention_head_size=trial.suggest_int(
                "attention_head_size", *attention_head_size_range
            ),
            log_interval=-1,
            **kwargs,
        )
        # find good learning rate
        if use_learning_rate_finder:
            lr_trainer = pl.Trainer(
                gradient_clip_val=gradient_clip_val,
                accelerator="auto",
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            tuner = Tuner(lr_trainer)
            res = tuner.lr_find(
                model,
                train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders,
                early_stop_threshold=10000,
                min_lr=learning_rate_range[0],
                num_training=100,
                max_lr=learning_rate_range[1],
            )

            loss_finite = np.isfinite(res.results["loss"])
            if (
                loss_finite.sum() > 3
            ):  # at least 3 valid values required for learning rate finder
                lr_smoothed, loss_smoothed = sm.nonparametric.lowess(
                    np.asarray(res.results["loss"])[loss_finite],
                    np.asarray(res.results["lr"])[loss_finite],
                    frac=1.0 / 10.0,
                )[min(loss_finite.sum() - 3, 10) : -1].T
                optimal_idx = np.gradient(loss_smoothed).argmin()
                optimal_lr = lr_smoothed[optimal_idx]
            else:
                optimal_idx = np.asarray(res.results["loss"]).argmin()
                optimal_lr = res.results["lr"][optimal_idx]
            optuna_logger.info(f"Using learning rate of {optimal_lr:.3g}")
            # add learning rate artificially
            model.hparams.learning_rate = trial.suggest_uniform(
                "learning_rate", optimal_lr, optimal_lr
            )
        else:
            model.hparams.learning_rate = trial.suggest_loguniform(
                "learning_rate", *learning_rate_range
            )

        # fit
        trainer.fit(
            model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders
        )

        # report result
        return trainer.callback_metrics["val_loss"].item()

    # setup optuna and run
    if study is None:
        study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study
