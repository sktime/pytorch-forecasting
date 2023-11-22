"""
Hyperparameters can be efficiently tuned with `optuna <https://optuna.readthedocs.io/>`_.
"""

__all__ = ["optimize_hyperparameters"]

from loguru import logger
import copy
import logging
import os
from typing import Any, Dict, Tuple, Union, Optional, Callable, Type, Sequence

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.tuner.lr_finder import _LRFinder
import numpy as np
import optuna
from optuna import Trial
from optuna.integration import PyTorchLightningPruningCallback
import optuna.logging
import statsmodels.api as sm
from torch import Tensor
from torch.utils.data import DataLoader

from pytorch_forecasting import TemporalFusionTransformer, BaseModel
from pytorch_forecasting.data import TimeSeriesDataSet

optuna_logger = logging.getLogger("optuna")


NUMBER = Union[float, int]


class PyTorchLightningPruningCallbackAdjusted(pl.Callback, PyTorchLightningPruningCallback):  # type: ignore
    """Need to inherit from callback for this to work."""


def input_params_generator_tft(
    trial: Trial,
    hidden_size_range: Tuple[int, int] = (16, 265),
    hidden_continuous_size_range: Tuple[int, int] = (8, 64),
    attention_head_size_range: Tuple[int, int] = (1, 4),
    dropout_range: Tuple[float, float] = (0.1, 0.3),
) -> dict:
    """Generates parameters for `TemporalFusionTransformer`."""
    hidden_size = trial.suggest_int("hidden_size", *hidden_size_range, log=True)
    dropout = trial.suggest_uniform("dropout", *dropout_range)
    hidden_continuous_size = trial.suggest_int(
        "hidden_continuous_size",
        hidden_continuous_size_range[0],
        min(hidden_continuous_size_range[1], hidden_size),
        log=True,
    )
    attention_head_size = trial.suggest_int("attention_head_size", *attention_head_size_range)
    params = dict(
        hidden_size=hidden_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        attention_head_size=attention_head_size,
    )
    return params


def optimize_hyperparameters(
    train_dataloaders: DataLoader,
    val_dataloaders: DataLoader,
    model_path: str = "hpo",
    monitor: str = "val_loss",
    direction: str = "minimize",
    model_class: Type[BaseModel] = TemporalFusionTransformer,
    max_epochs: int = 20,
    n_trials: int = 100,
    timeout: float = 3600 * 8.0,  # 8 hours
    gradient_clip_val_range: Tuple[float, float] = (0.01, 100.0),
    input_params: Dict[str, Dict[str, Any]] = None,
    input_params_generator: Callable = None,
    generator_params: dict = None,
    learning_rate_range: Tuple[float, float] = (1e-5, 1.0),
    use_learning_rate_finder: bool = True,
    trainer_kwargs: Dict[str, Any] = None,
    log_dir: str = "lightning_logs",
    study: optuna.Study = None,
    verbose: Union[int, bool] = None,
    pruner: optuna.pruners.BasePruner = optuna.pruners.SuccessiveHalvingPruner(),
    **kwargs: Any,
) -> optuna.Study:
    """
    Optimize hyperparameters. Run hyperparameter optimization. Learning rate for is determined with the PyTorch Lightning learning rate finder.

    Args:
        train_dataloaders (DataLoader):
            Dataloader for training model.
        val_dataloaders (DataLoader):
            Dataloader for validating model.
        model_path (str):
            Folder to which model checkpoints are saved.
        monitor (str):
            Metric to return. The hyper-parameter (HP) tuner trains a model for a certain HP config, and reads this metric to score configuration. By default, the lower the better.
        direction (str):
            By default, direction is "minimize", meaning that lower values of the specified `monitor` are better. You can change this, e.g. to "maximize".
        max_epochs (int, optional):
            Maximum number of epochs to run training. Defaults to 20.
        n_trials (int, optional):
            Number of hyperparameter trials to run. Defaults to 100.
        timeout (float, optional):
            Time in seconds after which training is stopped regardless of number of epochs or validation metric. Defaults to 3600*8.0.
        input_params (dict, optional):
            A dictionary, where each `key` contains another dictionary with two keys: `"method"` and `"ranges"`. Example:
                >>> {"hidden_size": {
                >>>     "method": "suggest_int",
                >>>     "ranges": (16, 265),
                >>> }}
            The method key has to be a method of the `optuna.Trial` object. The ranges key are the input ranges for the specified method.
        input_params_generator (Callable, optional):
            A function with the following signature: `fn(trial: optuna.Trial, **kwargs: Any) -> Dict[str, Any]`, returning the parameter values to set up your model for the current trial/run.
            Example:
                >>> def fn(trial: optuna.Trial, param_ranges: Tuple[int, int] = (16, 265)) -> Dict[str, Any]:
                >>>     param = trial.suggest_int("param", *param_ranges, log=True)
                >>>     model_params = {"param": param}
                >>>     return model_params
            Then, when your model is created (before training it and report the metrics for the current combination of hyperparameters), these dictionary is used as follows:
                >>> model = YourModelClass.from_dataset(
                >>>     train_dataloaders.dataset,
                >>>     log_interval=-1,
                >>>     **model_params,
                >>> )
        generator_params (dict, optional):
            The additional parameters to be passed to the `input_params_generator` function, if required.
        learning_rate_range (Tuple[float, float], optional):
            Learning rate range. Defaults to (1e-5, 1.0).
        use_learning_rate_finder (bool):
            If to use learning rate finder or optimize as part of hyperparameters. Defaults to True.
        trainer_kwargs (Dict[str, Any], optional):
            Additional arguments to the `PyTorch Lightning trainer <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html>` such as `limit_train_batches`. Defaults to {}.
        log_dir (str, optional):
            Folder into which to log results for tensorboard. Defaults to "lightning_logs".
        study (optuna.Study, optional):
            Study to resume. Will create new study by default.
        verbose (Union[int, bool]):
            Level of verbosity.
                * None: no change in verbosity level (equivalent to verbose=1 by optuna-set default).
                * 0 or False: log only warnings.
                * 1 or True: log pruning events.
                * 2: optuna logging level at debug level.
            Defaults to None.
        pruner (optuna.pruners.BasePruner, optional):
            The optuna pruner to use. Defaults to `optuna.pruners.SuccessiveHalvingPruner()`.
        **kwargs:
            Additional arguments for your model's class.

    Returns:
        optuna.Study: optuna study results
    """
    assert isinstance(train_dataloaders.dataset, TimeSeriesDataSet) and isinstance(
        val_dataloaders.dataset, TimeSeriesDataSet
    ), "Dataloaders must be built from TimeSeriesDataSet."

    logging_level = {
        None: optuna.logging.get_verbosity(),
        0: optuna.logging.WARNING,
        1: optuna.logging.INFO,
        2: optuna.logging.DEBUG,
    }
    optuna_verbose = logging_level[verbose]
    optuna.logging.set_verbosity(optuna_verbose)

    # need a deepcopy of loss as it will otherwise propagate from one trial to the next
    loss = kwargs.get("loss", None)

    # create objective function
    def objective(trial: optuna.Trial) -> float:
        # Filenames for each trial must be made unique in order to access each checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(model_path, f"trial_{trial.number}"),
            filename="{epoch}",
            monitor=monitor,
        )
        # Create Trainer
        learning_rate_callback = LearningRateMonitor()
        gradient_clip_val = trial.suggest_loguniform("gradient_clip_val", *gradient_clip_val_range)
        default_trainer_kwargs = dict(
            accelerator="auto",
            max_epochs=max_epochs,
            gradient_clip_val=gradient_clip_val,
            callbacks=[
                learning_rate_callback,
                checkpoint_callback,
                PyTorchLightningPruningCallbackAdjusted(trial, monitor=monitor),
            ],
            logger=TensorBoardLogger(log_dir, name="optuna", version=trial.number),
            enable_progress_bar=optuna_verbose < optuna.logging.INFO,
            enable_model_summary=[False, True][optuna_verbose < optuna.logging.INFO],
        )
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        trainer = pl.Trainer(**default_trainer_kwargs)  # type: ignore
        # Create model: set up kwargs
        if input_params_generator is None:
            assert (
                input_params is not None
            ), "Please provide `input_params` when not providing a `input_params_generator` function."
            params = dict()
            for key, cfg in input_params.items():
                fn, low, high, more_kwargs = extract_params(trial, cfg)
                try:
                    params[key] = fn(key, low, high, **more_kwargs)
                except ValueError as ex:
                    raise ValueError(f"Error while calling {fn} for {key}.") from ex
        else:
            if generator_params is None:
                generator_params = {}
            params = input_params_generator(trial, **generator_params)
        kwargs.update(params)
        kwargs["loss"] = copy.deepcopy(loss)
        # Create model
        logger.trace(f"Creating model with {kwargs.keys()}")
        model = model_class.from_dataset(
            train_dataloaders.dataset,
            log_interval=-1,
            **kwargs,
        )
        # find a good learning rate
        if use_learning_rate_finder:
            lr_trainer = pl.Trainer(
                gradient_clip_val=gradient_clip_val,
                accelerator=trainer_kwargs.get("accelerator", "auto"),
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            tuner = Tuner(lr_trainer)
            res: Optional[_LRFinder] = tuner.lr_find(
                model,
                train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders,
                early_stop_threshold=10000,
                min_lr=learning_rate_range[0],
                num_training=100,
                max_lr=learning_rate_range[1],
            )
            assert res is not None, "`tuner.lr_find()` return no results."
            loss_finite = np.isfinite(res.results["loss"])
            if loss_finite.sum() > 3:  # at least 3 valid values required for learning rate finder
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
            model.hparams.learning_rate = trial.suggest_uniform("learning_rate", optimal_lr, optimal_lr)
        else:
            model.hparams.learning_rate = trial.suggest_loguniform("learning_rate", *learning_rate_range)
        # fit
        trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)
        # report result: choose logged metric
        metrics: dict = trainer.callback_metrics
        logger.trace(f"Available metrics: {metrics.keys()}")
        try:
            metric_value: Tensor = metrics[monitor]
        except KeyError as ex:
            raise KeyError(f"Available metrics: {metrics.keys()}") from ex
        return metric_value.item()

    # setup optuna and run
    if study is None:
        study = optuna.create_study(direction=direction, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study


def extract_params(
    trial: Trial,
    cfg: Dict[str, Any],
) -> Tuple[Callable, NUMBER, NUMBER, Dict[str, Any]]:
    """Helper to extract config for one hp."""
    more_kwargs: Dict[str, Any] = {}
    for k, v in cfg.items():
        if k.lower() in ["method"]:
            method: str = cfg["method"]
            assert isinstance(method, str), f"You must provide a {str} as method."
            fn: Callable = getattr(trial, method)
        elif k.lower() in ["ranges"]:
            ranges: Sequence[NUMBER] = cfg["ranges"]
            assert isinstance(ranges, (list, tuple)), f"You must provide a {list} or {tuple} as ranges."
            assert len(ranges) == 2, f"Why did you provide {len(ranges)} values? Only provide 2."
            low = ranges[0]
            high = ranges[1]
        else:
            more_kwargs[k.lower()] = v
    return fn, low, high, more_kwargs
