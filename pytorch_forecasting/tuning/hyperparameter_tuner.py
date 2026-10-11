"""Hyperparameter tuning for v2 forecasters"""

import copy
import inspect
import typing

import lightning.pytorch as pl
from lightning.pytorch.core.datamodule import LightningDataModule
import torch.nn as nn

from pytorch_forecasting.data import TimeSeries
from pytorch_forecasting.models.base._base_model_v2 import BaseModel


class HyperparameterTuner:
    def __init__(
        self,
        model_cls,
        data,
        datamodule_cls: type[LightningDataModule] | None = None,
        **fixed_hparams,
    ):
        """Set up the tuner, build the datamodule once, and validate inputs.

        Parameters
        ----------
        model_cls : type
            The model class directly (e.g., DLinear, TFT).
        data : TimeSeries or LightningDataModule
            Dataset to split and reuse across all trials. If passing a
            TimeSeries, ``datamodule_cls`` must be provided.
        datamodule_cls : type[LightningDataModule], optional
            The DataModule class to construct when ``data`` is a TimeSeries,
            or against which to validate when ``data`` is a prebuilt DataModule.
        **fixed_hparams
            Any model parameter that should stay constant, e.g.
            ``hidden_size=128``.

        Examples
        --------
        Tune TFT with a prebuilt DataModule:

        >>> from pytorch_forecasting.tuning import HyperparameterTuner
        >>> from pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 import(
        ...     TFT,
        ... )

        >>> tuner = HyperparameterTuner(
        ...     model_cls=TFT,
        ...     data=encoder_decoder_dm,
        ...     hidden_size=8,
        ...     attention_head_size=2,
        ...     output_size=1,
        ... )

        >>> study = tuner.optimize(n_trials=20, max_epochs=5)
        >>> print(study.best_trial.params)

        Tune DLinear from a raw TimeSeries dataset:

        >>> from pytorch_forecasting.models.dlinear._dlinear_v2 import DLinear
        >>> from pytorch_forecasting.data.data_module import TslibDataModule

        >>> tuner = HyperparameterTuner(
        ...     model_cls=DLinear,
        ...     data=my_timeseries,
        ...     datamodule_cls=TslibDataModule,
        ... )

        >>> study = tuner.optimize(
        ...     n_trials=50,
        ...     max_epochs=10,
        ...     custom_ranges={"moving_avg": [5, 15, 25]},
        ... )
        >>> print(study.best_trial.params)
        """
        self.model_cls = model_cls
        self.fixed_hparams = fixed_hparams
        self._validate_fixed_hparams()

        # Handle data input
        if isinstance(data, TimeSeries):
            if datamodule_cls is None:
                raise ValueError(
                    "When passing a TimeSeries dataset, 'datamodule_cls' must "
                    "be specified: "
                    "(e.g., datamodule_cls=EncoderDecoderTimeSeriesDataModule)."
                )
            self.datamodule = datamodule_cls(data)
            self.datamodule.setup(stage="fit")

        elif isinstance(data, LightningDataModule):
            if datamodule_cls is not None and not isinstance(data, datamodule_cls):
                raise TypeError(
                    f"Expected datamodule of type {datamodule_cls.__name__}, "
                    f"got {type(data).__name__}."
                )
            data.setup("fit")
            self.datamodule = data

        else:
            raise TypeError(
                f"data must be a TimeSeries dataset or LightningDataModule, "
                f"got {type(data).__name__}."
            )

        self._metadata = self.datamodule.metadata

    def _validate_fixed_hparams(self):
        """Raise early if the user passed a parameter name
        that the model doesn't accept.
        """
        sig = inspect.signature(self.model_cls.__init__)
        valid_param_names = sig.parameters.keys()

        for param_name in self.fixed_hparams:
            if param_name not in valid_param_names and param_name != "self":
                raise ValueError(
                    f"'{param_name}' is not a valid parameter of "
                    f"{self.model_cls.__name__}. "
                    f"Valid parameters: {list(valid_param_names)}"
                )

    def _discover_hyperparameters(self, trial, custom_ranges=None):
        """Build a sampled config dict for one Optuna trial.

        Reads the model's ``__init__`` signature, discovers architecture
        hyperparameters, optimizers, and schedulers, while respecting
        user-fixed parameters and ``custom_ranges``.

        Parameters
        ----------
        trial : optuna.Trial
            Current trial whose ``suggest_*`` methods are called.
        custom_ranges : dict, optional
            Per-param overrides, e.g. ``{"hidden_size": [32, 64]}`` or
            ``{"lr": (1e-4, 1e-2)}``.

        Returns
        -------
        dict
            Sampled hyperparameters for this trial.
        """
        sig = inspect.signature(self.model_cls.__init__)
        sampled_cfg = {}
        custom_ranges = custom_ranges or {}

        for param_name, param_obj in sig.parameters.items():
            if param_name in ["self", "loss", "logging_metrics"]:
                continue

            if param_name in self.fixed_hparams:
                continue

            if param_name == "optimizer":
                optimizers = custom_ranges.get(
                    "optimizer", list(BaseModel._OPTIMIZER_REGISTRY.keys())
                )
                sampled_cfg["optimizer"] = trial.suggest_categorical(
                    "optimizer", optimizers
                )
                continue
            if param_name == "lr_scheduler":
                schedulers = custom_ranges.get(
                    "lr_scheduler", [None] + list(BaseModel._SCHEDULER_REGISTRY.keys())
                )
                sampled_cfg["lr_scheduler"] = trial.suggest_categorical(
                    "lr_scheduler", schedulers
                )
                continue

            if param_name == "optimizer_params":
                lr_range = custom_ranges.get(
                    "lr", custom_ranges.get("optimizer_params")
                )
                if lr_range is not None:
                    if isinstance(lr_range, list):
                        lr = trial.suggest_categorical("lr", lr_range)
                    else:
                        low, high = lr_range
                        lr = trial.suggest_float("lr", low, high, log=True)
                else:
                    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
                sampled_cfg["optimizer_params"] = {"lr": lr}
                continue

            if param_name == "lr_scheduler_params":
                chosen_sched = self.fixed_hparams.get(
                    "lr_scheduler", sampled_cfg.get("lr_scheduler")
                )
                if chosen_sched == "reduce_lr_on_plateau":
                    sampled_cfg["lr_scheduler_params"] = {
                        "patience": trial.suggest_int("sched_patience", 2, 10),
                        "factor": trial.suggest_float("sched_factor", 0.1, 0.5),
                    }
                elif chosen_sched == "step_lr":
                    sampled_cfg["lr_scheduler_params"] = {
                        "step_size": trial.suggest_int("sched_step_size", 5, 30),
                        "gamma": trial.suggest_float("sched_gamma", 0.1, 0.5),
                    }
                elif chosen_sched == "cosine_annealing":
                    sampled_cfg["lr_scheduler_params"] = {
                        "T_max": trial.suggest_int("sched_T_max", 5, 50),
                    }
                elif chosen_sched == "cosine_annealing_warm_restarts":
                    sampled_cfg["lr_scheduler_params"] = {
                        "T_0": trial.suggest_int("sched_T_0", 5, 20),
                        "T_mult": trial.suggest_int("sched_T_mult", 1, 3),
                    }
                else:
                    sampled_cfg["lr_scheduler_params"] = {}
                continue

            if param_name in custom_ranges:
                r = custom_ranges[param_name]
                if isinstance(r, list):
                    sampled_cfg[param_name] = trial.suggest_categorical(param_name, r)
                else:
                    low, high = r
                    if (
                        isinstance(param_obj.default, int)
                        and type(param_obj.default) is not bool
                    ):
                        sampled_cfg[param_name] = trial.suggest_int(
                            param_name, low, high
                        )
                    else:
                        sampled_cfg[param_name] = trial.suggest_float(
                            param_name, low, high
                        )
                continue

            default = param_obj.default
            if (
                default is inspect.Parameter.empty
                or default is None
                or isinstance(default, (dict, list))
            ):
                continue

            if type(default) is bool:
                sampled_cfg[param_name] = trial.suggest_categorical(
                    param_name, [True, False]
                )
            elif isinstance(default, int):
                if default <= 8:
                    low = max(1, default - 2)
                    high = default + 2
                    sampled_cfg[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    low = max(1, default // 4)
                    high = default * 4
                    sampled_cfg[param_name] = trial.suggest_int(
                        param_name, low, high, log=True
                    )
            elif isinstance(default, float):
                if default <= 1.0:
                    low = max(0.0, default / 2)
                    high = min(1.0, default * 3) if default > 0.0 else 0.1
                    sampled_cfg[param_name] = trial.suggest_float(param_name, low, high)
                else:
                    low = default / 4
                    high = default * 4
                    sampled_cfg[param_name] = trial.suggest_float(
                        param_name, low, high, log=True
                    )
            elif typing.get_origin(param_obj.annotation) is typing.Literal:
                choices = list(typing.get_args(param_obj.annotation))
                sampled_cfg[param_name] = trial.suggest_categorical(param_name, choices)

        return sampled_cfg

    def optimize(
        self,
        n_trials=100,
        timeout=3600 * 8,
        max_epochs=20,
        custom_ranges=None,
        study=None,
        direction="minimize",
    ):
        """Run the Optuna study.

        Parameters
        ----------
        n_trials : int
            How many trials to run.
        timeout : float
            Wall-clock budget in seconds (default 8 h).
        max_epochs : int
            Training epochs per trial.
        custom_ranges : dict, optional
            Per-param search ranges forwarded to
            :meth:`_discover_hyperparameters`.
        study : optuna.Study, optional
            Existing study to resume; a new one is created if ``None``.
        direction : str
            ``"minimize"`` or ``"maximize"``.

        Returns
        -------
        optuna.Study
            Completed study with all trial results.
        """
        from skbase.utils.dependencies import _safe_import

        optuna = _safe_import("optuna")

        def _objective(trial):
            sampled_cfg = self._discover_hyperparameters(trial, custom_ranges)
            model_cfg = {**sampled_cfg, **self.fixed_hparams}

            if "loss" not in model_cfg:
                model_cfg["loss"] = nn.MSELoss()
            else:
                model_cfg["loss"] = copy.deepcopy(model_cfg["loss"])

            model = self.model_cls(**model_cfg, metadata=self._metadata)

            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator="auto",
                enable_progress_bar=False,
                enable_model_summary=False,
            )

            trainer.fit(model, datamodule=self.datamodule)

            metrics = trainer.callback_metrics
            if "val_loss" not in metrics:
                raise RuntimeError(
                    "No 'val_loss' found in trainer.callback_metrics. "
                    "The tuner requires a validation set to avoid optimizing "
                    "on training loss (which leads to overfitting). "
                    "Ensure your DataModule provides a val_dataloader. "
                    f"Available metric keys: {list(metrics.keys())}"
                )
            return metrics["val_loss"].item()

        if study is None:
            study = optuna.create_study(direction=direction)

        study.optimize(_objective, n_trials=n_trials, timeout=timeout)

        return study
