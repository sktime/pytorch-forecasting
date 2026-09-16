"""Hyperparameter tuning for v2 forecasters"""

import inspect

from lightning.pytorch.core.datamodule import LightningDataModule
import torch.nn as nn

from pytorch_forecasting.data import TimeSeries


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

        # 1. Validate datamodule_cls if provided
        if datamodule_cls is not None:
            if not (
                isinstance(datamodule_cls, type)
                and issubclass(datamodule_cls, LightningDataModule)
            ):
                raise TypeError(
                    f"datamodule_cls must be a subclass of LightningDataModule, "
                    f"got {datamodule_cls}."
                )

        # 2. Handle data input
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

        Reads the model's ``__init__`` signature, skips base-class and
        user-fixed params, then picks values using ``custom_ranges`` when
        given or simple heuristics based on the default value's type.

        Parameters
        ----------
        trial : optuna.Trial
            Current trial whose ``suggest_*`` methods are called.
        custom_ranges : dict, optional
            Per-param overrides, e.g. ``{"dropout": (0.1, 0.5)}`` or
            ``{"hidden_size": [32, 64, 128]}``.

        Returns
        -------
        dict
            Sampled hyperparameters for this trial.
        """
        import typing

        from pytorch_forecasting.models.base._base_model_v2 import BaseModel

        base_params = set(inspect.signature(BaseModel.__init__).parameters.keys())
        sig = inspect.signature(self.model_cls.__init__)

        sampled_cfg = {}
        custom_ranges = custom_ranges or {}

        for param_name, param_obj in sig.parameters.items():
            if param_name in base_params or param_name == "self":
                continue

            if param_name in self.fixed_hparams:
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
            import copy

            import lightning.pytorch as pl

            sampled_cfg = self._discover_hyperparameters(trial, custom_ranges)

            model_cfg = {**sampled_cfg, **self.fixed_hparams}

            if "loss" not in model_cfg:
                model_cfg["loss"] = nn.MSELoss()
            elif hasattr(model_cfg["loss"], "__deepcopy__"):
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
            for key in ("val_loss", "train_loss_epoch", "train_loss"):
                if key in metrics:
                    return metrics[key].item()
            raise RuntimeError(
                "No loss metric found in trainer.callback_metrics. "
                f"Available keys: {list(metrics.keys())}"
            )

        if study is None:
            study = optuna.create_study(direction=direction)

        study.optimize(_objective, n_trials=n_trials, timeout=timeout)

        return study
