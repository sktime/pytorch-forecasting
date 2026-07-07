"""
HyperparameterTuner: Centralized, model-agnostic optimizer.

"""

import copy
import os

import optuna

from pytorch_forecasting.tuning.search_range import SearchRange


class HyperparameterTuner:
    """Model-agnostic hyperparameter optimizer for v2 models.

    Parameters
    ----------
    pkg_cls : type
        The package class (e.g., TimeXer_pkg_v2, DLinear_pkg_v2).
    data : TimeSeries or LightningDataModule
        Training data.
    base_model_cfg : dict, optional
        Fixed model config that won't be tuned.
    base_trainer_cfg : dict, optional
        Fixed trainer config.
    base_datamodule_cfg : dict, optional
        Fixed datamodule config.
    """

    def __init__(
        self,
        pkg_cls,
        data,
        base_model_cfg=None,
        base_trainer_cfg=None,
        base_datamodule_cfg=None,
    ):
        self.pkg_cls = pkg_cls
        self.data = data
        self.base_model_cfg = base_model_cfg or {}
        self.base_trainer_cfg = base_trainer_cfg or {}
        self.base_datamodule_cfg = base_datamodule_cfg or {}

    def _build_trial_config(self, trial, search_ranges):
        """Convert Optuna trial suggestions into model_cfg and trainer_cfg.

        This is the 'translation layer' between Optuna and Base_pkg.
        """
        model_cfg = copy.deepcopy(self.base_model_cfg)
        trainer_cfg = copy.deepcopy(self.base_trainer_cfg)

        for param_name, search_range in search_ranges.items():
            value = search_range.suggest(trial, param_name)

            if param_name == "gradient_clip_val":
                trainer_cfg["gradient_clip_val"] = value
            elif "." in param_name:
                parts = param_name.split(".")
                d = model_cfg
                for part in parts[:-1]:
                    d = d.setdefault(part, {})
                d[parts[-1]] = value
            else:
                model_cfg[param_name] = value

        return model_cfg, trainer_cfg

    def optimize(
        self,
        n_trials=100,
        timeout=3600 * 8,
        max_epochs=20,
        custom_ranges=None,
        study=None,
        direction="minimize",
    ):
        """Run hyperparameter optimization.

        Parameters
        ----------
        n_trials : int
            Number of Optuna trials.
        timeout : float
            Maximum time in seconds.
        max_epochs : int
            Max training epochs per trial.
        custom_ranges : dict[str, SearchRange], optional
            Override or add to auto-discovered ranges.
        study : optuna.Study, optional
            Existing study to resume.

        Returns
        -------
        optuna.Study
            The completed study with results.
        """
        model_cls = self.pkg_cls.get_cls()
        search_ranges = model_cls.get_tuneable_hyperparameters()

        if custom_ranges:
            search_ranges.update(custom_ranges)

        def objective(trial):
            model_cfg, trainer_cfg = self._build_trial_config(trial, search_ranges)
            trainer_cfg.setdefault("max_epochs", max_epochs)
            trainer_cfg.setdefault("enable_progress_bar", False)

            pkg = self.pkg_cls(
                model_cfg=model_cfg,
                trainer_cfg=trainer_cfg,
                datamodule_cfg=self.base_datamodule_cfg,
            )
            pkg.fit(self.data, save_ckpt=False)

            return pkg.trainer.callback_metrics["val_loss"].item()

        if study is None:
            study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        return study
