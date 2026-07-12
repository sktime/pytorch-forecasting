"""ForecastingSearchCV: Hyperparameter search with fit/predict interface."""

import copy

from pytorch_forecasting.tuning.hyperparameter_tuner import _HyperparameterTuner


class ForecastingSearchCV:
    """Hyperparameter search that acts as a model after optimization.

    After calling ``fit()``, this object holds the best model and can
    directly ``predict()``, similar to sktime's ForecastingOptunaSearchCV.

    Parameters
    ----------
    pkg_cls : type
        A Base_pkg subclass (e.g., ``TFT_pkg_v2``, ``DLinear_pkg_v2``).
    param_grid : dict[str, SearchRange]
        Search space. Keys are parameter names, values are SearchRange objects.
    n_trials : int, default=50
        Number of optimization trials.
    base_model_cfg : dict, optional
        Fixed model parameters that should NOT be tuned.
    base_trainer_cfg : dict, optional
        Fixed trainer parameters.
    base_datamodule_cfg : dict, optional
        Fixed datamodule parameters.

    Attributes
    ----------
    best_params_ : dict
        Best parameters found during optimization. Set after ``fit()``.
    best_estimator_ : Base_pkg
        Trained model instance with the best parameters. Set after ``fit()``.
    study_ : optuna.Study
        The Optuna study object with all trial results. Set after ``fit()``.

    Examples
    --------
    >>> from pytorch_forecasting.tuning import ForecastingSearchCV, SearchRange
    >>> search = ForecastingSearchCV(
    ...     pkg_cls=DLinear_pkg_v2,
    ...     param_grid={"moving_avg": SearchRange(
    ...     param_type="categorical",
    ...     choices=[3, 5, 7])},
    ...     n_trials=10,
    ... )
    >>> search.fit(train_data)
    >>> predictions = search.predict(test_data)
    """

    def __init__(
        self,
        pkg_cls,
        param_grid,
        n_trials=50,
        base_model_cfg=None,
        base_trainer_cfg=None,
        base_datamodule_cfg=None,
    ):
        self.pkg_cls = pkg_cls
        self.param_grid = param_grid
        self.n_trials = n_trials
        self.base_model_cfg = base_model_cfg or {}
        self.base_trainer_cfg = base_trainer_cfg or {}
        self.base_datamodule_cfg = base_datamodule_cfg or {}

        self.best_params_ = None
        self.best_estimator_ = None
        self.study_ = None

    def fit(self, data, max_epochs=20, timeout=3600 * 8, direction="minimize"):
        """Run optimization and train the best model.

        Parameters
        ----------
        data : TimeSeries or LightningDataModule
            Training data.
        max_epochs : int, default=20
            Max epochs per trial during search.
        timeout : float, default=28800
            Maximum total search time in seconds.
        direction : str, default="minimize"
            Optimization direction ("minimize" or "maximize").

        Returns
        -------
        self
            Returns self for method chaining.
        """

        tuner = _HyperparameterTuner(
            pkg_cls=self.pkg_cls,
            data=data,
            base_model_cfg=self.base_model_cfg,
            base_trainer_cfg=self.base_trainer_cfg,
            base_datamodule_cfg=self.base_datamodule_cfg,
        )

        self.study_ = tuner.optimize(
            n_trials=self.n_trials,
            timeout=timeout,
            max_epochs=max_epochs,
            custom_ranges=self.param_grid,
            direction=direction,
        )

        self.best_params_ = self.study_.best_params

        final_model_cfg = copy.deepcopy(self.base_model_cfg)
        final_model_cfg.update(self.best_params_)

        self.best_estimator_ = self.pkg_cls(
            model_cfg=final_model_cfg,
            trainer_cfg=self.base_trainer_cfg,
            datamodule_cfg=self.base_datamodule_cfg,
        )
        self.best_estimator_.fit(data, save_ckpt=False)

        return self

    def predict(self, data, **kwargs):
        """Generate predictions using the best model found during optimization.

        Parameters
        ----------
        data : TimeSeries, LightningDataModule, or DataLoader
            The data to predict on.
        **kwargs
            Additional arguments passed to the model's predict method.

        Returns
        -------
        dict[str, torch.Tensor]
            Prediction results.
        """
        if self.best_estimator_ is None:
            raise RuntimeError("No model available. Call fit() before predict().")
        return self.best_estimator_.predict(data, **kwargs)
