"""ForecastingSearchCV: Hyperparameter search with fit/predict interface."""

import copy
import inspect

from pytorch_forecasting.tuning.hyperparameter_tuner import _HyperparameterTuner


class ForecastingSearchCV:
    """Hyperparameter search that acts as a model after optimization.

    After calling ``fit()``, this object holds the best model and can
    directly ``predict()``, similar to sktime's ForecastingOptunaSearchCV.

    Parameters
    ----------
    pkg : Base_pkg
        An instantiated package object (e.g., ``DLinear_pkg_v2(...)``).
        The wrapper reads ``model_cfg``, ``trainer_cfg``, and
        ``datamodule_cfg`` directly from this instance.
    param_grid : dict, optional
        Search space overrides. Keys are parameter names, values can be:

        - ``list``: categorical choices, e.g. ``[3, 5, 7]``
        - ``tuple``: numeric range, e.g. ``(16, 512)`` for int,
          ``(0.01, 0.5)`` for float
        - ``_SearchRange``: advanced users can pass structured objects

        If ``None``, the search space is auto-discovered from the model's
        ``__init__`` signature
    n_trials : int, default=50
        Number of optimization trials.

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
    >>> from pytorch_forecasting.tuning import ForecastingSearchCV
    >>> pkg = DLinear_pkg_v2(
    ...     model_cfg={"context_length": 60},
    ...     trainer_cfg={"max_epochs": 30},
    ...     datamodule_cfg={"batch_size": 64},
    ... )
    >>> search = ForecastingSearchCV(pkg=pkg, n_trials=10)
    >>> search.fit(train_data)
    >>> predictions = search.predict(test_data)
    """

    def __init__(
        self,
        pkg,
        param_grid=None,
        n_trials=50,
    ):
        self.pkg_cls = type(pkg)
        self.param_grid = param_grid
        self.n_trials = n_trials
        self.base_model_cfg = pkg.model_cfg
        self.base_trainer_cfg = pkg.trainer_cfg
        self.base_datamodule_cfg = pkg.datamodule_cfg

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
        search_ranges = self._auto_discover_ranges()
        if self.param_grid:
            search_ranges.update(self._parse_param_grid(self.param_grid))

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
            custom_ranges=search_ranges,
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

    def _parse_param_grid(self, param_grid):
        """Converts user's plain Python types into _SearchRange objects.

        Parameters
        ----------
        param_grid : dict
            Keys are hyperparameter names. Values can be lists (for categorical),
            tuples of length 2 (for int/float ranges), or _SearchRange objects.
        """
        from pytorch_forecasting.tuning.search_range import _SearchRange

        parsed_param_grid = {}

        for key, value in param_grid.items():
            if isinstance(value, _SearchRange):
                parsed_param_grid[key] = value
            elif isinstance(value, list):
                parsed_param_grid[key] = _SearchRange(
                    param_type="categorical", choices=value
                )
            elif isinstance(value, tuple) and len(value) == 2:
                low, high = value

                if isinstance(low, bool) or isinstance(high, bool):
                    raise ValueError(
                        f"'{key}' received a boolean tuple {value}. "
                        "To tune a boolean parameter, use a list instead: "
                        f"'{key}': [True, False]"
                    )

                if isinstance(low, int) and isinstance(high, int):
                    parsed_param_grid[key] = _SearchRange(
                        low=low, high=high, param_type="int"
                    )
                else:
                    parsed_param_grid[key] = _SearchRange(
                        low=low, high=high, param_type="float"
                    )
            else:
                raise ValueError(f"Invalid search range for {key}: {value}")

        return parsed_param_grid

    def _auto_discover_ranges(self):
        """Inspect the model class and match params against the class tags.

        Returns
        -------
        dict[str, _SearchRange]
            Auto-discovered search ranges for parameters found in both
            the model's __init__ signature and class tags.
        """

        model_cls = self.pkg_cls.get_cls()
        tunable_params = self.pkg_cls.get_class_tags().get("tunable_params", {})
        common_params = self.pkg_cls.get_class_tags().get("common_params", {})
        sig = inspect.signature(model_cls.__init__)

        model_param_names = [name for name in sig.parameters if name != "self"]
        local_search_space = {**common_params, **tunable_params}
        discovered = {}

        for param_name in model_param_names:
            if param_name in local_search_space:
                discovered[param_name] = local_search_space[param_name]

        return discovered
