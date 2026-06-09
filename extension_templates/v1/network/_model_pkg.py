"""Extension template for v1 model package container.

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
    - the name of the file should be prefixed with an underscore and end with ``_pkg``.
    For eg, for ExampleNetwork, the name of the file would be
    _model_pkg.py.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- change docstrings for functions and the file
- once complete: use as a local library, or contribute to pytorch-forecasting via PR

Mandatory methods to implement:
    get_cls - method to access the model class (from model.py).
    get_base_test_params - method for defining the test fixtures
    _get_test_dataloaders_from - method for creating test dataloaders
"""

# todo: write an informative docstring for the file or module, remove the above

from pytorch_forecasting.models.base._base_object import _BasePtForecaster

# todo: add any necessary imports here
# import soft dependencies only inside methods of the class, not at the top of the file
# do not import the model class at the module level, it should
# be imported within the respective method to access that class (namely get_cls).


# todo: change class name and write docstring
class ExampleNetwork_pkg(_BasePtForecaster):
    """Package container for ExampleNetwork."""

    _tags = {
        # todo: update all tag values to match your model
        #
        # Human-readable model name — MUST match the model class name.
        # Valid values: str
        "info:name": "ExampleNetwork",
        # Approximate compute cost.
        # Valid values: int (1 = lightweight e.g. MLP, 3 = medium, 5 = very heavy)
        "info:compute": 2,
        # What type of predictions this model produces.
        # Valid values: list of str, containing one or more of:
        #   "point"     → deterministic point forecasts
        #   "quantile"  → probabilistic quantile forecasts
        #   "distr"     → full predictive distribution (e.g., DeepAR)
        "info:pred_type": ["point"],
        # What type of target the model supports.
        # Valid values: list of str, containing one or more of:
        #   "numeric"   → continuous/numeric target variables
        #   "category"  → categorical target variables
        "info:y_type": ["numeric"],
        # GitHub usernames of the contributors.
        # Valid values: list of str, containing GitHub handles.
        # todo: replace with your GitHub handle(s)
        "authors": ["your-github-handle"],
        # Whether the model can use exogenous covariates (X).
        # Valid values: bool
        # True  = model uses exogenous variables in a non-trivial way
        # False = model ignores exogenous inputs
        "capability:exogenous": True,
        # Whether the model supports multiple target variables.
        # Valid values: bool
        # True  = multivariate forecasting supported
        # False = univariate target only
        "capability:multivariate": True,
        # Whether the model supports probabilistic prediction intervals.
        # Valid values: bool
        "capability:pred_int": False,
        # Whether the model can work with variable-length encoder history.
        # Valid values: bool
        "capability:flexible_history_length": True,
        # Whether the model can make predictions without long history.
        # Valid values: bool
        "capability:cold_start": False,
        # External python packages required to run this model.
        # Delete or keep empty if no external packages are needed.
        # Valid values: list of str
        "python_dependencies": [],
    }

    # implement this is mandatory
    @classmethod
    def get_cls(cls):
        """Return the actual Lightning model class."""
        # todo: update the import to point to your model
        # using the complete absolute path.
        # Do NOT use relative imports.
        from extension_templates.v1.network.model import (
            ExampleNetwork,
        )

        return ExampleNetwork

    # implement this is mandatory
    @classmethod
    def get_base_test_params(cls):
        """Return testing parameter settings for the trainer.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting"
            test instance.
            ``create_test_instance`` uses the first dictionary
            in ``params`` by default.
        """
        # todo: set the testing parameters for the estimators
        # Testing parameter choice should cover internal cases well.
        #
        # A good parameter set should primarily satisfy two criteria:
        #   1. Low testing time (ideally a few seconds for the entire
        #      test suite). Avoid defaults that result in "big" models.
        #   2. Minimum two parameter sets with different values to
        #      ensure wide code coverage.
        #
        # IMPORTANT: Always keep the first param as empty dict
        # to test the defaults of the model.
        return [
            {},
            {"hidden_size": 8},
        ]

    # implement this is mandatory
    @classmethod
    def _get_test_dataloaders_from(cls, params):
        """Return train and validation dataloaders for testing.

        Parameters
        ----------
        params : dict
            One of the parameter dicts returned by
            ``get_base_test_params``.

        Returns
        -------
        dataloaders : dict
            Dictionary with keys "train", "val", "test" containing
            PyTorch DataLoaders.
        """
        # todo: choose the appropriate data scenario for your model.
        #
        # Choosing a Data Scenario:
        # -------------------------
        # Import from ``pytorch_forecasting.tests._data_scenarios``:
        #
        # - ``data_with_covariates()``:
        #   Small Stallion dataset with real/categorical known/unknown covariates.
        #   Use with ``make_dataloaders(dwc, target=..., ...)``.
        #   Best for: general-purpose models that accept exogenous inputs.
        #
        # - ``dataloaders_fixed_window_without_covariates()``:
        #   Synthetic AR time-series data, returns pre-made dataloaders.
        #   Best for: models that do NOT use covariates (e.g., N-BEATS).
        #
        # - ``dataloaders_with_different_encoder_decoder_length()``:
        #   Pre-made dataloaders with varying sequence lengths.
        #   Best for: testing flexible history length support.
        #
        # - ``dataloaders_multi_target()``:
        #   Pre-made dataloaders with multiple target columns.
        #   Best for: multivariate forecasting models.
        #
        # Loss-specific data handling:
        # ----------------------------
        # Some losses require specific data transformations. For example:
        # - ``NegativeBinomialDistributionLoss`` requires non-negative
        #   integer targets → round the target column.
        # - ``CrossEntropy`` requires a categorical target
        #   → switch the target to a categorical column.
        #
        # See ``DecoderMLP_pkg._get_test_dataloaders_from`` for a reference.

        # Example implementation using covariates:
        # ----------------------------------------
        # data_loader_kwargs = params.get("data_loader_kwargs", {})
        # from pytorch_forecasting.tests._data_scenarios import (
        #     data_with_covariates,
        #     make_dataloaders,
        # )
        # dwc = data_with_covariates()
        # dl_default_kwargs = dict(
        #     target="target",
        #     time_varying_known_reals=["price_actual"],
        #     time_varying_unknown_reals=["target"],
        #     static_categoricals=["agency"],
        #     add_relative_time_idx=True,
        # )
        # dl_default_kwargs.update(data_loader_kwargs)
        # dataloaders = make_dataloaders(dwc, **dl_default_kwargs)
        # return dataloaders

        raise NotImplementedError(
            "Implement _get_test_dataloaders_from() in your custom model pkg"
        )
