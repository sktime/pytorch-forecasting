"""
Package container template for a custom neural network (v1).

This class exposes metadata (tags) and links to the actual model class.
Copy and modify this when adding a new model to pytorch-forecasting.
"""

from pytorch_forecasting.models.base._base_object import _BasePtForecaster


class ExampleNetwork_pkg(_BasePtForecaster):
    """
    Package container for ExampleNetwork.

    This is required for CI discovery, registration, and testing of v1 models.

    TODO for contributors:
    -----------------------
    - Rename this class to ``YourModelName_pkg``
    - Update all ``_tags`` values to match your model
    - Implement ``get_base_test_params()`` with realistic test parameters
    - Implement ``_get_test_dataloaders_from()`` to create test dataloaders
    """

    _tags = {
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
        #   "category"  → categorical target variables (e.g., for classification losses)
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
        # Whether the model supports multiple target variables (multivariate target).
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
        # Whether the model can make predictions without long history (cold start).
        # Valid values: bool
        "capability:cold_start": False,
        # External python packages required to run this model (e.g. ["cpflows"]).
        # Delete or keep empty if no external packages are needed.
        # Valid values: list of str
        "python_dependencies": [],
    }

    @classmethod
    def get_cls(cls):
        """Return the actual Lightning model class.

        CRITICAL DESIGN REQUIREMENTS:
        - The import MUST use the absolute, fully qualified path
          (do NOT use relative imports like `from .model import ...`).
        - Example:
          ``from pytorch_forecasting.models.examplenetwork.model import (``
          ``    ExampleNetwork,``
          ``)``

        Returns
        -------
        class
            The model class (e.g., ``ExampleNetwork``).
        """
        # todo: update the import to point to your model using the
        # complete absolute path
        from extension_templates.v1_network_template.model import ExampleNetwork

        return ExampleNetwork

    @classmethod
    def get_base_test_params(cls):
        """Return testing parameter settings for the trainer.

        CRITICAL DESIGN REQUIREMENTS:
        -----------------------------
        - This method is NOT optional. It is a required test fixture
          for the CI and test runner.
        - It CANNOT return just an empty list `[{}]` or single empty dict.
        - It MUST return a list of dictionaries with realistic test parameter settings.
        - Testing parameter choice should cover internal edge cases
          and hyperparameters well.
        - A good parameter set should primarily satisfy two criteria:
          1. Low testing time: Chosen set of parameters should have a low testing time
             (ideally in the magnitude of a few seconds for the entire test suite).
             Avoid default values that result in "big" models which increase test time
             or risk causing test worker runner crashes/timeouts.
          2. Wide range of coverage: There should be a minimum of two parameter
             sets with different sets of values to ensure a wide range of code path
             coverage.
          3. Avoid external dependencies in defaults: Do not require external
             packages in the default parameter dictionary unless absolutely necessary.
          4. Test defaults: The first test parameter dictionary (or fixture) MUST
             be an empty dictionary ({}) to test the model's default configurations.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
            Each dict represents parameter settings to construct an
            "interesting" test instance.
            `create_test_instance` uses the first dictionary in `params` by default.
        """
        # todo: replace with parameters relevant to your model to test edge cases
        return [
            {},
            {"hidden_size": 8},
        ]

    @classmethod
    def _get_test_dataloaders_from(cls, params):
        """Return train and validation dataloaders for testing.

        This is REQUIRED for v1 models in CI. The method prepares the dataloaders
        used by the CI test runner to check if the model compiles, trains, and
        performs inference properly.

        What you should do here:
        ------------------------
        1. Retrieve custom data-related parameters from the ``params`` dict
           (e.g. custom ``data_loader_kwargs``).
        2. Pick a test data scenario from ``_data_scenarios``.
        3. If needed, adapt the data for your model's loss function
           (see "Loss-specific data handling" below).
        4. Call ``make_dataloaders`` to create and return the dataloaders.

        Choosing a Data Scenario:
        -------------------------
        Import from ``pytorch_forecasting.tests._data_scenarios``:

        - ``data_with_covariates()``:
          Small Stallion dataset with real/categorical known/unknown covariates.
          Use with ``make_dataloaders(dwc, target=..., ...)``.
          Best for: general-purpose models that accept exogenous inputs.

        - ``dataloaders_fixed_window_without_covariates()``:
          Synthetic AR time-series data, returns pre-made dataloaders.
          Best for: models that do NOT use covariates (e.g., N-BEATS).

        - ``dataloaders_with_different_encoder_decoder_length()``:
          Pre-made dataloaders with varying sequence lengths.
          Best for: testing flexible history length support.

        - ``dataloaders_multi_target()``:
          Pre-made dataloaders with multiple target columns.
          Best for: multivariate forecasting models.

        Loss-specific data handling:
        ----------------------------
        Some losses require specific data transformations. If your model's
        ``get_base_test_params`` includes a ``loss`` parameter, you may
        need to adapt the data accordingly. For example:

        - ``NegativeBinomialDistributionLoss`` requires non-negative integer
          targets → round the target column.
        - ``CrossEntropy`` requires a categorical target
          → switch the target to a categorical column.

        See ``DecoderMLP_pkg._get_test_dataloaders_from`` for a reference
        implementation that handles multiple loss types.

        Parameters
        ----------
        params : dict
            One of the parameter dicts returned by ``get_base_test_params``.

        Returns
        -------
        dataloaders : dict
            Dictionary with keys "train", "val", "test" containing
            PyTorch DataLoaders.
        """
        # todo: choose the appropriate data scenario for your model.
        # Example using data_with_covariates + make_dataloaders:
        #
        # data_loader_kwargs = params.get("data_loader_kwargs", {})
        # from pytorch_forecasting.tests._data_scenarios import (
        #     data_with_covariates,
        #     make_dataloaders,
        # )
        #
        # dwc = data_with_covariates()
        #
        # # todo: if your loss requires specific data transformations,
        # # handle them here. For example:
        # # loss = params.get("loss", None)
        # # if isinstance(loss, NegativeBinomialDistributionLoss):
        # #     dwc = dwc.assign(target=lambda x: x.volume.round())
        # # elif isinstance(loss, CrossEntropy):
        # #     data_loader_kwargs["target"] = "agency"
        #
        # dl_kwargs = dict(
        #     target="target",
        #     time_varying_known_reals=["price_actual"],
        #     time_varying_unknown_reals=["target"],
        #     static_categoricals=["agency"],
        #     add_relative_time_idx=True,
        # )
        # dl_kwargs.update(data_loader_kwargs)
        # return make_dataloaders(dwc, **dl_kwargs)
        #
        # Alternatively, for models without covariates:
        # from pytorch_forecasting.tests._data_scenarios import (
        #     dataloaders_fixed_window_without_covariates,
        # )
        # return dataloaders_fixed_window_without_covariates()

        raise NotImplementedError(
            "Implement _get_test_dataloaders_from() for your model. "
            "See the docstring above for guidance on choosing data scenarios."
        )
