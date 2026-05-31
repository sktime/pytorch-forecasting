"""Extension template for v1 model package container.

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
    - the name of the file should be prefixed with an underscore and end with ``_pkg``.
    For eg, for ExampleNetwork, the name of the file would be
    _examplenetwork_pkg.py.
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
        "info:name": "ExampleNetwork",  # must match the model class name
        "info:compute": 2,  # 1 = light, 3 = medium, 5 = heavy
        "info:pred_type": ["point"],  # "point", "quantile", or "distr"
        "info:y_type": ["numeric"],  # "numeric" or "category"
        "authors": ["your-github-handle"],
        "capability:exogenous": True,
        "capability:multivariate": True,
        "capability:pred_int": False,
        "capability:flexible_history_length": True,
        "capability:cold_start": False,
        "python_dependencies": [],
    }

    # implement this is mandatory
    @classmethod
    def get_cls(cls):
        """Return the actual Lightning model class."""
        # todo: update the import to point to your model
        # using the complete absolute path.
        # Do NOT use relative imports.
        from extension_templates.v1_network_template.model import (
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
        # Available scenarios from pytorch_forecasting.tests._data_scenarios:
        #   data_with_covariates() + make_dataloaders() - general purpose
        #   dataloaders_fixed_window_without_covariates() - no covariates
        #   dataloaders_with_different_encoder_decoder_length() - var lengths
        #   dataloaders_multi_target() - multivariate
        #
        # If your loss requires specific data transformations,
        # handle them here. See DecoderMLP_pkg for a reference.

        data_loader_kwargs = params.get("data_loader_kwargs", {})

        from pytorch_forecasting.tests._data_scenarios import (
            data_with_covariates,
            make_dataloaders,
        )

        dwc = data_with_covariates()
        dwc.assign(target=lambda x: x.volume)

        dl_default_kwargs = dict(
            target="target",
            time_varying_known_reals=["price_actual"],
            time_varying_unknown_reals=["target"],
            static_categoricals=["agency"],
            add_relative_time_idx=True,
        )
        dl_default_kwargs.update(data_loader_kwargs)
        dataloaders = make_dataloaders(dwc, **dl_default_kwargs)
        return dataloaders
