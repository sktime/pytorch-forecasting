"""Extension template for models.

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
    - if the name has more than one word (like Temporal Fusion Transformer), the name of
    the file should be created by separating these words by a underscore (_) and the end
    `_pkg` should be added.
    For eg, for Temporal Fusion Transformer model, the name of the file would be
    temporal_fusion_transformer_pkg.py.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- you can add more private methods, but do not override BaseModel's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by
    pytorch-forecasting.utils._estimator_checks.check_estimator
- once complete: use as a local library, or contribute to pytorch-forecasting via PR

Mandatory methods to implement:
    get_cls - method to access the model class (from MyModel.py).
    get_datamodule_cls - method to access the compatible datamodule class.
    get_test_train_params - method for defining the test fixtures
"""
# todo: write an informative docstring for the file or module, remove the above

from pytorch_forecasting.base._base_pkg import Base_pkg

# todo: add any necessary imports here
# import soft dependencies only inside methods of the class, not at the top of the file
# donot import the model class or the data module class at the module level, it should
# be imported within the respective methods to access those classes (namely get_cla,
# get_datamodule_cls respectively).


# todo: change class name and write docstring
class MyModel_pkg(Base_pkg):
    """Custom Model package container."""

    # todo: fill out estimator tags here
    #  tags are inherited from parent class if they are not set
    _tags = {
        # model name that MUST match the model class name.
        # Valid values: str
        "info:name": "ExampleNetwork",
        # Approximate compute cost.
        # Valid values: int (1 = lightweight e.g. MLP, 3 = medium, 5 = very heavy)
        "info:compute": 2,
        # What type of predictions this model produces.
        # Valid values: list of str, containing one or more of:
        #   "point"     - deterministic point forecasts
        #   "quantile"  - probabilistic quantile forecasts
        #   "distr"     - full predictive distribution (e.g., DeepAR)
        "info:pred_type": ["point"],
        # What type of target the model supports.
        # Valid values: list of str, containing one or more of:
        #   "numeric"   - continuous/numeric target variables
        #   "category"  - categorical target variables (e.g., for classification losses)
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
        # External python packages (not a core dependency)
        # required to run this model (e.g. ["cpflows"]).
        # Delete or keep empty if no external packages are needed.
        # Valid values: list of str
        "python_dependencies": [],
    }

    # we dont need any __init__() for this class

    # implement this is mandatory
    @classmethod
    def get_cls(cls):
        # import the corresponding model class here
        from pytorch_forecasting.models.my_model.model import MyModel

        return MyModel

    # implement this is mandatory
    @classmethod
    def get_datamodule_cls(cls):
        # import the corresponding compatible data module class(es) here
        # Each model class has to be compatible with atleast data module class
        # Please look at pytorch-forecasting.data.data_module folder and find out which
        # data module best suites the requirements of your model implementation
        # If no data module matches the requirements, you might need to implement a new
        # data module. Please look at extension_templates/datamodule.py for more info
        from pytorch_forecasting.data.data_module import CompatibleDatamodule

        return CompatibleDatamodule

    # todo: implement this if this is an estimator contributed to pytorch-forecasting
    #   or to run local automated unit and integration testing of estimator
    #   method should return default parameters, so that a test instance can be created
    # implement this is mandatory
    @classmethod
    def get_test_train_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        # Testing parameter choice should cover internal cases well.
        #
        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as metrics from pytorch-forecasting or sklearn
        #
        # IMPORTANT: all such imports should be *inside get_test_train_params*,
        #            not at the top since imports are used only at testing time
        #
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # IMPORTANT: Always keep the first param as empty dict to test the defaults of
        # the models.
        #
        # Below is an example on how to set params
        #
        # params = [{}, # empty default
        #           {"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        # return params

    # add any other helping method if needed
