"""Base Classes for pytorch-forecasting models, skbase compatible for indexing."""

import inspect

from pytorch_forecasting.utils._dependencies import _safe_import

_SkbaseBaseObject = _safe_import("skbase.base.BaseObject", pkg_name="scikit-base")


class _BaseObject(_SkbaseBaseObject):
    pass


class _BasePtForecaster(_BaseObject):
    """Base class for all PyTorch Forecasting forecaster packages.

    This class points to model objects and contains metadata as tags.
    """

    _tags = {
        "object_type": "forecaster_pytorch",
    }

    @classmethod
    def get_model_cls(cls):
        """Get model class."""
        raise NotImplementedError

    @classmethod
    def name(cls):
        """Get model name."""
        name = cls.get_class_tags().get("info:name", None)
        if name is None:
            name = cls.get_model_cls().__name__
        return name

    @classmethod
    def create_test_instance(cls, parameter_set="default"):
        """Construct an instance of the class, using first test parameter set.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        instance : instance of the class with default parameters

        """
        if "parameter_set" in inspect.getfullargspec(cls.get_test_params).args:
            params = cls.get_test_params(parameter_set=parameter_set)
        else:
            params = cls.get_test_params()

        if isinstance(params, list) and isinstance(params[0], dict):
            params = params[0]
        elif isinstance(params, dict):
            pass
        else:
            raise TypeError(
                "get_test_params should either return a dict or list of dict."
            )

        return cls.get_model_cls()(**params)

    @classmethod
    def create_test_instances_and_names(cls, parameter_set="default"):
        """Create list of all test instances and a list of names for them.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        objs : list of instances of cls
            i-th instance is ``cls(**cls.get_test_params()[i])``
        names : list of str, same length as objs
            i-th element is name of i-th instance of obj in tests.
            The naming convention is ``{cls.__name__}-{i}`` if more than one instance,
            otherwise ``{cls.__name__}``
        """
        if "parameter_set" in inspect.getfullargspec(cls.get_test_params).args:
            param_list = cls.get_test_params(parameter_set=parameter_set)
        else:
            param_list = cls.get_test_params()

        objs = []
        if not isinstance(param_list, (dict, list)):
            raise RuntimeError(
                f"Error in {cls.__name__}.get_test_params, "
                "return must be param dict for class, or list thereof"
            )
        if isinstance(param_list, dict):
            param_list = [param_list]
        for params in param_list:
            if not isinstance(params, dict):
                raise RuntimeError(
                    f"Error in {cls.__name__}.get_test_params, "
                    "return must be param dict for class, or list thereof"
                )
            objs += [cls.get_model_cls()(**params)]

        num_instances = len(param_list)
        if num_instances > 1:
            names = [cls.__name__ + "-" + str(i) for i in range(num_instances)]
        else:
            names = [cls.__name__]

        return objs, names
