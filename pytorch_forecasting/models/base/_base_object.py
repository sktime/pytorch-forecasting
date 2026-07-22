"""Base Classes for pytorch-forecasting models, skbase compatible for indexing."""

import inspect

from pytorch_forecasting.base._base_object import _BaseObject
from pytorch_forecasting.tuning.search_range import _SearchRange


class _BasePtForecaster_Common(_BaseObject):
    """Base class for all PyTorch Forecasting forecaster packages.

    This class points to model objects and contains metadata as tags.
    """

    @classmethod
    def get_cls(cls):
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
        if not isinstance(param_list, dict | list):
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


class _BasePtForecaster(_BasePtForecaster_Common):
    """Base class for PyTorch Forecasting v1 forecasters."""

    _tags = {
        "object_type": ["forecaster_pytorch", "forecaster_pytorch_v1"],
    }


class _BasePtForecasterV2(_BasePtForecaster_Common):
    """Base class for PyTorch Forecasting v2 forecasters."""

    _tags = {
        "object_type": "forecaster_pytorch_v2",
        "tunable_params": {
            "optimizer": _SearchRange(
                param_type="categorical", choices=["adam", "adamw"]
            ),
            "optimizer_params.lr": _SearchRange(
                param_type="float", low=1e-5, high=1e-1, log=True
            ),
        },
        "common_params": {
            "hidden_size": _SearchRange(param_type="int", low=16, high=512, log=True),
            "dropout": _SearchRange(param_type="float", low=0.05, high=0.5),
            "dropout_rate": _SearchRange(param_type="float", low=0.05, high=0.5),
            "activation": _SearchRange(
                param_type="categorical", choices=["relu", "gelu"]
            ),
            "n_heads": _SearchRange(param_type="categorical", choices=[1, 2, 4, 8]),
            "attention_head_size": _SearchRange(param_type="int", low=1, high=4),
            "e_layers": _SearchRange(param_type="int", low=1, high=4),
            "num_layers": _SearchRange(param_type="int", low=1, high=4),
            "d_ff": _SearchRange(param_type="int", low=64, high=2048, log=True),
            "d_model": _SearchRange(param_type="int", low=16, high=512, log=True),
            "patch_length": _SearchRange(
                param_type="categorical", choices=[1, 2, 4, 8, 12, 16, 24]
            ),
            "moving_avg": _SearchRange(
                param_type="categorical", choices=[3, 5, 7, 11, 15, 21, 25]
            ),
            "persistence_weight": _SearchRange(param_type="float", low=0.0, high=1.0),
            "factor": _SearchRange(param_type="int", low=1, high=10),
        },
    }
