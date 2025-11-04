"""Base Classes for pytorch-forecasting models, skbase compatible for indexing."""

import inspect

from pytorch_forecasting.base._base_object import _BaseObject


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


class _BasePtForecaster(_BasePtForecaster_Common):
    """Base class for PyTorch Forecasting v1 forecasters."""

    _tags = {
        "object_type": ["forecaster_pytorch", "forecaster_pytorch_v1"],
    }


class _BasePtForecasterV2(_BasePtForecaster_Common):
    """Base class for PyTorch Forecasting v2 forecasters."""

    _tags = {
        "object_type": "forecaster_pytorch_v2",
    }


class _EncoderDecoderConfigBase(_BasePtForecasterV2):
    def _check_metadata(self, metadata):
        assert isinstance(metadata, dict)
        required_keys = [
            "encoder_cat",
            "encoder_cont",
            "decoder_cat",
            "decoder_cont",
            "target",
            "max_encoder_length",
            "min_encoder_length",
            "max_prediction_length",
            "min_prediction_length",
            "static_categorical_features",
            "static_continuous_features",
        ]

        for key in required_keys:
            assert key in metadata, f"Key {key} missing in metadata"

        assert metadata["encoder_cat"] >= 0
        assert metadata["encoder_cont"] >= 0
        assert metadata["decoder_cat"] >= 0
        assert metadata["decoder_cont"] >= 0
        assert metadata["target"] > 0


class _TSlibConfigBase(_BasePtForecasterV2):
    def _check_metadata(self, metadata):
        assert isinstance(metadata, dict)
        required_keys = [
            "feature_names",
            "feature_indices",
            "n_features",
            "context_length",
            "prediction_length",
            "freq",
            "features",
        ]

        for key in required_keys:
            assert key in metadata, f"Key {key} missing in metadata"

        assert (
            metadata["n_features"]
            == len(metadata["feature_names"])
            == len(metadata["feature_indices"])
        )
        assert metadata["context_length"] > 0
        assert metadata["prediction_length"] > 0
        assert metadata["freq"] is not None
