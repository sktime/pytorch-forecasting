# copyright: pytorch-forecasting developers, BSD-3-Clause License (see LICENSE file)
# mostly based on the sktime utilities of the same name (BSD-3 Clause)
"""Registry and dispatcher for test classes.

Module does not contain tests, only test utilities.
"""

__author__ = ["fkiraly"]

from inspect import isclass


def get_test_class_registry():
    """Return test class registry.

    Wrapped in a function to avoid circular imports.

    Returns
    -------
    testclass_dict : dict
        test class registry
        keys are scitypes, values are test classes TestAll[Scitype]
    """
    from pytorch_forecasting.tests.test_all_estimators import TestAllPtForecasters
    from pytorch_forecasting.tests.test_all_estimators_v2 import TestAllPtForecastersV2

    testclass_dict = dict()
    testclass_dict["forecaster_pytorch_v1"] = TestAllPtForecasters
    testclass_dict["forecaster_pytorch_v2"] = TestAllPtForecastersV2

    return testclass_dict


def get_test_classes_for_obj(obj):
    """Get all test classes relevant for an object or estimator.

    Parameters
    ----------
    obj : object or estimator, descendant of sktime BaseObject or BaseEstimator
        object or estimator for which to get test classes

    Returns
    -------
    test_classes : list of test classes
        list of test classes relevant for obj
        these are references to the actual classes, not strings
        if obj was not a descendant of BaseObject or BaseEstimator, returns empty list
    """
    if hasattr(obj, "_pkg"):
        obj = obj._pkg

    testclass_dict = get_test_class_registry()

    try:
        obj_scitypes = obj.get_tag("object_type")
        if not isinstance(obj_scitypes, (list, tuple, set)):
            obj_scitypes = [obj_scitypes]
    except Exception:
        obj_scitypes = []

    test_clss = []
    for obj_scitype in obj_scitypes:
        if obj_scitype in testclass_dict:
            test_clss += [testclass_dict[obj_scitype]]

    return test_clss
