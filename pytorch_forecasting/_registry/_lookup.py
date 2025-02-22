"""Registry lookup methods.

This module exports the following methods for registry lookup:

all_objects(object_types, filter_tags)
    lookup and filtering of objects
"""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# based on the sktime module of same name

__author__ = ["fkiraly"]
# all_objects is based on the sklearn utility all_estimators


from copy import deepcopy
from operator import itemgetter
from pathlib import Path

import pandas as pd
from skbase.lookup import all_objects as _all_objects

from pytorch_forecasting.models.base import _BaseObject


def all_objects(
    object_types=None,
    filter_tags=None,
    exclude_objects=None,
    return_names=True,
    as_dataframe=False,
    return_tags=None,
    suppress_import_stdout=True,
):
    """Get a list of all objects from pytorch_forecasting.

    This function crawls the module and gets all classes that inherit
    from skbase compatible base classes.

    Not included are: the base classes themselves, classes defined in test
    modules.

    Parameters
    ----------
    object_types: str, list of str, optional (default=None)
        Which kind of objects should be returned.
        if None, no filter is applied and all objects are returned.
        if str or list of str, strings define scitypes specified in search
        only objects that are of (at least) one of the scitypes are returned
        possible str values are entries of registry.BASE_CLASS_REGISTER (first col)
        for instance 'regrssor_proba', 'distribution, 'metric'

    return_names: bool, optional (default=True)

        if True, estimator class name is included in the ``all_objects``
        return in the order: name, estimator class, optional tags, either as
        a tuple or as pandas.DataFrame columns

        if False, estimator class name is removed from the ``all_objects`` return.

    filter_tags: dict of (str or list of str), optional (default=None)
        For a list of valid tag strings, use the registry.all_tags utility.

        ``filter_tags`` subsets the returned estimators as follows:

        * each key/value pair is statement in "and"/conjunction
        * key is tag name to sub-set on
        * value str or list of string are tag values
        * condition is "key must be equal to value, or in set(value)"

    exclude_estimators: str, list of str, optional (default=None)
        Names of estimators to exclude.

    as_dataframe: bool, optional (default=False)

        True: ``all_objects`` will return a pandas.DataFrame with named
        columns for all of the attributes being returned.

        False: ``all_objects`` will return a list (either a list of
        estimators or a list of tuples, see Returns)

    return_tags: str or list of str, optional (default=None)
        Names of tags to fetch and return each estimator's value of.
        For a list of valid tag strings, use the registry.all_tags utility.
        if str or list of str,
        the tag values named in return_tags will be fetched for each
        estimator and will be appended as either columns or tuple entries.

    suppress_import_stdout : bool, optional. Default=True
        whether to suppress stdout printout upon import.

    Returns
    -------
    all_objects will return one of the following:
        1. list of objects, if return_names=False, and return_tags is None
        2. list of tuples (optional object name, class, ~optional object
          tags), if return_names=True or return_tags is not None.
        3. pandas.DataFrame if as_dataframe = True
        if list of objects:
            entries are objects matching the query,
            in alphabetical order of object name
        if list of tuples:
            list of (optional object name, object, optional object
            tags) matching the query, in alphabetical order of object name,
            where
            ``name`` is the object name as string, and is an
                optional return
            ``object`` is the actual object
            ``tags`` are the object's values for each tag in return_tags
                and is an optional return.
        if dataframe:
            all_objects will return a pandas.DataFrame.
            column names represent the attributes contained in each column.
            "objects" will be the name of the column of objects, "names"
            will be the name of the column of object class names and the string(s)
            passed in return_tags will serve as column names for all columns of
            tags that were optionally requested.

    Examples
    --------
    >>> from skpro.registry import all_objects
    >>> # return a complete list of objects as pd.Dataframe
    >>> all_objects(as_dataframe=True)  # doctest: +SKIP
    >>> # return all probabilistic regressors by filtering for object type
    >>> all_objects("regressor_proba", as_dataframe=True)  # doctest: +SKIP
    >>> # return all regressors which handle missing data in the input by tag filtering
    >>> all_objects(
    ...     "regressor_proba",
    ...     filter_tags={"capability:missing": True},
    ...     as_dataframe=True
    ... )  # doctest: +SKIP

    References
    ----------
    Adapted version of sktime's ``all_estimators``,
    which is an evolution of scikit-learn's ``all_estimators``
    """
    MODULES_TO_IGNORE = (
        "tests",
        "setup",
        "contrib",
        "utils",
        "all",
    )

    result = []
    ROOT = str(Path(__file__).parent.parent)  # skpro package root directory

    if isinstance(filter_tags, str):
        filter_tags = {filter_tags: True}
    filter_tags = filter_tags.copy() if filter_tags else None

    if object_types:
        if filter_tags and "object_type" not in filter_tags.keys():
            object_tag_filter = {"object_type": object_types}
        elif filter_tags:
            filter_tags_filter = filter_tags.get("object_type", [])
            if isinstance(object_types, str):
                object_types = [object_types]
            object_tag_update = {"object_type": object_types + filter_tags_filter}
            filter_tags.update(object_tag_update)
        else:
            object_tag_filter = {"object_type": object_types}
        if filter_tags:
            filter_tags.update(object_tag_filter)
        else:
            filter_tags = object_tag_filter

    result = _all_objects(
        object_types=[_BaseObject],
        filter_tags=filter_tags,
        exclude_objects=exclude_objects,
        return_names=return_names,
        as_dataframe=as_dataframe,
        return_tags=return_tags,
        suppress_import_stdout=suppress_import_stdout,
        package_name="skpro",
        path=ROOT,
        modules_to_ignore=MODULES_TO_IGNORE,
    )

    return result


def _check_list_of_str_or_error(arg_to_check, arg_name):
    """Check that certain arguments are str or list of str.

    Parameters
    ----------
    arg_to_check: argument we are testing the type of
    arg_name: str,
        name of the argument we are testing, will be added to the error if
        ``arg_to_check`` is not a str or a list of str

    Returns
    -------
    arg_to_check: list of str,
        if arg_to_check was originally a str it converts it into a list of str
        so that it can be iterated over.

    Raises
    ------
    TypeError if arg_to_check is not a str or list of str
    """
    # check that return_tags has the right type:
    if isinstance(arg_to_check, str):
        arg_to_check = [arg_to_check]
    if not isinstance(arg_to_check, list) or not all(
        isinstance(value, str) for value in arg_to_check
    ):
        raise TypeError(
            f"Error in all_objects!  Argument {arg_name} must be either\
             a str or list of str"
        )
    return arg_to_check
