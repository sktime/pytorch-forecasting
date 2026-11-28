"""Registry lookup methods.

This module exports the following methods for registry lookup:

all_objects(object_types, filter_tags)
    lookup and filtering of objects

all_tags(parent_types)
    lookup and filtering of object tags
"""

# based on the sktime module of same name

__author__ = ["fkiraly"]
# all_objects is based on the sklearn utility all_estimators

from difflib import get_close_matches
from inspect import isclass
from pathlib import Path

from skbase.lookup import all_objects as _all_objects

from pytorch_forecasting.base._base_object import _BaseObject


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

        * if None, no filter is applied and all objects are returned.
        * if str or list of str, strings define scitypes specified in search
          only objects that are of (at least) one of the scitypes are returned

    return_names: bool, optional (default=True)

        * if True, estimator class name is included in the ``all_objects``
          return in the order: name, estimator class, optional tags, either as
          a tuple or as pandas.DataFrame columns
        * if False, estimator class name is removed from the ``all_objects`` return.

    filter_tags: dict of (str or list of str or re.Pattern), optional (default=None)
        For a list of valid tag strings, use
        ``pytorch_forecasting._registry.all_tags``.

        ``filter_tags`` subsets the returned objects as follows:

        * each key/value pair is statement in "and"/conjunction
        * key is tag name to sub-set on
        * value str or list of string are tag values
        * condition is "key must be equal to value, or in set(value)"

        In detail, he return will be filtered to keep exactly the classes
        where tags satisfy all the filter conditions specified by ``filter_tags``.
        Filter conditions are as follows, for ``tag_name: search_value`` pairs in
        the ``filter_tags`` dict, applied to a class ``klass``:

        - If ``klass`` does not have a tag with name ``tag_name``, it is excluded.
          Otherwise, let ``tag_value`` be the value of the tag with name ``tag_name``.
        - If ``search_value`` is a string, and ``tag_value`` is a string,
          the filter condition is that ``search_value`` must match the tag value.
        - If ``search_value`` is a string, and ``tag_value`` is a list,
          the filter condition is that ``search_value`` is contained in ``tag_value``.
        - If ``search_value`` is a ``re.Pattern``, and ``tag_value`` is a string,
          the filter condition is that ``search_value.fullmatch(tag_value)``
          is true, i.e., the regex matches the tag value.
        - If ``search_value`` is a ``re.Pattern``, and ``tag_value`` is a list,
          the filter condition is that at least one element of ``tag_value``
          matches the regex.
        - If ``search_value`` is iterable, then the filter condition is that
          at least one element of ``search_value`` satisfies the above conditions,
          applied to ``tag_value``.

        Note: ``re.Pattern`` is supported only from ``scikit-base`` version 0.8.0.

    exclude_objects: str, list of str, optional (default=None)
        Names of objects to exclude.

    as_dataframe: bool, optional (default=False)

        * True: ``all_objects`` will return a ``pandas.DataFrame`` with named
          columns for all of the attributes being returned.
        * False: ``all_objects`` will return a list (either a list of
          objects or a list of tuples, see Returns)

    return_tags: str or list of str, optional (default=None)
        Names of tags to fetch and return each estimator's value of.
        For a list of valid tag strings, use
        ``pytorch_forecasting._registry.all_tags``.
        if str or list of str,
        the tag values named in return_tags will be fetched for each
        estimator and will be appended as either columns or tuple entries.

    suppress_import_stdout : bool, optional. Default=True
        whether to suppress stdout printout upon import.

    Returns
    -------
    all_objects will return one of the following:

        1. list of objects, if ``return_names=False``, and ``return_tags`` is None

        2. list of tuples (optional estimator name, class, optional estimator
        tags), if ``return_names=True`` or ``return_tags`` is not ``None``.

        3. ``pandas.DataFrame`` if ``as_dataframe = True``

        if list of objects:
            entries are objects matching the query,
            in alphabetical order of estimator name

        if list of tuples:
            list of (optional estimator name, estimator, optional estimator
            tags) matching the query, in alphabetical order of estimator name,
            where
            ``name`` is the estimator name as string, and is an
            optional return
            ``estimator`` is the actual estimator
            ``tags`` are the estimator's values for each tag in return_tags
            and is an optional return.

        if ``DataFrame``:
            column names represent the attributes contained in each column.
            "objects" will be the name of the column of objects, "names"
            will be the name of the column of estimator class names and the string(s)
            passed in return_tags will serve as column names for all columns of
            tags that were optionally requested.

    Examples
    --------
    >>> from pytorch_forecasting._registry import all_objects
    >>> # return a complete list of objects as pd.Dataframe
    >>> all_objects(as_dataframe=True)  # doctest: +SKIP

    References
    ----------
    Adapted version of sktime's ``all_estimators``,
    which is an evolution of scikit-learn's ``all_estimators``
    """
    _check_filter_tag_names(filter_tags)

    MODULES_TO_IGNORE = (
        "tests",
        "setup",
        "contrib",
        "utils",
        "all",
        "_registry",
    )

    result = []
    ROOT = str(Path(__file__).parent.parent)  # package root directory

    def _coerce_to_str(obj):
        if isinstance(obj, list | tuple):
            return [_coerce_to_str(o) for o in obj]
        if isclass(obj):
            obj = obj.get_tag("object_type")
        return obj

    def _coerce_to_list_of_str(obj):
        obj = _coerce_to_str(obj)
        if isinstance(obj, str):
            return [obj]
        return obj

    if object_types is not None:
        object_types = _coerce_to_list_of_str(object_types)
        object_types = list(set(object_types))

    if object_types is not None:
        if filter_tags is None:
            filter_tags = {}
        elif isinstance(filter_tags, str):
            filter_tags = {filter_tags: True}
        else:
            filter_tags = filter_tags.copy()

        if "object_type" in filter_tags:
            obj_field = filter_tags["object_type"]
            obj_field = _coerce_to_list_of_str(obj_field)
            obj_field = obj_field + object_types
        else:
            obj_field = object_types

        filter_tags["object_type"] = obj_field

    result = _all_objects(
        object_types=[_BaseObject],
        filter_tags=filter_tags,
        exclude_objects=exclude_objects,
        return_names=return_names,
        as_dataframe=as_dataframe,
        return_tags=return_tags,
        suppress_import_stdout=suppress_import_stdout,
        package_name="pytorch_forecasting",
        path=ROOT,
        modules_to_ignore=MODULES_TO_IGNORE,
    )

    return result


def _check_filter_tag_names(filter_tags):
    """Raise if ``filter_tags`` names a tag that is not in the register.

    Parameters
    ----------
    filter_tags : None, str, or dict
        the ``filter_tags`` argument of ``all_objects``, before any coercion

    Raises
    ------
    KeyError
        if a key of ``filter_tags`` is not a registered tag name
    """
    from pytorch_forecasting._registry._tags import OBJECT_TAG_LIST

    if filter_tags is None:
        return

    if isinstance(filter_tags, str):
        names = [filter_tags]
    else:
        names = list(filter_tags)

    for name in names:
        if name in OBJECT_TAG_LIST:
            continue
        msg = f"{name!r} is not a valid tag name."
        close = get_close_matches(name, OBJECT_TAG_LIST, n=1)
        if close:
            msg += f" Did you mean {close[0]!r}?"
        msg += " Use pytorch_forecasting._registry.all_tags() for the full list."
        raise KeyError(msg)


def all_tags(
    parent_types=None,
    return_names=True,
    as_dataframe=False,
):
    """Get a list of all object tags from pytorch_forecasting.

    Parameters
    ----------
    parent_types : str or list of str, optional (default=None)
        parent type to filter the tags by, e.g. ``"metric"``.
        If None, tags for all parent types are returned.
    return_names : bool, optional (default=True)
        if True, return tuples ``(name, scitype, type, description)``;
        if False, return tag names only, without duplicates.
    as_dataframe : bool, optional (default=False)
        if True, return a ``pd.DataFrame`` with one row per tag and columns
        ``["name", "scitype", "type", "description"]``. Takes precedence
        over ``return_names``.

    Returns
    -------
    list of tuple, list of str, or pd.DataFrame
        shape as described under ``return_names`` and ``as_dataframe``

    Examples
    --------
    >>> from pytorch_forecasting._registry import all_tags
    >>> metric_tags = all_tags(parent_types="metric", return_names=False)
    >>> "metric_type" in metric_tags
    True
    """
    import pandas as pd

    from pytorch_forecasting._registry._tags import OBJECT_TAG_REGISTER

    rows = list(OBJECT_TAG_REGISTER)

    if parent_types is not None:
        parent_types = _check_list_of_str_or_error(parent_types, "parent_types")
        rows = [row for row in rows if row[1] in parent_types]

    if as_dataframe:
        return pd.DataFrame(rows, columns=["name", "scitype", "type", "description"])

    if not return_names:
        names = []
        for row in rows:
            if row[0] not in names:
                names.append(row[0])
        return names

    return rows


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
