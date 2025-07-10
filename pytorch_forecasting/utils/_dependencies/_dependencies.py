"""Utilities for managing dependencies.

Copied from sktime/skbase.
"""

from functools import lru_cache


@lru_cache
def _get_installed_packages_private():
    """Get a dictionary of installed packages and their versions.

    Same as _get_installed_packages, but internal to avoid mutating the lru_cache
    by accident.
    """
    from importlib.metadata import distributions, version

    dists = distributions()
    package_names = {dist.metadata["Name"] for dist in dists}
    package_versions = {pkg_name: version(pkg_name) for pkg_name in package_names}
    # developer note:
    # we cannot just use distributions naively,
    # because the same top level package name may appear *twice*,
    # e.g., in a situation where a virtual env overrides a base env,
    # such as in deployment environments like databricks.
    # the "version" contract ensures we always get the version that corresponds
    # to the importable distribution, i.e., the top one in the sys.path.
    return package_versions


def _get_installed_packages():
    """Get a dictionary of installed packages and their versions.

    Returns
    -------
    dict : dictionary of installed packages and their versions
        keys are PEP 440 compatible package names, values are package versions
        MAJOR.MINOR.PATCH version format is used for versions, e.g., "1.2.3"
    """
    return _get_installed_packages_private().copy()


def _check_matplotlib(ref="This feature", raise_error=True):
    """Check if matplotlib is installed.

    Parameters
    ----------
    ref : str, optional (default="This feature")
        reference to the feature that requires matplotlib, used in error message
    raise_error : bool, optional (default=True)
        whether to raise an error if matplotlib is not installed

    Returns
    -------
    bool : whether matplotlib is installed
    """
    pkgs = _get_installed_packages()

    if raise_error and "matplotlib" not in pkgs:
        raise ImportError(
            f"{ref} requires matplotlib."
            " Please install matplotlib with `pip install matplotlib`."
        )

    return "matplotlib" in pkgs
