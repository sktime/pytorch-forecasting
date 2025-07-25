"""Utilities for managing dependencies.

Copied from sktime/skbase.
"""

from functools import lru_cache

from skbase.utils._dependencies import _check_soft_dependencies


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
    if raise_error and not _check_soft_dependencies("matplotlib", severity="none"):
        raise ImportError(
            f"{ref} requires matplotlib."
            " Please install matplotlib with `pip install matplotlib`."
        )

    return "matplotlib" in pkgs
