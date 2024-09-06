from functools import wraps

import pytest


def monkey_patch_torch_fn(torch_fn: str, return_value: bool):
    """Decorator to monkeypatch the torch function in tests.
    Parameters
    ----------
    torch_fn : str
        The torch function to monkeypatch.
    return_value : bool
        The return value of the torch function.

    Returns
    -------
    decorator
        The decorator function that will monkeypatch the environment variable.

    Examples
    --------
    import os
    import pytest
    from helpers import monkey_patch_torch_fn

    @monkey_patch_torch_fn("torch._C._mps_is_available", False)
    def test_get_lstm_cell():
        import torch
        assert torch._C._mps_is_available() == False
    """

    def decorator(test_func):
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            monkeypatch = kwargs.get("monkeypatch", pytest.MonkeyPatch())
            monkeypatch.setattr(torch_fn, lambda: return_value)

            return test_func(*args, **kwargs)

        return pytest.mark.usefixtures("monkeypatch")(wrapper)

    return decorator
