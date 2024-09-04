from functools import wraps

import pytest


def monkeypatch_env(key, value):
    """Decorator to monkeypatch environment variables in tests.
    Parameters
    ----------
    key : str
        The environment variable key.
    value : str
        The environment variable value.

    Returns
    -------
    decorator
        The decorator function that will monkeypatch the environment variable.

    Examples
    --------
    import os
    import pytest
    from helpers import monkeypatch_env

    @monkeypatch_env("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    def test_get_lstm_cell():
        assert os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    """

    def decorator(test_func):
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            @pytest.fixture
            def _monkeypatch_env(monkeypatch):
                monkeypatch.setenv(key, value)

            @pytest.mark.usefixtures("_monkeypatch_env")
            def run_test():
                return test_func(*args, **kwargs)

            return run_test()

        return wrapper

    return decorator
