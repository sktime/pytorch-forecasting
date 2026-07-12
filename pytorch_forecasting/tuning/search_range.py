"""
SearchRange: A typed container for hyperparameter search spaces.

"""

from dataclasses import dataclass
from typing import Any


@dataclass
class _SearchRange:
    """Defines a search range for a single hyperparameter.

    Parameters
    ----------
    param_type : str
        One of "int", "float", or "categorical".
    low : float or int, optional
        Lower bound (for int/float types).
    high : float or int, optional
        Upper bound (for int/float types).
    choices : list, optional
        Valid choices (for categorical type).
    log : bool, default=False
        If True, sample in log-uniform space.
        Use for params where relative change matters more than absolute
        (e.g., learning_rate: 1e-5 vs 1e-4 is a 10x change).
    step : int, optional
        Step size for integer parameters.
    """

    param_type: str
    low: float | int | None = None
    high: float | int | None = None
    choices: list[Any] | None = None
    log: bool = False
    step: int | None = None

    def suggest(self, trial, name: str):
        """Ask Optuna to suggest a value for this parameter.

        This bridges our SearchRange to Optuna's trial API.
        """
        if self.param_type == "int":
            return trial.suggest_int(name, self.low, self.high, log=self.log)
        elif self.param_type == "float":
            return trial.suggest_float(name, self.low, self.high, log=self.log)
        elif self.param_type == "categorical":
            return trial.suggest_categorical(name, self.choices)
        else:
            raise ValueError(f"Unknown param_type: {self.param_type}")
