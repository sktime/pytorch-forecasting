"""Validation checks."""


def _check_type(value, expected, name, allow_none=False, hint=""):
    """Raise ``TypeError`` unless ``value`` is an instance of ``expected``.

    Parameters
    ----------
    value : object
        The value to check.
    expected : type or tuple of type
        Acceptable type(s).
    name : str
        Parameter name, quoted in the message.
    allow_none : bool, default=False
        Whether ``None`` is acceptable.
    hint : str, default=""
        Sentence appended to the message, telling the user what to do.
    """
    if value is None and allow_none:
        return
    if isinstance(value, expected):
        return

    types = expected if isinstance(expected, tuple) else (expected,)
    wanted = " or ".join(t.__name__ for t in types)
    raise TypeError(
        f"`{name}` must be {wanted}, got {type(value).__name__}."
        + (f" {hint}" if hint else "")
    )


def _check_positive(value, name, allow_none=False):
    """Raise ``ValueError`` unless ``value`` is at least 1.

    Parameters
    ----------
    value : int or None
        The value to check.
    name : str
        Parameter name, quoted in the message.
    allow_none : bool, default=False
        Whether ``None`` is acceptable.
    """
    if value is None and allow_none:
        return
    if value < 1:
        raise ValueError(f"`{name}` must be at least 1, got {value}.")


def _check_within(value, upper, name, upper_name, allow_none=True):
    """Raise ``ValueError`` unless ``value`` lies in ``[1, upper]``.

    Parameters
    ----------
    value : int or None
        The value to check.
    upper : int
        Inclusive upper bound.
    name, upper_name : str
        Parameter names, quoted in the message.
    allow_none : bool, default=True
        Whether ``None`` is acceptable.
    """
    if value is None and allow_none:
        return
    if value < 1 or value > upper:
        raise ValueError(
            f"`{name}` must be between 1 and `{upper_name}` ({upper}), " f"got {value}."
        )


def _check_fractions(value, name, lengths=(2, 3)):
    """Raise ``ValueError`` unless ``value`` is fractions summing to at most 1.

    Parameters
    ----------
    value : sequence of float
        The value to check.
    name : str
        Parameter name, quoted in the message.
    lengths : tuple of int, default=(2, 3)
        Acceptable numbers of entries.
    """
    if len(value) not in lengths or any(fraction < 0 for fraction in value):
        wanted = " or ".join(str(length) for length in lengths)
        raise ValueError(
            f"`{name}` must be {wanted} non-negative fractions, got {value!r}."
        )
    if sum(value) > 1 + 1e-6:
        raise ValueError(
            f"`{name}` must sum to at most 1, got {value!r} "
            f"summing to {sum(value)}."
        )


def _check_column_names(named, allowed, allowed_label="columns of `data`", hint=""):
    """Raise ``ValueError`` if any parameter names a column that does not exist.

    Parameters
    ----------
    named : dict of str to (str, iterable of str, or None)
        Maps parameter name to the column name(s) it requested. ``None`` is
        skipped and a bare string is treated as a single name.
    allowed : iterable of str
        Column names that exist.
    allowed_label : str, default="columns of `data`"
        How ``allowed`` is described in the message.
    hint : str, default=""
        Sentence appended to the message.
    """
    allowed = list(allowed)
    for param, requested in named.items():
        if requested is None:
            continue
        if isinstance(requested, str):
            requested = [requested]

        unknown = [col for col in requested if col not in allowed]
        if unknown:
            raise ValueError(
                f"`{param}` names {allowed_label} that do not exist: "
                f"{unknown}. Available: {sorted(allowed)}."
                + (f" {hint}" if hint else "")
            )
