"""
Deprecated: Flatten_Head has been merged into FlattenHead.

Use ``pytorch_forecasting.layers.FlattenHead`` with ``individual=True`` instead.
"""

import warnings

from pytorch_forecasting.layers._output._flatten_head import FlattenHead


class Flatten_Head(FlattenHead):
    """Deprecated alias for :class:`FlattenHead` with ``individual`` support.

    .. deprecated::
        Use :class:`pytorch_forecasting.layers.FlattenHead` with the
        ``individual`` keyword argument instead.  ``Flatten_Head`` will be
        removed in a future release.

    Parameters
    ----------
    individual : bool
        Passed through to :class:`FlattenHead` as ``individual``.
    n_vars : int
        Number of variables.
    nf : int
        Number of features (flattened).
    target_window : int
        Length of the target window.
    head_dropout : float
        Dropout rate. Defaults to ``0``.
    """

    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        warnings.warn(
            "Flatten_Head is deprecated and will be removed in a future release. "
            "Use FlattenHead with individual=True/False instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            n_vars=n_vars,
            nf=nf,
            target_window=target_window,
            head_dropout=head_dropout,
            individual=individual,
        )
