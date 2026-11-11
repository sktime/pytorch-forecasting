"""
Implementation of output layers from `nn.Module` for time series forecasting models.
"""

import torch
import torch.nn as nn


class FlattenHead(nn.Module):
    """
    Flatten head for the output of a time series forecasting model.

    Flattens the last two dimensions of the input (patch/time × feature), projects
    to the target window, and optionally produces per-quantile outputs.  When
    ``individual=True`` a separate linear projection is used for each input
    variable (channel-independence head), mirroring the behaviour of the former
    ``Flatten_Head`` class.

    Parameters
    ----------
    n_vars : int
        Number of input variables / channels.
    nf : int
        Size of the flattened feature dimension fed into the linear projection
        (e.g. ``d_model * n_patches``).
    target_window : int
        Number of forecast time steps to predict.
    head_dropout : float
        Dropout rate applied after the linear projection. Defaults to ``0``.
    n_quantiles : int or None
        If not ``None``, the linear layer projects to
        ``target_window * n_quantiles`` and the output is reshaped to expose the
        quantile dimension. Defaults to ``None``.
    individual : bool
        If ``True``, a separate ``nn.Linear`` (and ``nn.Flatten`` / ``nn.Dropout``)
        is created for each of the ``n_vars`` input variables.  Defaults to
        ``False``.

    Output shape
    ------------
    * ``n_quantiles=None``  → ``(batch, target_window, n_vars)``
    * ``n_quantiles=k``     → ``(batch, target_window * n_vars, k)``

    Both ``individual=True`` and ``individual=False`` produce the same output
    shape for a given ``(target_window, n_quantiles)`` pair.
    """

    def __init__(
        self,
        n_vars,
        nf,
        target_window,
        head_dropout=0,
        n_quantiles=None,
        individual=False,
    ):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        self.n_quantiles = n_quantiles

        out_dim = target_window * n_quantiles if n_quantiles is not None else target_window

        if individual:
            self.flattens = nn.ModuleList(
                [nn.Flatten(start_dim=-2) for _ in range(n_vars)]
            )
            self.linears = nn.ModuleList(
                [nn.Linear(nf, out_dim) for _ in range(n_vars)]
            )
            self.dropouts = nn.ModuleList(
                [nn.Dropout(head_dropout) for _ in range(n_vars)]
            )
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, out_dim)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # (B, n_vars, target_window [* n_q])
            x = x.permute(0, 2, 1)         # (B, target_window [* n_q], n_vars)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
            x = x.permute(0, 2, 1)         # (B, target_window [* n_q], n_vars)

        if self.n_quantiles is not None:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1, self.n_quantiles)
        return x
