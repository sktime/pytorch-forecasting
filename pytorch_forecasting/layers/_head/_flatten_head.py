"""
Flatten Head layer for time series forecasting models.
"""

import torch
import torch.nn as nn


class Flatten_Head(nn.Module):
    """
    Flatten Head.

    This layer flattens the input and projects
    it to the target window.

    Parameters
    ----------
    individual : bool
        If True, uses a separate linear projection per variable.
    n_vars : int
        Number of variables.
    nf : int
        Number of features.
    target_window : int
        Length of the target window.
    head_dropout : float
        Dropout rate.
    """

    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
