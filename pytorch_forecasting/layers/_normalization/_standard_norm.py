"""
Normalize layer.
---------------------------------------------
"""

import torch
import torch.nn as nn


class Normalize(nn.Module):
    """
    Standard normalization utility used by normalization wrappers.

    Parameters
    ----------
    num_features : int
        Number of features or channels (C).
    eps : float, optional
        Small value added for numerical stability when computing
        standard deviation (default: 1e-5).
    affine : bool, optional
        If True, learnable affine parameters (`affine_weight` and
        `affine_bias`) are created and applied after normalization
        (default: False).
    subtract_last : bool, optional
        If True, statistics are computed relative to the last time
        step instead of the mean (useful for some forecasting setups)
        (default: False).
    non_norm : bool, optional
        If True, normalization and denormalization are no-ops; kept for
        compatibility with callers that may toggle normalization off
        (default: False).
    """

    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        """Initialize learnable affine parameters.

        """
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """Compute and cache statistics used for normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(B, T, C, ...)` where reduction is
            performed across temporal (and any extra) dimensions.
        """
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        """Apply normalization to input using cached statistics.
        """
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        """Reverse the normalization operation.
        """
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x