"""
Reverse Instance Normalization (RevIN) layer.
---------------------------------------------
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        """
        Reverse Instance Normalization (RevIN) layer.

        Parameters
        ----------
        num_features : int
            Number of input features.
        eps : float, optional
            A small value added to the denominator for numerical stability (default: 1e-5).
        affine : bool, optional
            If True, the layer will have learnable affine parameters (default: True).
        subtract_last: bool, optional
            If True, the last feature will be subtracted from the mean (default: False).
        """  # noqa: E501
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self._init_params()

    def _init_params(self):
        """Initialize learnable parameters if affine is True."""
        self.weight = nn.Parameter(torch.ones(self.num_features))
        self.bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()  # noqa: E501

    def _normalize(self, x):
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
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
