"""
Transpose helper module for use in normalization wrappers.
"""

import torch.nn as nn


class Transpose(nn.Module):
    """Helper module to transpose two dimensions of a tensor.

    Commonly used to wrap ``BatchNorm1d`` so it can be applied along the
    feature (``d_model``) dimension inside a ``nn.Sequential`` norm layer.

    Parameters
    ----------
    *dims : int
        The two dimensions to transpose.
    contiguous : bool, default=False
        If ``True``, call ``.contiguous()`` on the result.
    """

    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        """Transpose the specified dimensions."""
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)
