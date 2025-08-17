# The following implementation of KANLayer is inspired by the pykan library.
# Reference: https://github.com/KindXiaoming/pykan/blob/master/kan/KANLayer.py

import numpy as np
import torch
import torch.nn as nn

from pytorch_forecasting.layers._kan._utils import (
    coef2curve,
    curve2coef,
    extend_grid,
    sparse_mask,
)


class KANLayer(nn.Module):
    """
    Initialize a KANLayer

    Parameters
    ----------
    in_dim : int
        input dimension. Default: 2.
    out_dim : int
        output dimension. Default: 3.
    num : int
        the number of grid intervals = G. Default: 5.
    k : int
        the order of piecewise polynomial. Default: 3.
    noise_scale : float
        the scale of noise injected at initialization. Default: 0.1.
    scale_base_mu : float
        the scale of the residual function b(x) is intialized to be
        N(scale_base_mu, scale_base_sigma^2).
    scale_base_sigma : float
        the scale of the residual function b(x) is intialized to be
        N(scale_base_mu, scale_base_sigma^2).
    scale_sp : float
        the scale of the base function spline(x).
    base_fun : function
        residual function b(x). Default: None
    grid_eps : float
        When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is
        partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates
        between the two extremes.
    grid_range : list or np.array of shape (2,)
        setting the range of grids. Default: None.
    sp_trainable : bool
        If true, scale_sp is trainable.
    sb_trainable : bool
        If true, scale_base is trainable.
    sparse_init : bool
        if sparse_init = True, sparse initialization is applied.

    Returns
    -------
    self : reference to self

    Examples
    --------
    The following is an example from the original `pykan` library, adapted here
    for illustration within the PyTorch Forecasting integration.

    Install the `pykan` package first:
    pip install pykan
    Then use:

    >>> from kan.KANLayer import *
    >>> model = KANLayer(in_dim=3, out_dim=5)
    >>> (model.in_dim, model.out_dim)
    """

    def __init__(
        self,
        in_dim=3,
        out_dim=2,
        num=5,
        k=3,
        noise_scale=0.5,
        scale_base_mu=0.0,
        scale_base_sigma=1.0,
        scale_sp=1.0,
        base_fun=None,
        grid_eps=0.02,
        grid_range=None,
        sp_trainable=True,
        sb_trainable=True,
        sparse_init=False,
    ):
        super().__init__()

        # Handle mutable parameters
        if grid_range is None:
            grid_range = [-1, 1]
        if base_fun is None:
            base_fun = torch.nn.SiLU()
        # size
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[
            None, :
        ].expand(self.in_dim, num + 1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        noises = (
            (torch.rand(self.num + 1, self.in_dim, self.out_dim) - 1 / 2)
            * noise_scale
            / num
        )

        self.coef = torch.nn.Parameter(
            curve2coef(self.grid[:, k:-k].permute(1, 0), noises, self.grid, k)
        )

        if sparse_init:
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(
                False
            )
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(
                False
            )

        self.scale_base = torch.nn.Parameter(
            scale_base_mu * 1 / np.sqrt(in_dim)
            + scale_base_sigma
            * (torch.rand(in_dim, out_dim) * 2 - 1)
            * 1
            / np.sqrt(in_dim)
        ).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(
            torch.ones(in_dim, out_dim) * scale_sp * 1 / np.sqrt(in_dim) * self.mask
        ).requires_grad_(sp_trainable)  # make scale trainable
        self.base_fun = base_fun

        self.grid_eps = grid_eps

    def forward(self, x):
        """
        KANLayer forward given input x

        Parameters
        -----
        x : torch.Tensor
            Input tensor of shape (batch_size, in_dim), where:
              - batch_size is the number of input samples.
              - in_dim is the input feature dimension.

        Returns
        --------
        y : torch.Tensor
            Output tensor, the result of applying spline and residual
            transformations followed by weighted summation.

        Examples
        --------
        The following is an example from the original `pykan` library, adapted here
        for illustration within the PyTorch Forecasting integration.

        Install the `pykan` package first:
        pip install pykan
        Then use:

        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, _, _, _ = model(x)
        >>> y.shape
        """

        base = self.base_fun(x)  # (batch, in_dim)
        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k)
        y = (
            self.scale_base[None, :, :] * base[:, :, None]
            + self.scale_sp[None, :, :] * y
        )
        y = self.mask[None, :, :] * y
        y = torch.sum(y, dim=1)
        return y

    def update_grid_from_samples(self, x):
        """
        Update grid from samples

        Parameters
        -----
        x : 2D torch.float
            inputs, shape (number of samples, input dimension)

        Returns:
        --------
        None

        Examples
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        """

        batch = x.shape[0]
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        num_interval = self.grid.shape[1] - 1 - 2 * self.k

        def get_grid(num_interval):
            """
            Generate adaptive or uniform grid points from sorted input samples.

            Parameters
            -----
            num_interval : int
                Number of intervals between grid points.

            Returns:
            --------
            grid : torch.Tensor
                New grid of shape (in_dim, num_interval + 1).
            """
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1, 0)
            h = (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]]) / num_interval
            grid_uniform = (
                grid_adaptive[:, [0]]
                + h * torch.arange(num_interval + 1, device=h.device)[None, :]
            )
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid

        grid = get_grid(num_interval)
        self.grid.data = extend_grid(grid, k_extend=self.k)
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)
