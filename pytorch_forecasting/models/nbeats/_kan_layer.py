# The following implementation of KANLayer is inspired by the pykan library.
# Reference: https://github.com/KindXiaoming/pykan/blob/master/kan/KANLayer.py

import numpy as np
import torch
import torch.nn as nn


def b_batch(x, grid, k=0):
    """
    evaluate x on B-spline bases

    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension
            (zero boundary condition). Default: True

    Returns:
    --------
        spline values : 3D torch.tensor
            shape (batch, in_dim, G+k). G: the number of grid intervals,
            k: spline order.

    Example
    -------
    Install the `pykan` package first:
    >>> pip install pykan
    Then use:

    >>> from kan.spline import B_batch
    >>> import torch
    >>> x = torch.rand(100, 2)
    >>> grid = torch.linspace(-1, 1, steps=11)[None, :].expand(2, 11)
    >>> B_batch(x, grid, k=3).shape

    """

    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)

    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = b_batch(x[:, :, 0], grid=grid[0], k=k - 1)

        value = (x - grid[:, :, : -(k + 1)]) / (
            grid[:, :, k:-1] - grid[:, :, : -(k + 1)]
        ) * B_km1[:, :, :-1] + (grid[:, :, k + 1 :] - x) / (
            grid[:, :, k + 1 :] - grid[:, :, 1:(-k)]
        ) * B_km1[:, :, 1:]

    # in case grid is degenerate
    value = torch.nan_to_num(value)
    return value


def coef2curve(x_eval, grid, coef, k):
    """
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves
    (summing up b_batch results over B-spline basis).

    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        grid : 2D torch.tensor
            shape (in_dim, G+2k). G: the number of grid intervals; k: spline order.
        coef : 3D torch.tensor
            shape (in_dim, out_dim, G+k)
        k : int
            the piecewise polynomial order of splines.

    Returns:
    --------
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)

    """

    b_splines = b_batch(x_eval, grid, k=k)
    y_eval = torch.einsum("ijk,jlk->ijl", b_splines, coef.to(b_splines))

    return y_eval


def curve2coef(x_eval, y_eval, grid, k):
    """
    converting B-spline curves to B-spline coefficients using least squares.

    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)
        grid : 2D torch.tensor
            shape (in_dim, grid+2*k)
        k : int
            spline order
        lamb : float
            regularized least square lambda

    Returns:
    --------
        coef : 3D torch.tensor
            shape (in_dim, out_dim, G+k)
    """
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1
    mat = b_batch(x_eval, grid, k)
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)
    y_eval = y_eval.permute(1, 2, 0).unsqueeze(dim=3)
    try:
        coef = torch.linalg.lstsq(mat, y_eval).solution[:, :, :, 0]
    except Exception as e:
        print(f"lstsq failed with error: {e}")

    return coef


def extend_grid(grid, k_extend=0):
    """
    Extend a grid tensor by padding both ends with equal spacing.

    Args:
    -----
        grid : torch.Tensor
            Grid of shape (in_dim, grid_points).
        k_extend : int
            Number of points to extend on both ends.

    Returns:
    --------
        grid : torch.Tensor
            Extended grid of shape (in_dim, grid_points + 2 * k_extend).
    """
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    return grid


def sparse_mask(in_dim, out_dim):
    """
    Generate a sparse connection mask between input and output units.

    Args:
    -----
        in_dim : int
            Number of input units.
        out_dim : int
            Number of output units.

    Returns:
    --------
        mask : torch.Tensor
            Sparse binary mask of shape (in_dim, out_dim).
    """
    in_coord = torch.arange(in_dim) * 1 / in_dim + 1 / (2 * in_dim)
    out_coord = torch.arange(out_dim) * 1 / out_dim + 1 / (2 * out_dim)

    dist_mat = torch.abs(out_coord[:, None] - in_coord[None, :])
    in_nearest = torch.argmin(dist_mat, dim=0)
    in_connection = torch.stack([torch.arange(in_dim), in_nearest]).permute(1, 0)
    out_nearest = torch.argmin(dist_mat, dim=1)
    out_connection = torch.stack([out_nearest, torch.arange(out_dim)]).permute(1, 0)
    all_connection = torch.cat([in_connection, out_connection], dim=0)
    mask = torch.zeros(in_dim, out_dim)
    mask[all_connection[:, 0], all_connection[:, 1]] = 1.0

    return mask


class KANLayer(nn.Module):
    """
    KANLayer class
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
        """'
        Initialize a KANLayer

        Args:
        -----
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
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: None.
            sp_trainable : bool
                If true, scale_sp is trainable
            sb_trainable : bool
                If true, scale_base is trainable
            sparse_init : bool
                if sparse_init = True, sparse initialization is applied.

        Returns:
        --------
            self

        Example
        -------
        Install the `pykan` package first:
        >>> pip install pykan
        Then use:

        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        """
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

        Args:
        -----
        x : torch.Tensor
            Input tensor of shape (batch_size, in_dim), where:
              - batch_size is the number of input samples.
              - in_dim is the input feature dimension.

        Returns:
        --------
        y : torch.Tensor
            Output tensor, the result of applying spline and residual
            transformations followed by weighted summation.

        Example
        -------
        Install the `pykan` package first:
        >>> pip install pykan
        Then use:

        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, preacts, postacts, postspline = model(x)
        >>> y.shape, preacts.shape, postacts.shape, postspline.shape
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
        update grid from samples

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
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

            Args:
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
