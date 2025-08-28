"""
Utility functions for KAN (Kolmogorov Arnold Network) Layer.
Contains B-spline computations, curve transformations, and grid manipulation functions.
"""

import torch


def b_batch(x, grid, k=0):
    """
    Evaluate x on B-spline bases

    Parameters
    ----------
    x : torch.Tensor
        2D tensor of inputs, shape (number of splines, number of samples).
    grid : torch.Tensor
        2D tensor of grids, shape (number of splines, number of grid points).
    k : int
        The piecewise polynomial order of splines.
    extend : bool
        If True, k points are extended on both ends. If False, no extension
        (zero boundary condition). Default: True.

    Returns
    -------
    spline values : torch.Tensor
        3D tensor of shape (batch, in_dim, G+k), where G is the number of
        grid intervals and k is the spline order.

    Examples
    --------
    The following is an example from the original `pykan` library, adapted here
    for illustration within the PyTorch Forecasting integration.

    Install the `pykan` package first:
    pip install pykan
    Then use:

    >>> from pytorch_forecasting.layers._kan._utils import b_batch
    >>> import torch
    >>> x = torch.rand(100, 2)
    >>> grid = torch.linspace(-1, 1, steps=11)[None, :].expand(2, 11)
    >>> b_batch(x, grid, k=3).shape
    torch.Size([100, 2, 7])
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
    Converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves
    (summing up b_batch results over B-spline basis).

    Parameters
    ----------
    x_eval : torch.Tensor
        2D tensor of shape (batch, in_dim).
    grid : torch.Tensor
        2D tensor of shape (in_dim, G+2k). G: the number of grid intervals;
        k: spline order.
    coef : torch.Tensor
        3D tensor of shape (in_dim, out_dim, G+k).
    k : int
        The piecewise polynomial order of splines.

    Returns
    -------
    y_eval : torch.Tensor
        3D tensor of shape (batch, in_dim, out_dim).
    """

    b_splines = b_batch(x_eval, grid, k=k)
    y_eval = torch.einsum("ijk,jlk->ijl", b_splines, coef.to(b_splines))

    return y_eval


def curve2coef(x_eval, y_eval, grid, k):
    """
    Estimate spline coefficients via batched least squares.

    Parameters
    ----------
    x_eval : torch.Tensor
        2D tensor of shape (batch, in_dim).
    y_eval : torch.Tensor
        3D tensor of shape (batch, in_dim, out_dim).
    grid : torch.Tensor
        2D tensor of shape (in_dim, grid + 2 * k).
    k : int
        Spline order.
    lamb : float
        Regularized least square lambda.

    Returns
    -------
    coef : torch.Tensor
        3D tensor of shape (in_dim, out_dim, G + k).
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

    Parameters
    ----------
    grid : torch.Tensor
        Grid of shape (in_dim, grid_points).
    k_extend : int
        Number of points to extend on both ends.

    Returns
    -------
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

    Parameters
    ----------
    in_dim : int
        Number of input units.
    out_dim : int
        Number of output units.

    Returns
    -------
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
