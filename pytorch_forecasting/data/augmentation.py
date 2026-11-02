"""Augmentation transforms for time series windows.

The transforms here perturb a single training window each time it is accessed,
so the same window looks different in every epoch. They are applied by
``_TslibDataset.__getitem__`` on the training split only, and are off unless
explicitly enabled.

The set follows ``thuml``'s ``utils/augmentation.py``, with two deliberate
deviations, both documented on the classes concerned: ``rotation`` keeps only
the sign flip, and time axis warps are shared across channels.
"""

__author__ = ["echo-xiao"]

from abc import abstractmethod

import numpy as np
import torch

from pytorch_forecasting.base._base_object import _BaseObject


class BaseAugmentation(_BaseObject):
    """Perturb a training window. Called once per ``__getitem__``.

    Subclasses declare what they need through tags, and the data module
    dispatches on those tags rather than on isinstance checks.

    Tags
    ----
    requires_neighbours : bool
        Whether the transform needs a pool of same-label windows. Only
        transforms declaring ``True`` are handed a ``NeighbourIndex``.
    touches_time_axis : bool
        Whether the transform reorders or resamples along time. Those
        transforms must go through ``resample_positions`` so that continuous,
        categorical and target blocks all receive the same position mapping.
    """

    _tags = {
        "object_type": "augmentation",
        "requires_neighbours": False,
        "touches_time_axis": False,
    }

    @abstractmethod
    def transform(
        self,
        window: dict,
        rng: torch.Generator,
        neighbours=None,
    ) -> dict:
        """Return a new window dict. Must not mutate the input.

        Parameters
        ----------
        window : dict
            ``cont`` of shape ``(ctx + pred, n_cont)``, ``cat`` of shape
            ``(ctx + pred, n_cat)``, ``target`` of shape
            ``(ctx + pred, n_targets)``, plus ``ctx_len`` and ``pred_len``.
            ``target`` is always 2D, with ``n_targets == 1`` in the univariate
            case.
        rng : torch.Generator
            The only source of randomness a transform may use.
        neighbours : NeighbourIndex, optional
            Non-``None`` only when the transform declares
            ``requires_neighbours``.
        """

    @staticmethod
    def resample_positions(window: dict, src_pos: torch.Tensor) -> dict:
        """Resample every block onto float source positions ``src_pos``.

        ``cont`` and ``target`` are interpolated linearly. ``cat`` snaps to the
        nearest step, because categorical codes carry no ordering that would
        make an interpolated value meaningful.

        Applying one position mapping to all three blocks is what keeps them
        aligned: a transform that warped only the continuous features would
        leave the categorical codes describing a different timestep, and
        nothing would raise.

        Parameters
        ----------
        window : dict
            the window to resample; left unmodified
        src_pos : torch.Tensor
            float positions into the input window, one per output step

        Returns
        -------
        dict
            a new window dict with the same ``ctx_len`` and ``pred_len``
        """
        n_steps = window["cont"].shape[0]
        pos = src_pos.clamp(0, n_steps - 1)

        lower = pos.floor().long()
        upper = (lower + 1).clamp(max=n_steps - 1)
        weight = (pos - lower.to(pos.dtype)).unsqueeze(-1)

        nearest = pos.round().long().clamp(0, n_steps - 1)

        out = dict(window)
        for key in ("cont", "target"):
            block = window[key]
            out[key] = block[lower] * (1 - weight) + block[upper] * weight
        out["cat"] = window["cat"][nearest]
        return out


def _random_spline_curve(
    n_steps: int,
    n_channels: int,
    sigma: float,
    knot: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """A smooth random curve centred on 1.0, of shape ``(n_steps, n_channels)``.

    The knot values are drawn from ``rng`` so that seeding is reproducible and
    per-worker; ``scipy`` only interpolates between them. ``thuml`` draws these
    from ``numpy.random``, which is not reseeded per dataloader worker and can
    therefore give every worker the same curve.
    """
    from scipy.interpolate import CubicSpline

    knot_x = np.linspace(0, n_steps - 1, knot + 2)
    knot_y = 1.0 + sigma * torch.randn(knot + 2, n_channels, generator=rng)
    knot_y = knot_y.numpy()

    steps = np.arange(n_steps)
    channels = [CubicSpline(knot_x, knot_y[:, c])(steps) for c in range(n_channels)]
    curve = np.stack(channels, axis=1)
    return torch.tensor(curve, dtype=torch.float32)
