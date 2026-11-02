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


class Jitter(BaseAugmentation):
    """Add gaussian noise to every value.

    Parameters
    ----------
    sigma : float, default=0.03
        Standard deviation of the noise. Interpreted in units of a standard
        deviation of the scaled series, because augmentation runs after
        scaling; on raw values this default would mean nothing comparable.
    """

    def __init__(self, sigma: float = 0.03):
        self.sigma = sigma
        super().__init__()

    def transform(self, window, rng, neighbours=None):
        out = dict(window)
        for key in ("cont", "target"):
            block = window[key]
            noise = torch.randn(block.shape, generator=rng) * self.sigma
            out[key] = block + noise
        return out


class Scaling(BaseAugmentation):
    """Multiply each channel by its own random factor, constant over time.

    Parameters
    ----------
    sigma : float, default=0.1
        Standard deviation of the factor around 1.0.
    """

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma
        super().__init__()

    def transform(self, window, rng, neighbours=None):
        out = dict(window)
        for key in ("cont", "target"):
            block = window[key]
            n_channels = block.shape[1]
            factor = 1.0 + self.sigma * torch.randn((1, n_channels), generator=rng)
            out[key] = block * factor
        return out


class MagnitudeWarp(BaseAugmentation):
    """Warp amplitude with a smooth random curve along time.

    Parameters
    ----------
    sigma : float, default=0.2
        Standard deviation of the knot values around 1.0.
    knot : int, default=4
        Number of interior knots of the spline.
    """

    def __init__(self, sigma: float = 0.2, knot: int = 4):
        self.sigma = sigma
        self.knot = knot
        super().__init__()

    def transform(self, window, rng, neighbours=None):
        out = dict(window)
        n_steps = window["cont"].shape[0]
        for key in ("cont", "target"):
            block = window[key]
            curve = _random_spline_curve(
                n_steps, block.shape[1], self.sigma, self.knot, rng
            )
            out[key] = block * curve
        return out


class SignFlip(BaseAugmentation):
    """Flip the sign of a random subset of channels.

    This is ``thuml``'s ``rotation`` with the feature permutation removed. That
    permutation is sound for classification, where channels are exchangeable,
    but not here: the data module's ``cont_known_mask`` selects known features
    by column position, so permuting columns would take "known future" values
    from the wrong feature without raising. The class is named for what it
    does rather than for the upstream function, so the difference is visible at
    the call site.
    """

    def transform(self, window, rng, neighbours=None):
        out = dict(window)
        for key in ("cont", "target"):
            block = window[key]
            draw = torch.randint(
                0, 2, (1, block.shape[1]), generator=rng, dtype=block.dtype
            )
            out[key] = block * (draw * 2 - 1)
        return out


class Permutation(BaseAugmentation):
    """Cut the window into equal segments and reorder them.

    ``thuml`` draws the segment count from ``1 .. max_segments - 1`` and
    returns the window untouched when it draws 1, so the transform silently
    does nothing for some fraction of accesses. Here the count is drawn from
    ``2 .. max_segments``, so every access reorders. A transform that
    sometimes no-ops is both weaker as a regulariser and untestable: a test
    asserting the data changed would fail intermittently.

    Parameters
    ----------
    max_segments : int, default=5
        Upper bound on the number of segments. The actual count is drawn
        uniformly from ``2 .. max_segments``.
    """

    _tags = {"touches_time_axis": True}

    def __init__(self, max_segments: int = 5):
        self.max_segments = max_segments
        super().__init__()

    def transform(self, window, rng, neighbours=None):
        n_steps = window["cont"].shape[0]
        n_segments = int(
            torch.randint(2, self.max_segments + 1, (1,), generator=rng).item()
        )
        n_segments = min(n_segments, n_steps)

        positions = torch.arange(n_steps)
        segments = list(torch.chunk(positions, n_segments))
        order = torch.randperm(len(segments), generator=rng)
        src_pos = torch.cat([segments[i] for i in order]).to(torch.float32)

        return self.resample_positions(window, src_pos)


class TimeWarp(BaseAugmentation):
    """Warp the time axis with a smooth random curve.

    The warp is **shared across channels**, unlike ``thuml``, which draws one
    per channel. Warping channels independently would mean that row ``t`` of
    the output mixes different timesteps of different features, destroying the
    cross-feature alignment a forecasting model relies on. Sharing it is also
    what lets the categorical codes follow the same mapping.

    Parameters
    ----------
    sigma : float, default=0.2
        Standard deviation of the knot values around 1.0.
    knot : int, default=4
        Number of interior knots of the spline.
    """

    _tags = {"touches_time_axis": True}

    def __init__(self, sigma: float = 0.2, knot: int = 4):
        self.sigma = sigma
        self.knot = knot
        super().__init__()

    def transform(self, window, rng, neighbours=None):
        n_steps = window["cont"].shape[0]

        # one positive curve, shared by every block and channel
        curve = _random_spline_curve(n_steps, 1, self.sigma, self.knot, rng)
        curve = curve[:, 0].clamp(min=1e-3)

        walk = torch.cumsum(curve, dim=0)
        src_pos = (walk - walk[0]) / (walk[-1] - walk[0]) * (n_steps - 1)

        return self.resample_positions(window, src_pos)


class WindowSlice(BaseAugmentation):
    """Slice a random sub-span and stretch it back to the full length.

    Parameters
    ----------
    reduce_ratio : float, default=0.9
        Fraction of the window the slice covers.
    """

    _tags = {"touches_time_axis": True}

    def __init__(self, reduce_ratio: float = 0.9):
        self.reduce_ratio = reduce_ratio
        super().__init__()

    def transform(self, window, rng, neighbours=None):
        n_steps = window["cont"].shape[0]
        span = max(2, int(round(n_steps * self.reduce_ratio)))
        if span >= n_steps:
            return dict(window)

        start = int(torch.randint(0, n_steps - span + 1, (1,), generator=rng).item())
        src_pos = torch.linspace(float(start), float(start + span - 1), n_steps)

        return self.resample_positions(window, src_pos)


class WindowWarp(BaseAugmentation):
    """Speed up or slow down one random window, keeping the edges in place.

    Parameters
    ----------
    window_ratio : float, default=0.1
        Fraction of the series the warped window covers.
    """

    _tags = {"touches_time_axis": True}

    def __init__(self, window_ratio: float = 0.1, scales=(0.5, 2.0)):
        self.window_ratio = window_ratio
        self.scales = scales
        super().__init__()

    def transform(self, window, rng, neighbours=None):
        n_steps = window["cont"].shape[0]
        warp_size = max(2, int(round(n_steps * self.window_ratio)))
        if warp_size >= n_steps - 2:
            return dict(window)

        start = int(
            torch.randint(1, n_steps - warp_size - 1, (1,), generator=rng).item()
        )
        end = start + warp_size

        pick = int(torch.randint(0, len(self.scales), (1,), generator=rng).item())
        scale = float(self.scales[pick])

        warped = torch.linspace(
            float(start), float(end - 1), max(2, int(round(warp_size * scale)))
        )
        src_pos = torch.cat(
            [
                torch.arange(0, start, dtype=torch.float32),
                warped,
                torch.arange(end, n_steps, dtype=torch.float32),
            ]
        )
        # stretch the concatenated path back onto the original step count
        src_pos = torch.nn.functional.interpolate(
            src_pos.view(1, 1, -1), size=n_steps, mode="linear", align_corners=True
        ).view(-1)

        return self.resample_positions(window, src_pos)
