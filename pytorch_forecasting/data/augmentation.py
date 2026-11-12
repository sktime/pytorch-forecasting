"""Augmentation transforms for time series windows.

The transforms here perturb a single training window each time it is accessed,
so the same window looks different in every epoch. They are applied by
``_TslibDataset.__getitem__`` on the training split only, and are off unless
explicitly enabled.

The set follows ``thuml``'s ``utils/augmentation.py``. Where it departs from
upstream the reason is documented on the class concerned: ``rotation`` keeps
only the sign flip, time axis warps are shared across channels, ``Permutation``
always reorders, and ``DiscriminativeGuidedWarp`` omits ``use_variable_slice``.
Only the symmetric slope constraint is implemented, so ``slope_constraint`` is
not exposed.
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
            case. ``window_idx`` carries the window's position in the
            dataset's window list, which is how the label guided transforms
            address ``neighbours``.
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

    def transform(
        self,
        window: dict,
        rng: torch.Generator,
        neighbours=None,
    ) -> dict:
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

    def transform(
        self,
        window: dict,
        rng: torch.Generator,
        neighbours=None,
    ) -> dict:
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

    def transform(
        self,
        window: dict,
        rng: torch.Generator,
        neighbours=None,
    ) -> dict:
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

    def transform(
        self,
        window: dict,
        rng: torch.Generator,
        neighbours=None,
    ) -> dict:
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

    Segments come from ``torch.chunk``, which is not ``numpy.array_split``:
    it returns *at most* the requested count, fewer when the length does not
    divide evenly, and distributes the remainder differently. So the effective
    segmentation differs from upstream's even where the drawn count agrees.

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

    def transform(
        self,
        window: dict,
        rng: torch.Generator,
        neighbours=None,
    ) -> dict:
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

    def transform(
        self,
        window: dict,
        rng: torch.Generator,
        neighbours=None,
    ) -> dict:
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

    def transform(
        self,
        window: dict,
        rng: torch.Generator,
        neighbours=None,
    ) -> dict:
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
    scales : tuple of float, default=(0.5, 2.0)
        Candidate speed factors; one is chosen at random to stretch or
        compress the warped window.
    """

    _tags = {"touches_time_axis": True}

    def __init__(self, window_ratio: float = 0.1, scales=(0.5, 2.0)):
        self.window_ratio = window_ratio
        self.scales = scales
        super().__init__()

    def transform(
        self,
        window: dict,
        rng: torch.Generator,
        neighbours=None,
    ) -> dict:
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


class NeighbourIndex:
    """A pool of same-label windows, for the label guided transforms.

    The guided transforms warp a window towards another window of the same
    class, so they need to see windows other than their own. ``transform``'s
    signature only carries the current window, so the pool is handed in
    separately, and only to transforms declaring ``requires_neighbours``.

    Lookups go through ``dataset._slice_window``, never ``__getitem__``: a
    neighbour must be a raw window, or fetching one would recurse back into
    augmentation and need a re-entrancy flag to stop.

    Parameters
    ----------
    dataset : _TslibDataset
        the dataset the window indices refer to
    labels : torch.Tensor
        class label per window, shape ``(n_windows,)``
    series_of_window : list of int
        the series each window was cut from; a neighbour must come from a
        different series, because ``thuml``'s notion of a sample maps to our
        series rather than our window

    Raises
    ------
    ValueError
        if any label covers only one series, which would leave windows of that
        label with an empty pool
    """

    def __init__(self, dataset, labels: torch.Tensor, series_of_window: list):
        self.dataset = dataset
        self.labels = labels
        self.series_of_window = series_of_window

        self._by_label = {}
        for window_idx, label in enumerate(labels.tolist()):
            self._by_label.setdefault(label, []).append(window_idx)

        for label, members in self._by_label.items():
            series = {series_of_window[i] for i in members}
            if len(series) < 2:
                raise ValueError(
                    f"label {label!r} covers only one series in the training "
                    "split, so windows with that label have no same-label "
                    "neighbour from another series. Guided augmentations need "
                    "at least two series per label there. Note the split is a "
                    "random, unstratified permutation of the series, so a "
                    "dataset that looks balanced overall can still land a "
                    "label on a single training series: give each label enough "
                    "series that it keeps two for any draw, or widen "
                    "train_val_test_split."
                )

    def sample_index(self, window_idx: int, rng: torch.Generator) -> int:
        """Index of a same-label window from a different series."""
        label = int(self.labels[window_idx].item())
        own_series = self.series_of_window[window_idx]
        candidates = [
            i for i in self._by_label[label] if self.series_of_window[i] != own_series
        ]
        pick = int(torch.randint(0, len(candidates), (1,), generator=rng).item())
        return candidates[pick]

    def sample(self, window_idx: int, rng: torch.Generator) -> dict:
        """A raw, unaugmented same-label window from a different series."""
        return self.dataset._slice_window(self.sample_index(window_idx, rng))

    def sample_other_label_index(self, window_idx: int, rng: torch.Generator) -> int:
        """Index of a window carrying a different label.

        ``DiscriminativeGuidedWarp`` picks the prototype that is closest to the
        same class and furthest from other classes, so it needs negative
        examples as well as positive ones.
        """
        label = int(self.labels[window_idx].item())
        candidates = [
            i
            for lab, members in self._by_label.items()
            if lab != label
            for i in members
        ]
        if not candidates:
            raise ValueError(
                "the training split carries a single label, so there are no "
                "differently-labelled windows. DiscriminativeGuidedWarp needs "
                "at least two labels."
            )
        pick = int(torch.randint(0, len(candidates), (1,), generator=rng).item())
        return candidates[pick]

    @staticmethod
    def as_array(window: dict) -> torch.Tensor:
        """Flatten a window into the single array DTW needs.

        Continuous features and the target, concatenated along the feature
        axis. The categorical block is left out: integer codes have no
        ordering, so a euclidean distance over them is meaningless.
        """
        return torch.cat([window["cont"], window["target"]], dim=1)


def _cummulative_matrix(cost: torch.Tensor, window: int) -> np.ndarray:
    """Accumulate a local cost matrix under the symmetric slope constraint.

    Port of ``_cummulative_matrix`` in ``thuml``'s ``dtw.py``, symmetric branch
    only. The result is one larger than ``cost`` in each dimension, with the
    extra row and column acting as the boundary.

    The recurrence is inherently sequential, so it runs as a Python loop. It
    accumulates in ``float64`` numpy rather than in torch: building a small
    tensor per cell costs about twenty times more than the same loop over
    numpy floats, which on a 96/24 window is the difference between tens of
    milliseconds and a second per item for the guided transforms. ``float64``
    also matches what upstream accumulates in.
    """
    cost_np = cost.detach().cpu().numpy().astype(np.float64, copy=False)
    p, s = cost_np.shape

    dtw = np.full((p + 1, s + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, p + 1):
        for j in range(max(1, i - window), min(s, i + window) + 1):
            dtw[i, j] = cost_np[i - 1, j - 1] + min(
                dtw[i - 1, j - 1], dtw[i, j - 1], dtw[i - 1, j]
            )
    return dtw


def _traceback(dtw: np.ndarray):
    """Walk an accumulated matrix back into an alignment path.

    Port of ``_traceback`` in ``thuml``'s ``dtw.py``, symmetric branch only.
    The branch order reproduces ``argmin`` over ``(diagonal, up, left)``, which
    takes the first minimum on a tie, as both torch and upstream numpy do.
    """
    i = dtw.shape[0] - 1
    j = dtw.shape[1] - 1
    path_p = [i - 1]
    path_q = [j - 1]

    while i > 1 or j > 1:
        diagonal = dtw[i - 1, j - 1]
        up = dtw[i - 1, j]
        left = dtw[i, j - 1]
        if diagonal <= up and diagonal <= left:
            i, j = i - 1, j - 1
        elif up <= left:
            i = i - 1
        else:
            j = j - 1
        path_p.insert(0, i - 1)
        path_q.insert(0, j - 1)

    return torch.tensor(path_p), torch.tensor(path_q)


def _pointwise_cost(
    prototype: torch.Tensor, sample: torch.Tensor, window: int
) -> torch.Tensor:
    """Band-limited matrix of pointwise distances, as ``thuml``'s ``dtw`` builds it.

    Factored out of ``_dtw_path`` only because ``Wdba`` needs the accumulated
    value as well as the path, and two copies of this loop would be free to
    drift apart. The body is ``dtw.py:64-68`` unchanged.
    """
    p = prototype.shape[0]
    s = sample.shape[0]

    cost = torch.full((p, s), float("inf"))
    for i in range(p):
        start = max(0, i - window)
        end = min(s, i + window) + 1
        cost[i, start:end] = torch.linalg.norm(sample[start:end] - prototype[i], dim=1)
    return cost


def _dtw_path(prototype: torch.Tensor, sample: torch.Tensor, window=None):
    """Alignment path between two multivariate series, pointwise distance.

    Port of ``dtw`` in ``thuml``'s ``dtw.py`` with ``return_flag=RETURN_PATH``
    and ``slope_constraint="symmetric"``.

    Parameters
    ----------
    prototype, sample : torch.Tensor
        shapes ``(p, n_features)`` and ``(s, n_features)``
    window : int, optional
        Sakoe-Chiba band half-width. ``None`` means no constraint. Callers
        pass ``ceil(n_steps / 10)``, as ``thuml`` does.

    Returns
    -------
    tuple of torch.Tensor
        two index sequences of equal length, giving the aligned pairs
    """
    p = prototype.shape[0]
    s = sample.shape[0]
    if p == 0 or s == 0:
        raise ValueError("dtw needs two non-empty series")
    if window is None:
        window = s

    cost = _pointwise_cost(prototype, sample, window)

    return _traceback(_cummulative_matrix(cost, window))


def _dtw_value(prototype: torch.Tensor, sample: torch.Tensor, window=None) -> float:
    """Accumulated DTW distance between two series.

    Port of ``dtw`` in ``thuml``'s ``dtw.py`` with the default
    ``return_flag=RETURN_VALUE``, which returns ``DTW[-1, -1]``.
    """
    p = prototype.shape[0]
    s = sample.shape[0]
    if p == 0 or s == 0:
        raise ValueError("dtw needs two non-empty series")
    if window is None:
        window = s

    cost = _pointwise_cost(prototype, sample, window)

    return float(_cummulative_matrix(cost, window)[-1, -1].item())


def _shape_cost(
    prototype: torch.Tensor,
    sample: torch.Tensor,
    window: int,
    descr_ratio: float,
) -> torch.Tensor:
    """Band-limited matrix of shape descriptor distances, as ``shape_dtw`` builds it.

    Each position is described by a short slice of the edge-padded series
    rather than by its own value, so the cost compares local shapes. The body
    is ``dtw.py:120-137`` unchanged. Factored out so that the path and the
    accumulated value cannot drift apart, exactly as ``_pointwise_cost`` is.
    """
    p = prototype.shape[0]
    s = sample.shape[0]

    p_len = int(min(100, max(5, round(p * descr_ratio))))
    s_len = int(min(100, max(5, round(s * descr_ratio))))

    def _pad(series, feature_len):
        front = int(np.ceil(feature_len / 2.0))
        back = int(np.floor(feature_len / 2.0))
        first = series[:1].expand(front, -1)
        last = series[-1:].expand(back, -1)
        return torch.cat([first, series, last], dim=0)

    prototype_pad = _pad(prototype, p_len)
    sample_pad = _pad(sample, s_len)

    cost = torch.full((p, s), float("inf"))
    for i in range(p):
        descr_p = prototype_pad[i : i + p_len]
        for j in range(max(0, i - window), min(s, i + window)):
            descr_s = sample_pad[j : j + s_len]
            cost[i, j] = torch.linalg.norm(descr_s - descr_p)
    return cost


def _shape_dtw_path(
    prototype: torch.Tensor,
    sample: torch.Tensor,
    window=None,
    descr_ratio: float = 0.05,
):
    """Alignment path using shape descriptors instead of pointwise distance.

    Port of ``shape_dtw`` in ``thuml``'s ``dtw.py``, symmetric branch only,
    with ``return_flag=RETURN_PATH``. This is what the ``*Shape`` augmentation
    variants align on; it is not the same as running the pointwise distance on
    a differenced series.
    """
    p = prototype.shape[0]
    s = sample.shape[0]
    if p == 0 or s == 0:
        raise ValueError("shape dtw needs two non-empty series")
    if window is None:
        window = s

    cost = _shape_cost(prototype, sample, window, descr_ratio)

    return _traceback(_cummulative_matrix(cost, window))


def _shape_dtw_value(
    prototype: torch.Tensor,
    sample: torch.Tensor,
    window=None,
    descr_ratio: float = 0.05,
) -> float:
    """Accumulated shape DTW distance between two series.

    Port of ``shape_dtw`` in ``thuml``'s ``dtw.py`` with the default
    ``return_flag=RETURN_VALUE``, which returns ``DTW[-1, -1]``. Needed because
    ``discriminative_guided_warp``'s shape branch scores its candidate
    prototypes with ``shape_dtw(..., RETURN_VALUE)``, not with the pointwise
    distance.
    """
    p = prototype.shape[0]
    s = sample.shape[0]
    if p == 0 or s == 0:
        raise ValueError("shape dtw needs two non-empty series")
    if window is None:
        window = s

    cost = _shape_cost(prototype, sample, window, descr_ratio)

    return float(_cummulative_matrix(cost, window)[-1, -1].item())


_DISTANCE_FOR_ALIGNMENT = {
    _dtw_path: _dtw_value,
    _shape_dtw_path: _shape_dtw_value,
}
"""The ``RETURN_VALUE`` counterpart of each alignment function.

``DiscriminativeGuidedWarp`` needs both a path and a scalar distance, and
upstream takes both from the same ``dtw`` / ``shape_dtw`` call. Deriving the
distance from the chosen alignment function keeps the pair in step: a subclass
that switches ``_align`` cannot end up scoring with the other cost.
"""


def _stretch_path(path: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Spread an alignment path of any length over ``n_steps`` output positions.

    ``thuml`` finishes its guided transforms with
    ``np.interp(orig_steps, np.linspace(0, T - 1, len(path)), values[path])``,
    interpolating the *values* it gathered along the path. Returning float
    source positions instead is equivalent, because a symmetric DTW path steps
    by at most one index at a time: blending ``series[k]`` with ``series[k + 1]``
    and reading ``series`` at ``k + w`` are the same number. Positions are what
    ``resample_positions`` needs in order to carry the categorical block along.
    """
    length = path.shape[0]
    positions = path.to(torch.float32)
    if n_steps == 1 or length == 1:
        return positions[:1].expand(n_steps).clone()

    grid = torch.arange(n_steps, dtype=torch.float32) * (length - 1) / (n_steps - 1)
    lower = grid.floor().long().clamp(0, length - 1)
    upper = (lower + 1).clamp(max=length - 1)
    weight = grid - lower.to(grid.dtype)
    return positions[lower] * (1 - weight) + positions[upper] * weight


def _window_index(window: dict, name: str) -> int:
    """The window's own index, or a clear error naming what is missing."""
    window_idx = window.get("window_idx")
    if window_idx is None:
        raise ValueError(
            f"{name} needs 'window_idx' in the window dict to look its "
            "neighbour pool up. The data module supplies it; a transform "
            "called on a hand-built window must supply it too."
        )
    return int(window_idx)


def _require_neighbours(neighbours, name: str) -> None:
    """Refuse to run without a pool, rather than silently doing nothing."""
    if neighbours is None:
        raise ValueError(
            f"{name} needs a same-label neighbour pool. It is only usable on "
            "the training split of a data module configured with "
            "augmentation_label."
        )


def _draw_distinct(sampler, window_idx: int, rng: torch.Generator, wanted: int) -> list:
    """Distinct window indices, standing in for ``choice(..., replace=False)``.

    ``NeighbourIndex`` samples one index at a time, so distinctness is enforced
    by rejection. The attempt budget bounds the loop when the pool is smaller
    than ``wanted``; drawing fewer is the same situation as upstream's
    ``k = min(choices.size, batch_size)``.

    Parameters
    ----------
    sampler : callable
        ``(window_idx, rng) -> int``, one of ``NeighbourIndex``'s samplers
    window_idx : int
        the window asking for neighbours
    rng : torch.Generator
        the only source of randomness
    wanted : int
        how many distinct indices to aim for

    Returns
    -------
    list of int
        up to ``wanted`` distinct window indices
    """
    picked = []
    seen = set()
    for _ in range(4 * (wanted + 1)):
        if len(picked) >= wanted:
            break
        candidate = sampler(window_idx, rng)
        if candidate in seen:
            continue
        seen.add(candidate)
        picked.append(candidate)
    return picked


class Spawner(BaseAugmentation):
    """Average the window with a same-label neighbour along their DTW path.

    Ported from ``thuml``'s ``spawner``. The neighbour is drawn from the
    training split only, via ``NeighbourIndex``, and always from a different
    series.

    Following ``thuml``, the alignment is computed in two independent halves
    split at a random step, which lets the two series trade time differently
    before and after the split. The DTW band is ``ceil(n_steps / 10)``
    computed on the *full* window, as upstream does, not on each half.

    Parameters
    ----------
    sigma : float, default=0.05
        Standard deviation of the jitter applied after the average.
    """

    _tags = {
        "requires_neighbours": True,
        "touches_time_axis": True,
    }

    def __init__(self, sigma: float = 0.05):
        self.sigma = sigma
        super().__init__()

    def _jitter(self, out: dict, rng: torch.Generator) -> dict:
        for key in ("cont", "target"):
            noise = torch.randn(out[key].shape, generator=rng) * self.sigma
            out[key] = out[key] + noise
        return out

    def transform(
        self,
        window: dict,
        rng: torch.Generator,
        neighbours=None,
    ) -> dict:
        _require_neighbours(neighbours, "Spawner")
        window_idx = _window_index(window, "Spawner")

        n_steps = window["cont"].shape[0]
        if n_steps < 3:
            # upstream draws the split from ``randint(1, n_steps - 1)``, which
            # is empty below three steps; nothing to align, so only jitter.
            return self._jitter(dict(window), rng)

        split = int(torch.randint(1, n_steps - 1, (1,), generator=rng).item())
        band = int(np.ceil(n_steps / 10.0))

        neighbour = neighbours.sample(window_idx, rng)
        own = NeighbourIndex.as_array(window)
        other = NeighbourIndex.as_array(neighbour)

        head_own, head_other = _dtw_path(own[:split], other[:split], window=band)
        tail_own, tail_other = _dtw_path(own[split:], other[split:], window=band)
        path_own = torch.cat([head_own, tail_own + split])
        path_other = torch.cat([head_other, tail_other + split])

        out = self.resample_positions(window, _stretch_path(path_own, n_steps))
        warped = self.resample_positions(neighbour, _stretch_path(path_other, n_steps))
        for key in ("cont", "target"):
            out[key] = 0.5 * (out[key] + warped[key])

        return self._jitter(out, rng)


class Wdba(BaseAugmentation):
    """Weighted DBA: replace the window with a weighted average of its class.

    Ported from ``thuml``'s ``wdba``. A handful of same-label windows are
    drawn, the one closest to all the others under DTW becomes the medoid, and
    the rest are warped onto the medoid's time axis and averaged in with a
    weight that halves at the distance of the medoid's nearest neighbour.

    Two adaptations are worth knowing about, both forced by the fact that
    ``NeighbourIndex`` deliberately excludes the window's own series:

    * Upstream's intra-class pool includes the window itself, so
      ``np.random.choice`` may or may not pick it. Here the window is always
      the first prototype and the rest are sampled neighbours, which keeps the
      augmented window anchored to its own data.
    * The categorical block is left on the window's own time axis. The
      averaged values live on the medoid's index axis, which runs ``0..T-1``
      just as the window's own does, so output step ``t`` still describes step
      ``t`` and no categorical code moves.

    Parameters
    ----------
    batch_size : int, default=6
        How many same-label windows to average, the window itself included.
    """

    _tags = {
        "requires_neighbours": True,
        "touches_time_axis": True,
    }

    def __init__(self, batch_size: int = 6):
        self.batch_size = batch_size
        super().__init__()

    def _draw_prototypes(self, neighbours, window_idx, rng):
        """Distinct neighbour windows, standing in for ``choice(replace=False)``."""
        picked = _draw_distinct(
            neighbours.sample_index, window_idx, rng, self.batch_size - 1
        )
        return [neighbours.dataset._slice_window(i) for i in picked]

    def transform(
        self,
        window: dict,
        rng: torch.Generator,
        neighbours=None,
    ) -> dict:
        _require_neighbours(neighbours, "Wdba")
        window_idx = _window_index(window, "Wdba")

        n_steps = window["cont"].shape[0]
        band = int(np.ceil(n_steps / 10.0))

        prototypes = [window] + self._draw_prototypes(neighbours, window_idx, rng)
        n_protos = len(prototypes)
        identity = torch.arange(n_steps, dtype=torch.float32)
        out = self.resample_positions(window, identity)
        if n_protos < 2:
            return out

        arrays = [NeighbourIndex.as_array(proto) for proto in prototypes]

        distances = torch.zeros((n_protos, n_protos))
        for p in range(n_protos):
            for s in range(n_protos):
                if p != s:
                    distances[p, s] = _dtw_value(arrays[p], arrays[s], window=band)

        # ``argsort(...)[0]`` rather than ``argmin``: upstream relies on numpy
        # breaking ties by index, and ties are common here because the matrix
        # is symmetric under the symmetric slope constraint.
        medoid = int(torch.argsort(distances.sum(dim=1), stable=True)[0].item())
        nearest_order = torch.argsort(distances[medoid], stable=True).tolist()
        # ``nearest_order[0]`` is the medoid itself, at distance zero.
        reference = float(distances[medoid, nearest_order[1]].item())

        totals = {key: torch.zeros_like(out[key]) for key in ("cont", "target")}
        total_weight = 0.0
        for nid in nearest_order:
            if nid == medoid or reference == 0.0:
                source = prototypes[medoid]
                src_pos = identity
                weight = 1.0
            else:
                path_medoid, path_other = _dtw_path(
                    arrays[medoid], arrays[nid], window=band
                )
                src_pos = _last_aligned_position(path_medoid, path_other, n_steps)
                value = float(distances[medoid, nid].item())
                weight = float(np.exp(np.log(0.5) * value / reference))
                source = prototypes[nid]

            warped = self.resample_positions(source, src_pos)
            for key in ("cont", "target"):
                totals[key] = totals[key] + weight * warped[key]
            total_weight += weight

        for key in ("cont", "target"):
            out[key] = totals[key] / total_weight
        return out


def _last_aligned_position(
    path_prototype: torch.Tensor, path_sample: torch.Tensor, n_steps: int
) -> torch.Tensor:
    """Which sample step each prototype step reads, keeping the *last* match.

    ``thuml``'s ``wdba`` accumulates with
    ``average_pattern[path[0]] += weight * warped``. A DTW path repeats
    prototype indices wherever the sample runs faster, and numpy's buffered
    fancy-index assignment does not accumulate over duplicates: the last write
    wins. So each prototype step ends up reading exactly one sample step, the
    one from the final occurrence of that index in the path. Reproducing that
    is what makes this port agree with upstream numerically; ``index_add_``
    would accumulate instead and quietly give different numbers.

    The same applies to ``weighted_sums[path[0]] += weight``, which is why the
    divisor in ``Wdba`` is a single scalar rather than a per-step vector.
    """
    aligned = list(range(n_steps))
    for prototype_step, sample_step in zip(
        path_prototype.tolist(), path_sample.tolist()
    ):
        if 0 <= prototype_step < n_steps:
            aligned[prototype_step] = sample_step
    return torch.tensor(aligned, dtype=torch.float32)


class RandomGuidedWarp(BaseAugmentation):
    """Warp the window onto a same-label neighbour's DTW path.

    Ported from ``thuml``'s ``random_guided_warp``. A same-label window from
    another series is drawn as the prototype, the window is aligned to it under
    DTW with a Sakoe-Chiba band of ``ceil(n_steps / 10)``, and the window is
    then read back along that alignment. The result keeps the window's own
    values and only redistributes them in time, towards the prototype's pace.

    Notes
    -----
    ``thuml`` exposes a ``slope_constraint`` and an ``asymmetric`` mode here.
    Every call site in upstream defaults to ``"symmetric"``, so that mode is
    the only one implemented and the parameter is not exposed.

    Upstream's intra-class candidate set is built with
    ``np.where(l[np.delete(np.arange(n), i)] == l[i])[0]``, whose indices are
    positions in the *reduced* array but are then used to index the full array.
    That off-by-one lets upstream pick the window itself, or a window of a
    different class. ``NeighbourIndex`` selects correctly instead.
    """

    _tags = {
        "requires_neighbours": True,
        "touches_time_axis": True,
    }

    _align = staticmethod(_dtw_path)

    def transform(
        self,
        window: dict,
        rng: torch.Generator,
        neighbours=None,
    ) -> dict:
        name = type(self).__name__
        _require_neighbours(neighbours, name)
        window_idx = _window_index(window, name)

        n_steps = window["cont"].shape[0]
        band = int(np.ceil(n_steps / 10.0))

        neighbour = neighbours.sample(window_idx, rng)
        own = NeighbourIndex.as_array(window)
        other = NeighbourIndex.as_array(neighbour)

        # upstream aligns ``dtw(prototype=neighbour, sample=own)`` and gathers
        # the own window along the sample side of the path
        _, path_own = type(self)._align(other, own, window=band)

        return self.resample_positions(window, _stretch_path(path_own, n_steps))


class RandomGuidedWarpShape(RandomGuidedWarp):
    """``RandomGuidedWarp`` aligning on shape descriptors instead of points.

    Ported from ``thuml``'s ``random_guided_warp_shape``, which is
    ``random_guided_warp`` with ``dtw_type="shape"``. The only difference is
    the cost the alignment minimises: a distance between short slices of the
    edge-padded series rather than between single timesteps.
    """

    _align = staticmethod(_shape_dtw_path)


class DiscriminativeGuidedWarp(BaseAugmentation):
    """Warp the window onto the most discriminative same-label prototype.

    Ported from ``thuml``'s ``discriminative_guided_warp``. Several same-label
    and several differently-labelled windows are drawn; each same-label
    candidate is scored by its mean DTW distance to the other same-label
    candidates and to the differently-labelled ones, and the candidate
    maximising ``mean distance to other labels - mean distance within the
    label`` becomes the prototype. The window is then read back along its
    alignment to that prototype, as in :class:`RandomGuidedWarp`.

    Notes
    -----
    ``slope_constraint`` is not exposed, for the reason given on
    :class:`RandomGuidedWarp`.

    Upstream's ``use_variable_slice=True`` tail is **not** ported. It slices
    every row of the batch with ``reduce_ratio = 0.9 + 0.1 * warp_amount[i] /
    max(warp_amount)``, so the ratio is defined only relative to the other rows
    of the same batch. A transform acting on one window at a time has no batch:
    its own warp is the maximum, the ratio is exactly ``1.0``, and upstream's
    ``window_slice`` returns its input unchanged at that ratio. The only
    surviving case would be the batch-wide degenerate one where *no* row warped
    at all, which cannot be decided from a single window. Compose
    :class:`WindowSlice` explicitly if slicing is wanted.

    Parameters
    ----------
    batch_size : int, default=6
        How many candidates to compare. ``ceil(batch_size / 2)`` same-label
        candidates and ``floor(batch_size / 2)`` differently-labelled ones,
        as upstream splits it.
    """

    _tags = {
        "requires_neighbours": True,
        "touches_time_axis": True,
    }

    _align = staticmethod(_dtw_path)

    def __init__(self, batch_size: int = 6):
        self.batch_size = batch_size
        super().__init__()

    def _select_prototype(self, positives, negatives, distance, band) -> int:
        """Index of the candidate closest to its own class, furthest from others.

        ``augmentation.py:285-303``: the scores are means, the within-class one
        over the ``pos_k - 1`` other positives and the between-class one over
        all ``neg_k`` negatives.
        """
        n_pos = len(positives)
        n_neg = len(negatives)

        scores = torch.zeros(n_pos)
        for p in range(n_pos):
            within = 0.0
            for other in range(n_pos):
                if other != p:
                    within += distance(positives[p], positives[other], window=band)
            if n_pos > 1:
                within = within / (n_pos - 1)
            between = 0.0
            for negative in negatives:
                between += distance(positives[p], negative, window=band)
            scores[p] = between / n_neg - within

        return int(torch.argmax(scores).item())

    def transform(
        self,
        window: dict,
        rng: torch.Generator,
        neighbours=None,
    ) -> dict:
        name = type(self).__name__
        _require_neighbours(neighbours, name)
        window_idx = _window_index(window, name)

        n_steps = window["cont"].shape[0]
        band = int(np.ceil(n_steps / 10.0))

        positive_batch = int(np.ceil(self.batch_size / 2.0))
        negative_batch = int(np.floor(self.batch_size / 2.0))

        positive_ids = _draw_distinct(
            neighbours.sample_index, window_idx, rng, positive_batch
        )
        negative_ids = _draw_distinct(
            neighbours.sample_other_label_index, window_idx, rng, negative_batch
        )

        identity = torch.arange(n_steps, dtype=torch.float32)
        if not positive_ids or not negative_ids:
            # upstream's ``else`` branch returns the pattern untouched
            return self.resample_positions(window, identity)

        fetch = neighbours.dataset._slice_window
        positives = [NeighbourIndex.as_array(fetch(i)) for i in positive_ids]
        negatives = [NeighbourIndex.as_array(fetch(i)) for i in negative_ids]

        align = type(self)._align
        distance = _DISTANCE_FOR_ALIGNMENT[align]
        selected = self._select_prototype(positives, negatives, distance, band)

        own = NeighbourIndex.as_array(window)
        _, path_own = align(positives[selected], own, window=band)

        return self.resample_positions(window, _stretch_path(path_own, n_steps))


class DiscriminativeGuidedWarpShape(DiscriminativeGuidedWarp):
    """``DiscriminativeGuidedWarp`` scoring and aligning on shape descriptors.

    Ported from ``thuml``'s ``discriminative_guided_warp_shape``, which is
    ``discriminative_guided_warp`` with ``dtw_type="shape"``. Upstream switches
    both the candidate scoring and the final alignment to ``shape_dtw``, which
    is why the distance used for scoring is looked up from ``_align`` rather
    than fixed.
    """

    _align = staticmethod(_shape_dtw_path)
