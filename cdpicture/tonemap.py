"""How a picture loses contrast when files need room: evenly, or keeping its detail.

At contrast c the picture must fit a band of width c around the data tone (hide.squeeze). How it
gets there doesn't change the room for files: nearly every sector's stretch of track crosses the
picture's darkest and lightest parts, so only the band width counts. It does change what is seen.

`even` squeezes every tone toward the data tone alike, so fine edges lose as much as broad
shading. `detail` splits the picture into layers and squeezes the coarse ones harder than the fine
ones: at contrast c a layer keeps c^p of its amplitude, p small for fine detail and near 1 for
broad shading. Under low contrast the eye discounts slow shading far more than edges. The finest
layer (under 0.4 mm) is a plain Gaussian split, so boosting it overshoots edges only by a
fraction of a millimetre, which reads as sharpening. The coarser layers are edge-aware
(self-guided filters), because a Gaussian there leaves halos millimetres wide along strong
edges. At c = 1 both modes return the picture unchanged.
"""

from __future__ import annotations

import numpy as np

from . import design as D

FINE_MM = 0.4               # Gaussian sigma of the finest layer
RADII_MM = (1.6, 6.4)       # edge-aware layer boundaries above it
POWERS = (0.0, 0.3, 0.9)    # finest layer .. coarsest; the base below the last one takes the rest
EPS = 0.01                  # guided filter edge threshold, in tone^2: edges stronger than ~0.1 are kept
CLIP = 2.0                   # % of the picture at each end that may roll off at full squeeze


def guided(img, radius_px, eps):
    """Self-guided filter (He, Sun, Tang 2013): edge-preserving smoothing over a box of this radius."""
    from scipy.ndimage import uniform_filter
    k = 2 * radius_px + 1
    m = uniform_filter(img, k)
    var = uniform_filter(img * img, k) - m * m
    a = var / (var + eps)
    return uniform_filter(a, k) * img + uniform_filter(m - a * m, k)


class Even:
    """Today's squeeze: the picture as is, hide.squeeze does the rest."""

    fixed = True  # the same tone at every contrast

    def __init__(self, tone):
        self.tone = tone

    def __call__(self, contrast):
        return self.tone


class Detail:
    """Detail-keeping tone at a contrast, still 0..1: hide.squeeze then puts it in the band."""

    def __init__(self, tone, fine_mm=FINE_MM, radii_mm=RADII_MM, powers=POWERS, eps=EPS, clip=CLIP):
        self.tone, self.fine_mm, self.radii_mm = tone, fine_mm, radii_mm
        self.powers, self.eps, self.clip = powers, eps, clip
        self.layers = None  # split on first use: at full contrast it isn't needed
        x, y = D.grid_coords()
        r = np.hypot(x, y)[::8, ::8]
        self.inside = (r > 25) & (r < 57)  # percentiles from the part that can be burned
        self._cache = {}

    def __call__(self, contrast):
        if contrast >= 1:
            return self.tone
        key = round(float(contrast), 4)
        if key not in self._cache:
            self._cache = {key: self._at(max(contrast, 0.01))}  # place() asks for one contrast at a time
        return self._cache[key]

    def _split(self):
        from scipy.ndimage import gaussian_filter
        rest = gaussian_filter(self.tone, self.fine_mm / D.CELL_MM)
        self.layers, prev = [self.tone - rest], rest
        for r in self.radii_mm:
            low = guided(rest, max(1, round(r / D.CELL_MM)), self.eps).astype(np.float32)
            self.layers.append(prev - low)
            prev = low
        self.base = prev

    def _at(self, c):
        if self.layers is None:
            self._split()
        y = self.base.copy()
        for p, layer in zip(self.powers, self.layers):
            y += np.float32(c ** (p - 1)) * layer  # relative to the base; squeeze scales it all by c
        # Same spread as the picture, measured inside the clip percentiles; the tails roll off.
        pct = self.clip * (1 - c)
        a, b = np.percentile(y[::8, ::8][self.inside], [pct, 100 - pct])
        a0, b0 = np.percentile(self.tone[::8, ::8][self.inside], [pct, 100 - pct])
        z = a0 + (y - a) * ((b0 - a0) / max(b - a, 1e-6))
        return _rolloff(z, 0.08 * (1 - c)).astype(np.float32)


def _rolloff(x, knee):
    """Clip to 0..1, easing into the ends over `knee` instead of cutting flat."""
    if knee <= 0:
        return np.clip(x, 0, 1)
    lo, hi = knee, 1 - knee
    y = x.copy()
    up, dn = x > hi, x < lo
    y[up] = hi + knee * np.tanh((x[up] - hi) / knee)
    y[dn] = lo - knee * np.tanh((lo - x[dn]) / knee)
    return y


MODES = {"even": Even, "detail": Detail}
