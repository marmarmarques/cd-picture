"""Hide file data inside a CD picture: which sectors take data, and how the picture is re-toned.

File data is recorded scrambled, so on the disc it reads as a flat grey tone t between the
dark (0) and light (1) textures, like any random bytes. A track is only 1.6 um wide, far below
what the eye resolves, so mixing a fraction f of data sectors into an area of tone T is not
seen as long as the picture sectors around them can make up the difference:

    f t + (1 - f) p = T,  0 <= p <= 1   ->   f <= T / t (T < t),  f <= (1 - T) / (1 - t) (T >= t)

A sector covers about two sector lengths of track (CIRC), so its limit is the smallest one
along that stretch. That is what keeps detail: a sector crossing an edge sees both tones and
gets a low limit. Because the limit is unimodal in T, the smallest value along the stretch is
reached at its darkest or lightest point, so each sector stores only those two tones.

Contrast c pulls the picture toward the data tone, T' = t + c (T - t). With outlines in the
picture almost every sector's stretch reaches both its darkest and lightest tone, so the
limit of nearly every sector is about 1 - c: contrast, not the picture, sets the room for
files. `place` keeps full contrast when the files fit, and otherwise lowers it just enough.

Data sectors are chosen per track turn against a low-discrepancy threshold (turn k uses
the van der Corput value of k), which spreads them evenly in radius and keeps each turn's
data in a few contiguous runs, so files split into few extents. `place` then raises one headroom h
(the share of each sector's limit used) until the chosen sectors just hold the files, so a
small folder is spread thinly and a large one uses more of the picture. Headroom does not
change the look; spreading the data thinly only makes a wrong data tone less visible. The picture is then
re-toned to p = (T - f t) / (1 - f) with the data fraction f actually realised; the writer
dithers it per turn with a different sequence (see track.picture_threshold).
"""

from __future__ import annotations

import math

import numpy as np

from . import design as I, ecma130 as cd
from .track import Geometry, data_threshold, turn
MAP_CELLS = 600                     # realised data-fraction map, 0.2 mm cells


def squeeze(tone, contrast, data_tone):
    """The picture at this contrast (1 = as is, 0 = flat data grey)."""
    return data_tone + contrast * (np.asarray(tone) - data_tone)


def sector_tones(tone, geo: Geometry, total_sectors):
    """Darkest and lightest tone along every sector's footprint."""
    lo, hi = np.empty(total_sectors, np.float32), np.empty(total_sectors, np.float32)
    for lbas, t in geo.sample_footprints(tone, total_sectors):
        lo[lbas], hi[lbas] = t.min(axis=1), t.max(axis=1)
    return lo, hi


def limit(tone, data_tone):
    """Largest data fraction an area of this tone can hide."""
    tone = np.asarray(tone, np.float64)
    return np.where(tone >= data_tone, (1 - tone) / (1 - data_tone), tone / data_tone).clip(0, 1)


def sector_limit(lo, hi, data_tone):
    return np.minimum(limit(lo, data_tone), limit(hi, data_tone))


def turn_threshold(u, v, total_sectors):
    """Per sector: the data threshold of the track turn where the sector starts."""
    return data_threshold(turn(np.arange(total_sectors) * float(cd.SECTOR_BITS), u, v))


def choose(f, threshold, headroom):
    return headroom * f > threshold


def runs(mask):
    """Contiguous runs of True: each one becomes at least one file extent."""
    m = np.r_[False, mask, False].astype(np.int8)
    return int(np.count_nonzero(np.diff(m) == 1))


def metadata_sectors(n_runs):
    """Rough directory size: two trees, about 50 bytes per extent record."""
    return 16 + math.ceil(2 * 50 * n_runs / 2048)


def realised_fraction(mask, geo: Geometry):
    """Fraction of track that carries file data, on a 0.2 mm grid (footprints, not sector
    starts, because CIRC spreads a sector over about two sector lengths)."""
    total = len(mask)
    data = np.zeros(MAP_CELLS * MAP_CELLS)
    hits = np.zeros(MAP_CELLS * MAP_CELLS)
    offsets = np.linspace(3 * cd.FRAME_BITS, 205 * cd.FRAME_BITS, 160)
    cell = 2 * I.HALF_MM / MAP_CELLS
    for a in range(0, total, 20000):
        lbas = np.arange(a, min(a + 20000, total))
        n = (lbas[:, None] * float(cd.SECTOR_BITS) + offsets[None, :]).ravel()
        r, th = geo.position(n)
        ix = np.clip(((r * np.cos(th) + I.HALF_MM) / cell).astype(np.int64), 0, MAP_CELLS - 1)
        iy = np.clip(((I.HALF_MM - r * np.sin(th)) / cell).astype(np.int64), 0, MAP_CELLS - 1)
        idx = iy * MAP_CELLS + ix
        hits += np.bincount(idx, minlength=len(hits))
        data += np.bincount(idx, weights=np.repeat(mask[lbas], len(offsets)).astype(float), minlength=len(hits))
    frac = np.divide(data, hits, out=np.zeros_like(data), where=hits > 0)
    return frac.reshape(MAP_CELLS, MAP_CELLS), hits.reshape(MAP_CELLS, MAP_CELLS) > 0


def compensated_design(tone_sharp, frac_small, data_tone):
    """Re-tone the picture for the data mixed into it, as a darkness design."""
    from PIL import Image
    f = np.asarray(Image.fromarray(frac_small.astype(np.float32)).resize((I.SIZE, I.SIZE), Image.BILINEAR))
    keep = 1 - f
    p = np.where(keep > 0.02, (tone_sharp - f * data_tone) / np.maximum(keep, 0.02), tone_sharp)
    return I.picture_design(p)


DATA_RGB = np.array([235, 165, 20])  # orange: where the file data goes


def _small(a, size, resample):
    from PIL import Image
    return np.asarray(Image.fromarray(np.asarray(a, np.float32)).resize((size, size), resample))


def on_disc(geo: Geometry, total_sectors, size):
    """Which pixels of a size x size data-side view the track covers (start to end radius)."""
    c = (np.arange(size) + 0.5) / size * 2 * I.HALF_MM - I.HALF_MM
    r = np.hypot(c[None, :], c[:, None])
    return (r >= geo.u * geo.bit_mm) & (r <= geo.end_radius(total_sectors))


def disc_view(design, frac_small, data_tone, size):
    """(tone the disc shows, data share): the picture's average grey, data as flat grey."""
    from PIL import Image
    f = _small(frac_small, size, Image.BILINEAR)
    return (1 - f) * _small(1 - I.darkness(design), size, Image.BOX) + f * data_tone, f


def _grey(a):
    return np.repeat((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)[..., None], 3, axis=2)


def _data_panel(tone_sharp, f, size):
    from PIL import Image
    under = _grey(0.35 + 0.65 * _small(tone_sharp, size, Image.BOX)).astype(np.float32)
    return (under * (1 - f[..., None]) + DATA_RGB * f[..., None]).astype(np.uint8)


def _save_on_disc(path, rgb, disc):
    """Opaque where the track is, transparent everywhere else (hole, beyond the edge)."""
    from PIL import Image
    alpha = np.where(disc, 255, 0).astype(np.uint8)[..., None]
    Image.fromarray(np.concatenate([np.where(disc[..., None], rgb, 0).astype(np.uint8), alpha], axis=2)).save(path)


def previews(paths, tone_sharp, design, frac_small, data_tone, geo: Geometry, total_sectors, title,
             layout=True, size=1200, sheet_size=900):
    """Write the data-side views: `preview` (what the disc shows), `layout` (where the data goes,
    orange = data share), and `sheet` (picture | preview | layout side by side). Only the part
    the track covers is drawn; the hole and everything past the end of the track are left out."""
    from PIL import Image, ImageDraw
    disc = on_disc(geo, total_sectors, size)
    sim, f = disc_view(design, frac_small, data_tone, size)
    _save_on_disc(paths["preview"], _grey(sim), disc)
    if layout:
        _save_on_disc(paths["layout"], _data_panel(tone_sharp, f, size), disc)
    n = sheet_size
    disc = on_disc(geo, total_sectors, n)
    sim, f = disc_view(design, frac_small, data_tone, n)
    white = lambda rgb: np.where(disc[..., None], rgb, 255).astype(np.uint8)
    panels = [_grey(_small(tone_sharp, n, Image.BOX)), white(_grey(sim)), white(_data_panel(tone_sharp, f, n))]
    out = Image.new("RGB", (3 * n + 40, n + 70), "white")
    d = ImageDraw.Draw(out)
    for i, (p, label) in enumerate(zip(panels, ("picture", "simulated disc (data = flat grey)",
                                                "where the data goes (orange = data share)"))):
        out.paste(Image.fromarray(p), (10 + i * (n + 10), 50))
        d.text((10 + i * (n + 10), 30), label, fill="black")
    d.text((10, 8), title, fill="black")
    out.save(paths["sheet"])


class Limits:
    """Per sector: the largest data share its stretch of picture can hide at the current
    contrast, and its turn threshold. The squeeze is affine, so it only rescales lo and hi;
    a tone that changes with contrast (`tone_at`, a tonemap mode) is measured again each time."""

    def __init__(self, tone_sharp, geo: Geometry, total_sectors, data_tone, contrast=1.0, tone_at=None):
        self.geo, self.total = geo, total_sectors
        self.lo, self.hi = self._measure(tone_sharp)
        self.data_tone = data_tone
        self.threshold = turn_threshold(geo.u, geo.v, total_sectors)
        varies = tone_at is not None and not getattr(tone_at, "fixed", False)
        self.tone_at, self._measured_at = (tone_at if varies else None), 1.0  # every mode is the picture at c = 1
        self.set_contrast(contrast)

    def _measure(self, tone):
        from scipy.ndimage import gaussian_filter
        seen = gaussian_filter(tone, 0.1 / I.CELL_MM)  # the eye never sees single dither dots
        return sector_tones(seen, self.geo, self.total)

    def set_contrast(self, contrast):
        t = self.data_tone
        self.contrast = contrast
        tone_at = getattr(self, "tone_at", None)
        if tone_at is not None and self._measured_at != contrast:
            self.lo, self.hi = self._measure(tone_at(contrast))
            self._measured_at = contrast
        self.f = sector_limit(squeeze(self.lo, contrast, t), squeeze(self.hi, contrast, t), t)

    def mask(self, headroom):
        """Data sectors at this headroom, plus the directory area from LBA 16."""
        m = choose(self.f, self.threshold, headroom)
        m[16:metadata_sectors(runs(m))] = True
        return m

    def capacity_mb(self, headroom=1.0):
        m = choose(self.f, self.threshold, headroom)
        return max(0, int(m.sum()) - metadata_sectors(runs(m))) * 2048 / 1e6


def place(limits: Limits, build, contrast=None, steps=14):
    """Highest contrast, then smallest headroom, whose data sectors hold the files:
    (file system, mask, headroom, contrast).

    contrast=None picks it: full contrast when the files fit there, else the highest whole
    percent that holds them at full headroom. build(allowed LBAs) returns an
    iso9660.FileSystem, or raises ValueError when the files don't fit; when they don't fit
    at all, that error is passed on.
    """
    def fits(pct):
        limits.set_contrast(pct / 100)
        try:
            return build(np.flatnonzero(limits.mask(1.0)))
        except ValueError:
            return None

    if contrast is None and fits(100) is not None:
        contrast = 1.0
    elif contrast is None:
        if fits(0) is None:
            raise ValueError("the files don't fit even with the picture flattened to grey")
        lo, hi = 0, 100  # lo fits, hi doesn't
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if fits(mid) is not None else (lo, mid)
        contrast = lo / 100
    limits.set_contrast(contrast)
    full = limits.mask(1.0)
    best = (build(np.flatnonzero(full)), full, 1.0)
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        mid = (lo + hi) / 2
        m = limits.mask(mid)
        try:
            best, hi = (build(np.flatnonzero(m)), m, mid), mid
        except ValueError:
            lo = mid
    return (*best, contrast)


def retone(tone_sharp, mask, geo: Geometry, data_tone):
    """(design, realised data fraction map) for data in exactly these sectors."""
    frac, _ = realised_fraction(mask, geo)
    return compensated_design(tone_sharp, frac, data_tone), frac
