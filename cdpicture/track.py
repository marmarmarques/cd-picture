"""Where a channel bit lands on the disc, from a measured spiral geometry.

The model is the one disc-probe's `geo` ladder fits (u = r0 / b, v = pitch / b):

    Theta(n) = 2n / (S + u),  S = sqrt(u^2 + v n / pi),  radius = b * S

u and v fix every angle; the assumed bit length b only scales the picture. The disc turns
counter-clockwise as seen by the optical head, so in the data-side view (y up) the track
runs clockwise. An optional correction phi(r), measured on a burned grid, is added to the
angle: a byte placed at model angle theta shows up at theta + phi(r).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import design as D, ecma130 as E


def theta(n, u, v):
    """Accumulated track angle in radians at channel bit n."""
    n = np.asarray(n, dtype=np.float64)
    s = np.sqrt(u * u + v * n / math.pi)
    # 2 pi (s - u) / v, with s - u written as (s^2 - u^2) / (s + u) to keep precision.
    return 2.0 * n / (s + u)


def turn(n, u, v):
    """Index of the track turn that holds channel bit n."""
    return np.floor(theta(n, u, v) / (2 * math.pi))


# Two per-turn threshold sequences. Each spreads its values evenly over the ~60 turns that
# cross 0.1 mm, so the share of turns below a level matches that level closely. They must
# be different sequences: the picture dither runs only on turns not taken by data, and with
# the same sequence for both, those turns would all have high thresholds (6-7 % darker).
GOLDEN = (math.sqrt(5) - 1) / 2


def picture_threshold(k):
    """Dither threshold of turn k: frac(k * golden ratio)."""
    return np.mod(np.asarray(k, np.float64) * GOLDEN, 1.0)


def data_threshold(k):
    """Data-sector threshold of turn k: van der Corput (k with its binary digits mirrored)."""
    k = np.asarray(k, np.int64).copy()
    out, weight = np.zeros(k.shape), 0.5
    while np.any(k):
        out += weight * (k & 1)
        k >>= 1
        weight /= 2
    return out


def load_fit(geo_path: str, allow_stopped: bool = False) -> dict:
    """The `fit` of a disc-probe geo summary.json, refused unless it is a whole run on a CD data track.

    Since disc-probe 7d670ff, geo also measures a DiscImageStudio burn (one CD-DA track) and marks it
    track=audio. That fit counts from the end of each sector in the audio stream, not from the CIRC
    read end this package uses (108 frames later), so a picture drawn from it lands in the wrong place.
    Since a689f5a, a run that ended early (tray opened, Ctrl+C) carries `stopped`: its fit can look fine
    but only covers the radii it reached. Older results have neither field and can't be told apart.
    """
    with open(geo_path, encoding="utf-8") as f:
        summary = json.load(f)
    fit = summary["fit"]
    if fit.get("media") != "cd":
        raise ValueError("need a CD geometry fit (media=cd); DVD fits must not be reused")
    if fit.get("track") == "audio":
        raise ValueError("this geo run measured a CD-DA track (a DiscImageStudio picture burn). Its sectors count "
                         "from a different origin, so the picture would land in the wrong place. Run geo on the "
                         "random-data disc instead (python -m discprobe random --media cd)")
    if summary.get("stopped") and not allow_stopped:
        raise ValueError(f"this geo run stopped early ({summary['stopped']}), so it only covers the radii it "
                         "reached. Run geo again, or pass --allow-stopped to use it anyway")
    return fit


def load_correction(path: str) -> Callable:
    """phi(r) in degrees from an angle-correction file (polynomial, held constant outside its range)."""
    with open(path, encoding="utf-8") as f:
        return correction_from(json.load(f))


def correction_from(c: dict) -> Callable:
    lo, hi = c["radius_valid_mm"]
    coef = np.array(c["poly_in_x"])
    return lambda r: np.polyval(coef, (np.clip(r, lo, hi) - 40.0) / 15.0)


@dataclass
class Geometry:
    u: float
    v: float
    bit_nm: float
    correction: Callable | None = None

    @classmethod
    def nominal(cls, correction=None) -> "Geometry":
        """Red Book nominal spiral (25 mm start, 1.6 um pitch), for tests."""
        return cls(E.R0_MM / (E.BIT_NM * 1e-6), E.PITCH_UM * 1e3 / E.BIT_NM, E.BIT_NM, correction)

    @classmethod
    def from_files(cls, geo_path: str, correction_path: str | None = None,
                   along_track_frames: float = 0.0) -> "Geometry":
        """Read a disc-probe CD geo summary.json, and optionally a correction.json.

        Timing cannot see an along-track offset d (read completion vs written position):
        theta(n - d) is the same spiral with u^2 - v d / pi. Measure d on a burned grid disc.
        """
        fit = load_fit(geo_path)
        return cls.measured(fit["u"], fit["v"], fit["bit_nm_assumed"],
                            load_correction(correction_path) if correction_path else None, along_track_frames)

    @classmethod
    def measured(cls, u, v, bit_nm, correction=None, along_track_frames=0.0) -> "Geometry":
        if along_track_frames:
            u = math.sqrt(u * u - v * along_track_frames * E.FRAME_BITS / math.pi)
        return cls(u, v, bit_nm, correction)

    @property
    def bit_mm(self) -> float:
        return self.bit_nm * 1e-6

    def position(self, n):
        """(radius mm, angle rad) of channel bit n in the data-side view, correction included."""
        r = self.bit_mm * np.sqrt(self.u * self.u + self.v * n / math.pi)
        th = -theta(n, self.u, self.v)
        if self.correction is not None:
            th += np.radians(self.correction(r))
        return r, th

    def sectors_to_radius(self, radius: float) -> int:
        """Number of whole sectors that fit inside this radius."""
        n = math.pi * ((radius / self.bit_mm) ** 2 - self.u * self.u) / self.v
        return int(n // E.SECTOR_BITS)

    def end_radius(self, total_sectors: int) -> float:
        return self.bit_mm * math.sqrt(self.u * self.u + self.v * total_sectors * E.SECTOR_BITS / math.pi)

    def files_radius(self, end_lba: int) -> float:
        """Outer radius (mm) of the CIRC footprint of sectors below end_lba (the last one
        reaches 108 frames past its own end)."""
        return self.bit_mm * math.sqrt(self.u * self.u + self.v * float(E.read_end_bits(end_lba - 1)) / math.pi)

    @staticmethod
    def half_frame_bit(frame, slot):
        """Channel bit at the middle of the 12 data symbols of a physical half-frame.

        Data halves (F2 slots 0..11, 16..27) are contiguous on the track, unlike user byte
        groups, so one sample at the middle of a half-frame stands for all 12 symbols.
        """
        return frame * 588 + 51 + np.where(slot < 12, 5.5, 21.5) * 17

    def _lookup(self, grid, n):
        r, th = self.position(n)
        ix = np.floor((r * np.cos(th) + D.HALF_MM) / D.CELL_MM).astype(np.int64)
        iy = np.floor((D.HALF_MM - r * np.sin(th)) / D.CELL_MM).astype(np.int64)
        ok = (ix >= 0) & (ix < D.SIZE) & (iy >= 0) & (iy < D.SIZE)
        return np.where(ok, grid[np.clip(iy, 0, D.SIZE - 1), np.clip(ix, 0, D.SIZE - 1)], 0)

    def sample(self, grid: np.ndarray, frame, slot) -> np.ndarray:
        """Value of a design-grid array at each physical half-frame (0 outside the grid)."""
        return self._lookup(grid, self.half_frame_bit(frame, slot))

    def labels(self, design: np.ndarray, frame, slot) -> np.ndarray:
        """Texture label of each physical half-frame. A darkness design (float, 0 = light,
        1 = dark) is dithered here, on the track itself: a half-frame is dark (1) when its
        darkness is above its turn's picture threshold."""
        n = self.half_frame_bit(frame, slot)
        value = self._lookup(design, n)
        if design.dtype.kind != "f":
            return value
        return (value > picture_threshold(turn(n, self.u, self.v))).astype(np.uint8)

    def sample_footprints(self, grid: np.ndarray, total_sectors: int, chunk: int = 20000):
        """Yield (lbas, values) with the grid sampled along every sector's CIRC footprint
        (about two sector lengths of track): values has shape (len(lbas), 52)."""
        for a in range(0, total_sectors, chunk):
            lbas = np.arange(a, min(a + chunk, total_sectors))
            frame = (lbas[:, None] * E.FRAMES_PER_SECTOR + E.FOOTPRINT_FRAMES[None, :])[:, :, None]
            yield lbas, self.sample(grid, frame, np.array([0, 16])[None, None, :]).reshape(len(lbas), -1)

    def raster_index(self, n, size: int):
        """(ix, iy) of channel bits n on a size x size raster of the 120 mm square, clipped."""
        r, th = self.position(n)
        ix = np.clip(((r * np.cos(th) + D.HALF_MM) / (2 * D.HALF_MM) * size).astype(np.int64), 0, size - 1)
        iy = np.clip(((D.HALF_MM - r * np.sin(th)) / (2 * D.HALF_MM) * size).astype(np.int64), 0, size - 1)
        return ix, iy
