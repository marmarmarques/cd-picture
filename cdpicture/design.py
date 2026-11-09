"""Designs and regions on a fixed grid over the disc, in data-side view (x right, y up).

A design is SIZE x SIZE cells of CELL_MM: boolean (True = dark) or float darkness (0 = light,
1 = dark; the writer dithers it per track turn). In both, 0 is the light background.
A region is boolean (True = sectors there may hold file data).
"""

from __future__ import annotations

import math

import numpy as np

CELL_MM = 0.02
HALF_MM = 60.0
SIZE = int(2 * HALF_MM / CELL_MM)


def grid_coords():
    c = (np.arange(SIZE) + 0.5) * CELL_MM - HALF_MM
    return c[None, :], -c[:, None]  # x right, y up


def in_circle(r_mm: float) -> np.ndarray:
    x, y = grid_coords()
    return x ** 2 + y ** 2 <= r_mm ** 2


def grid_design(r_min=26.0, r_max=56.0, spacing=5.0, line=0.6) -> np.ndarray:
    """Calibration grid, True = dark: Cartesian lines, radial lines every 30 degrees,
    circles every `spacing` mm, and a block letter F (asymmetric, so a mirrored burn is obvious)."""
    x, y = grid_coords()
    r = np.hypot(x, y)
    th = np.degrees(np.arctan2(y, x))
    half = line / 2
    near = lambda v: np.abs((v + spacing / 2) % spacing - spacing / 2) < half
    dark = near(x) | near(y)
    dark |= np.abs(r - np.round(r / spacing) * spacing) < half
    for a in range(0, 360, 30):
        d = np.abs(((th - a + 180) % 360) - 180)
        dark |= (np.radians(d) * r < max(0.2, half)) & (d < 90)
    # Letter F, 10 mm tall, in the upper half (stroke 1.2 mm).
    fx, fy, s = -3.0, 33.0, 1.2
    dark |= (x >= fx) & (x < fx + s) & (y >= fy) & (y < fy + 10)
    dark |= (x >= fx) & (x < fx + 6) & (y >= fy + 10 - s) & (y < fy + 10)
    dark |= (x >= fx) & (x < fx + 4.5) & (y >= fy + 5 - s / 2) & (y < fy + 5 + s / 2)
    return dark & (r >= r_min) & (r <= r_max)


def _load(path: str, diameter_mm: float):
    """(CIE L* 0..1, alpha 0..1) of a picture, fitted inside a square of this diameter without
    changing its shape; alpha is 0 outside the picture."""
    from PIL import Image
    with Image.open(path) as im:
        rgba = np.asarray(im.convert("RGBA"), np.float32) / 255
    rgb = rgba[..., :3]
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lstar = np.where(y > 216 / 24389, 116 * np.cbrt(y) - 16, y * 24389 / 27) / 100
    side = round(diameter_mm / CELL_MM)
    h, w = y.shape
    size = (max(1, round(w * side / max(w, h))), max(1, round(h * side / max(w, h))))
    fit = lambda a: np.clip(np.asarray(Image.fromarray(a.astype(np.float32)).resize(size, Image.LANCZOS)), 0, 1)
    out = np.zeros((2, SIZE, SIZE), np.float32)
    oy, ox = (SIZE - size[1]) // 2, (SIZE - size[0]) // 2
    out[0, oy:oy + size[1], ox:ox + size[0]] = fit(lstar)
    out[1, oy:oy + size[1], ox:ox + size[0]] = fit(rgba[..., 3])
    return out[0], out[1]


def load_tone(path: str, diameter_mm: float = 116.0) -> np.ndarray:
    """A picture as tone 0 (dark) .. 1 (light), fitted to a circle of this diameter.

    Tone is perceptual lightness (CIE L*), stretched so the picture's darkest 1 % and lightest
    0.5 % reach 0 and 1: every picture starts at full contrast, and --contrast takes it from
    there. Transparent parts and the area outside the picture are light.
    """
    lstar, alpha = _load(path, diameter_mm)
    inside = (alpha > 0.5)[::4, ::4]
    if inside.any():
        lo, hi = np.percentile(lstar[::4, ::4][inside], [1, 99.5])
        if hi > lo:
            lstar = np.clip((lstar - lo) / (hi - lo), 0, 1)
    return (alpha * lstar + (1 - alpha)).astype(np.float32)


def picture_design(tone: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Darkness 1 - tone, which the writer dithers along the track; or dark = True below
    `threshold` (for line art and logos).

    Not dithered here on the grid: the writer can change texture only once per half-frame,
    about 0.08 mm of track, so grid dots of 0.02 mm would mostly be skipped.
    """
    return (1 - np.clip(tone, 0, 1)).astype(np.float32) if threshold is None else tone < threshold


def darkness(design: np.ndarray) -> np.ndarray:
    """Average darkness 0..1 of a design, cell by cell."""
    return design.astype(np.float32)


def mask_region(path: str, diameter_mm: float = 116.0) -> np.ndarray:
    """Where files may go: white areas of a PNG, fitted like the picture."""
    lstar, alpha = _load(path, diameter_mm)
    return (lstar >= 0.5) & (alpha > 0.5) & in_circle(diameter_mm / 2)


def _inner(r_mm: float):
    """Index box around the centre, and which of its cells lie within r_mm."""
    k = min(SIZE // 2, int(math.ceil(r_mm / CELL_MM)) + 1)
    box = slice(SIZE // 2 - k, SIZE // 2 + k)
    c = ((np.arange(SIZE // 2 - k, SIZE // 2 + k) + 0.5) * CELL_MM - HALF_MM).astype(np.float32)
    return box, c[None, :] ** 2 + c[:, None] ** 2 <= np.float32(r_mm) ** 2


def drawn_inside(design: np.ndarray, r_mm: float) -> bool:
    """True when the design draws anything within r_mm of the centre."""
    box, inside = _inner(r_mm)
    return bool(np.any(design[box, box][inside] != 0))


def clear_inside(design: np.ndarray, r_mm: float) -> None:
    """Set everything within r_mm of the centre to background, in place."""
    box, inside = _inner(r_mm)
    design[box, box][inside] = 0


def clear_outside(design: np.ndarray, r_mm: float) -> None:
    """Set everything at r_mm or further from the centre to background, in place."""
    design[~in_circle(r_mm)] = 0
