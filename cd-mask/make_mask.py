"""Painting template for a CD picture.

3200 x 3200 px covering 116 mm (the default --diameter-mm), data-side view.
By default the whole drawable ring, 25.6-56 mm, is free for the picture.

Writes, next to this script:
  cd-mask{-NAME}.png          white = drawable, black = not (for selections / alpha)
  cd-mask{-NAME}-overlay.png  transparent ring, translucent red outside it, black hole, blue edges

With --measured A B, the part of the ring outside A-B mm (where the angle correction is
extrapolated rather than measured) is tinted amber in the overlay and dashed-edged, and a
strict mask covering only A-B is written too (cd-mask{-NAME}-strict.png).

    python cd-mask/make_mask.py                       # the default template
    python cd-mask/make_mask.py --name sony --r-in 25.3 --r-out 56 --measured 26.41 54.28
"""
import argparse
import os

import numpy as np
from PIL import Image

N, DIAMETER_MM = 3200, 116.0
R_IN, R_OUT = 25.6, 56.0    # track starts at 25.28 mm (geo fit), measured span 25.58-56.6 mm
HOLE, EDGE = 7.5, 57.9      # centre hole and disc edge, for orientation only
LINE_MM = 0.15

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--name", help="suffix for the output files, e.g. sony -> cd-mask-sony.png")
ap.add_argument("--r-in", type=float, default=R_IN, help=f"inner edge of the drawable ring, mm (default {R_IN})")
ap.add_argument("--r-out", type=float, default=R_OUT, help=f"outer edge, mm (default {R_OUT}; match --r-max)")
ap.add_argument("--measured", type=float, nargs=2, metavar=("A", "B"),
                help="radii the angle correction was measured over; outside them it is extrapolated")
a = ap.parse_args()

out = os.path.dirname(os.path.abspath(__file__))
stem = "cd-mask" + (f"-{a.name}" if a.name else "")
c = ((np.arange(N) + 0.5) / N - 0.5) * DIAMETER_MM
r = np.hypot(c[None, :], c[:, None])
ring = (r >= a.r_in) & (r <= a.r_out)

Image.fromarray(np.where(ring, 255, 0).astype(np.uint8)).save(os.path.join(out, f"{stem}.png"))

rgba = np.zeros((N, N, 4), np.uint8)
rgba[~ring] = (200, 30, 30, 150)
if a.measured:
    lo, hi = a.measured
    rgba[ring & ((r < lo) | (r > hi))] = (240, 170, 0, 90)
    strict = (r >= max(lo, a.r_in)) & (r <= min(hi, a.r_out))
    Image.fromarray(np.where(strict, 255, 0).astype(np.uint8)).save(os.path.join(out, f"{stem}-strict.png"))
rgba[(r < HOLE) | (r > EDGE)] = (0, 0, 0, 255)
for edge in (a.r_in, a.r_out):
    rgba[np.abs(r - edge) < LINE_MM] = (0, 120, 255, 255)
if a.measured:
    ang = np.degrees(np.arctan2(c[:, None], c[None, :]))
    dash = (ang % 6) < 3
    for edge in a.measured:
        if a.r_in < edge < a.r_out:
            rgba[(np.abs(r - edge) < LINE_MM) & dash] = (240, 170, 0, 255)
Image.fromarray(rgba).save(os.path.join(out, f"{stem}-overlay.png"))

print(f"{N} px over {DIAMETER_MM} mm ({N / DIAMETER_MM:.2f} px/mm), drawable {a.r_in}-{a.r_out} mm"
      + (f", measured {a.measured[0]}-{a.measured[1]} mm" if a.measured else "") + f" -> {out}/{stem}*.png")
