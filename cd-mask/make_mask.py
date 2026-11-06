"""Painting template for a CD picture.

3200 x 3200 px covering 116 mm (the default --diameter-mm), data-side view.
The whole drawable ring, 25.6-56 mm, is free for the picture.

Writes, next to this script:
  cd-mask.png          white = drawable, black = not (for selections / alpha)
  cd-mask-overlay.png  transparent ring, translucent red outside it, black hole, blue edges
"""
import os

import numpy as np
from PIL import Image

N, DIAMETER_MM = 3200, 116.0
R_IN, R_OUT = 25.6, 56.0    # track starts at 25.28 mm (geo fit), measured span 25.58-56.6 mm
HOLE, EDGE = 7.5, 57.9      # centre hole and disc edge, for orientation only
LINE_MM = 0.15

out = os.path.dirname(os.path.abspath(__file__))
c = ((np.arange(N) + 0.5) / N - 0.5) * DIAMETER_MM
r = np.hypot(c[None, :], c[:, None])
ring = (r >= R_IN) & (r <= R_OUT)

Image.fromarray(np.where(ring, 255, 0).astype(np.uint8)).save(os.path.join(out, "cd-mask.png"))

rgba = np.zeros((N, N, 4), np.uint8)
rgba[~ring] = (200, 30, 30, 150)
rgba[(r < HOLE) | (r > EDGE)] = (0, 0, 0, 255)
for edge in (R_IN, R_OUT):
    rgba[np.abs(r - edge) < LINE_MM] = (0, 120, 255, 255)
Image.fromarray(rgba).save(os.path.join(out, "cd-mask-overlay.png"))

print(f"{N} px over {DIAMETER_MM} mm ({N / DIAMETER_MM:.2f} px/mm), drawable {R_IN}-{R_OUT} mm -> {out}")
