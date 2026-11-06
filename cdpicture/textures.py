"""Textures: the recorded (pre-scramble) bytes that draw the light or the dark parts of a design.

A texture is a hex byte sequence: "0D" one byte, "10AA63" three in turn. It is phased by
physical frame * 32 + F2 slot, including the parity slots in between, so it does not restart
at a sector boundary. No EFM pattern is implied by a byte sequence: merging bits, parity and
the drive's DSV decisions matter.
"""

from __future__ import annotations

import numpy as np


def texture_bytes(spec, frame, slot):
    """Bytes of one texture at these physical positions."""
    try:
        seq = np.frombuffer(bytes.fromhex(spec), np.uint8)
    except ValueError:
        seq = ()
    if len(seq) == 0:
        raise ValueError(f"texture {spec!r} must be hex bytes, e.g. 0D or 10AA63")
    return seq[(frame * 32 + slot) % len(seq)]
