"""A disc profile: everything that belongs to one drive + disc batch + burner, in one file.

Made once from disc-probe's measurements, then passed to every command with --disc. It holds
copies of the measured numbers, not paths, so it keeps working when the result folders move.
Edit `along_track_frames` by hand once a grid disc has shown the right value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from . import ecma130 as E
from .textures import texture_bytes
from .track import Geometry, correction_from, load_fit

KIND = "cdpicture-disc"


def build(geo_path: str, correction_path: str | None = None, along_track_frames: float = 0.0,
          cd_sector: str = "mode1", dark: str = "0D", light: str = "AA", data_tone: float = 0.5,
          allow_stopped: bool = False) -> dict:
    fit = load_fit(geo_path, allow_stopped)
    corr = None
    if correction_path:
        with open(correction_path, encoding="utf-8") as f:
            c = json.load(f)
        corr = {"radius_valid_mm": c["radius_valid_mm"], "poly_in_x": c["poly_in_x"]}
    prof = {"kind": KIND, "u": fit["u"], "v": fit["v"], "bit_nm": fit["bit_nm_assumed"], "correction": corr,
            "along_track_frames": along_track_frames, "cd_sector": cd_sector,
            "dark": dark, "light": light, "data_tone": data_tone,
            "measured_from": {"geo": geo_path, "correction": correction_path}}
    Profile.from_dict(prof)  # validate
    return prof


@dataclass
class Profile:
    u: float
    v: float
    bit_nm: float
    correction: dict | None
    along_track_frames: float
    cd_sector: str
    dark: str
    light: str
    data_tone: float

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        if d.get("kind") != KIND:
            raise ValueError("not a cdpicture disc profile (make one with: cdpicture profile)")
        p = cls(**{k: d[k] for k in cls.__dataclass_fields__})
        if p.cd_sector not in E.SECTOR_LAYOUTS:
            raise ValueError(f"cd_sector must be one of {sorted(E.SECTOR_LAYOUTS)}, not {p.cd_sector!r}")
        if not 0 < p.data_tone < 1:
            raise ValueError("data_tone must be between 0 (dark texture) and 1 (light texture)")
        for t in (p.dark, p.light):
            texture_bytes(t, np.array([0]), np.array([0]))
        return p

    @classmethod
    def load(cls, path: str) -> "Profile":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @property
    def offset(self) -> int:
        return E.SECTOR_LAYOUTS[self.cd_sector]

    def geometry(self, along_track_frames: float | None = None) -> Geometry:
        a = self.along_track_frames if along_track_frames is None else along_track_frames
        corr = correction_from(self.correction) if self.correction else None
        return Geometry.measured(self.u, self.v, self.bit_nm, corr, a)
