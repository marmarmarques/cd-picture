"""Write a CD picture image, and check it with an independent inverse reconstruction.

The sampling unit is one 12-data-symbol half-frame (F2 slots 0..11 or 16..27): every
half-frame gets the texture of the design label at its place on the disc (a darkness design
is dithered per half-frame, see Geometry.labels), and CIRC maps those symbols back to user bytes. The file system (empty, or real files) is written over
the picture afterwards; with a picture/data mask the files take exactly their sectors.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from . import design as D, ecma130 as E, iso9660
from .textures import texture_bytes
from .track import Geometry

CHUNK = 2048


def _sector_q(design, texs, geo: Geometry, lba0, count, offset):
    lbas = np.arange(lba0, lba0 + count)
    frame, slot = E.user_to_physical(lbas[:, None], np.arange(2048)[None, :], offset)
    # Evaluate the geometry once per physical half-frame, then gather in user
    # order. CIRC scatters a sector, but many bytes still share one sample.
    first, last = int(frame.min()), int(frame.max())
    labels_by_frame = geo.labels(design, np.arange(first, last + 1)[:, None], np.array([0, 16])[None, :])
    labels = labels_by_frame[frame - first, (slot >= 16).astype(np.int8)]
    q = np.empty(labels.shape, np.uint8)
    for i, spec in enumerate(texs):
        m = labels == i
        q[m] = texture_bytes(spec, frame[m], slot[m])
    return q


def output_paths(path, image=True):
    """The image and its side files. Refuses to overwrite an image or its spec; the preview
    pictures are replaced, so a `plan` can be followed by `make` with the same name."""
    base = os.path.splitext(path)[0]
    paths = {"image": path, "spec": base + ".image.json", "preview": base + ".preview.png",
             "layout": base + ".layout.png", "sheet": base + ".sheet.png"}
    if image and any(os.path.exists(paths[k]) for k in ("image", "spec")):
        raise ValueError(f"output already exists: {path} or {paths['spec']}; choose a new filename")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return paths


def write_iso(path, design, geo: Geometry, total_sectors, texs, offset, files=None, meta=None, log=print,
              volume_id="CDPICTURE_IMAGE"):
    """files: an iso9660.FileSystem written on the inner zone instead of the empty one.
    When it was built with `allowed` sectors (a picture/data mask) the files replace the
    picture in exactly those sectors, so the design may draw anywhere."""
    E.check_capacity(total_sectors)
    if files is None:
        # Metadata sectors and their CIRC footprint must be in background label 0.
        f, s = E.user_to_physical(np.arange(16, 21)[:, None], np.arange(2048)[None, :], offset)
        if np.any(geo.sample(design, f, s) != 0):
            raise ValueError("ISO 9660 sectors overlap the drawing; leave an inner background margin")
    elif files.end_lba > total_sectors:  # end_lba is one past the last file sector, so == fills the image
        raise ValueError(f"files need {files.end_lba} sectors, the picture image only has {total_sectors}")
    elif files.allowed is None:
        r_files = geo.files_radius(files.end_lba)
        if D.drawn_inside(design, r_files):
            raise ValueError(f"the files reach {r_files:.2f} mm but the design draws inside that radius; "
                             f"start the picture further out or use fewer files")
    mask = E.mask(offset)
    done, next_report = 0, 0.0
    with open(path, "xb") as fh:
        while done < total_sectors:
            count = min(CHUNK, total_sectors - done)
            q = _sector_q(design, texs, geo, done, count, offset)
            fh.write((q ^ mask).tobytes())
            done += count
            if done / total_sectors >= next_report:
                log(f"  {100 * done / total_sectors:5.1f} %")
                next_report += 0.2
    if files is None:
        iso9660.write_empty(path, total_sectors, volume_id)
    else:
        files.write(path, total_sectors)
    spec = {"kind": "cd-picture", "media": "cd", "user_offset": offset, "u": geo.u, "v": geo.v, "bit_nm": geo.bit_nm,
            "total_sectors": total_sectors, "textures": texs, "cell_mm": D.CELL_MM,
            "sampling": "physical 12-data-symbol half-frame", "origin": "ECMA-130 fig C.4, sector byte 0 at F1 slot 1",
            "physical_origin_measured": False, **(meta or {})}
    if files is not None:
        spec["files"] = {"volume_id": files.volume_id, "end_lba": files.end_lba, "meta_end": files.meta_end,
                         "masked": files.allowed is not None,
                         "radius_mm": geo.files_radius(files.end_lba), "entries": files.entries()}
    spec.update(dark=texs[1], light=texs[0])
    if design.dtype.kind == "f":
        spec["dither"] = "per half-frame: dark when darkness > frac(track turn * golden ratio)"
    spec_path = os.path.splitext(path)[0] + ".image.json"
    with open(spec_path, "x", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    return spec_path


def fill_random(path, lbas, seed=5):
    """Overwrite these sectors with random bytes, which record like file data."""
    rng = np.random.default_rng(seed)
    with open(path, "r+b") as fh:
        for lba in np.asarray(lbas, np.int64):
            fh.seek(int(lba) * 2048)
            fh.write(rng.integers(0, 256, 2048, dtype=np.uint8).tobytes())


def data_sectors(region, geo: Geometry, total_sectors, threshold=0.5):
    """Per sector: True when at least `threshold` of its CIRC footprint lies in `region`.

    A sector's bytes reach frames LBA*98+3 .. LBA*98+205, about two sector lengths of
    track, so the picture/data border on the disc is blurred over roughly one sector
    whichever way a sector is assigned. The choice only changes the look, not the data.
    """
    out = np.empty(total_sectors, bool)
    for lbas, hit in geo.sample_footprints(region.astype(np.uint8), total_sectors):
        out[lbas] = hit.mean(axis=1) >= threshold
    return out


# --- independent check ----------------------------------------------------------------------
# Nothing below uses the writer's tables, sampling or scrambler mask: positions come from an
# explicit enumeration of figure C.4, the scrambler from a bit-serial LFSR, and the geometry
# from the closed form written out again.


def inverse_positions(lba0, count, offset):
    """Independent figure C.4 enumeration: physical output -> (lba, user byte, frame, slot).

    Enumerate the output frames, discard parity and sector overhead, then recover user indices.
    """
    slots = np.array(list(range(12)) + list(range(16, 28)))
    words = np.repeat([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], 2)
    # Explicit output delays from figure C.4, D=4 (A, B for each word).
    delays = np.array([3, 6, 11, 14, 19, 22, 27, 30, 35, 38, 43, 46,
                       65, 68, 73, 76, 81, 84, 89, 92, 97, 100, 105, 108])
    frames = np.arange(lba0 * 98, (lba0 + count) * 98 + 108)[:, None]
    raw = ((frames - delays) * 24 + 2 * words + np.tile([0, 1], 12)) ^ 1
    lbas, byte = raw // 2352, raw % 2352 - offset
    ok = (lbas >= lba0) & (lbas < lba0 + count) & (byte >= 0) & (byte < 2048)
    return lbas[ok], byte[ok], np.broadcast_to(frames, raw.shape)[ok], np.broadcast_to(slots, raw.shape)[ok]


def verify_iso(path, design, geo: Geometry, total_sectors, texs, offset, samples=300, seed=3, full=False,
               reconstruction=None, skip=None):
    """Inverse reconstruct physical samples, compare bytes and optionally splat a PNG.

    Checks sampled sectors (every sector with full=True), except those True in `skip` (file
    system and files; by default the five sectors of the empty file system). The raster
    shows only controlled symbols, not parity or optical appearance.
    """
    if os.path.getsize(path) != total_sectors * 2048:
        return ["size mismatch"]
    u, v, bit_nm, correction = geo.u, geo.v, geo.bit_nm, geo.correction
    if skip is None:
        skip = np.zeros(total_sectors, bool)
        skip[iso9660.EMPTY_SECTORS.start:iso9660.EMPTY_SECTORS.stop] = True
    if full:
        ranges = [(a, min(32, total_sectors - a)) for a in range(0, total_sectors, 32)]
    else:
        free = np.flatnonzero(~skip)
        picks = {int(free[0]), int(free[-1]), *np.random.default_rng(seed).choice(free, samples).tolist()}
        ranges = [(int(a), 1) for a in sorted(picks)]
    sums = np.zeros((1200, 1200), np.float64) if reconstruction else None
    hits = np.zeros((1200, 1200), np.int64) if reconstruction else None
    problems = []
    bits, reg = [], 1
    for _ in range(2340 * 8):
        bits.append(reg & 1)
        reg = (reg >> 1) ^ (0x4000 if (reg ^ (reg >> 1)) & 1 else 0)
    first = offset - 12  # the scrambler starts at sector byte 12
    mask = np.packbits(np.array(bits, np.uint8), bitorder="little")[first:first + 2048]
    with open(path, "rb") as fh:
        for start, count in ranges:
            fh.seek(start * 2048)
            data = np.frombuffer(fh.read(count * 2048), np.uint8).reshape(count, 2048)
            lba, byte, frame, slot = inverse_positions(start, count, offset)
            keep = ~skip[lba]
            lba, byte, frame, slot = (a[keep] for a in (lba, byte, frame, slot))
            rec = data[lba - start, byte] ^ mask[byte]
            # Reconstruct the group's location with the rationalized closed form.
            n = frame * 588 + np.where(slot < 12, 144.5, 416.5)
            s = np.sqrt(u ** 2 + v * n / np.pi)
            angle = -2 * n / (s + u)
            radius = bit_nm / 1e6 * s
            if correction is not None:
                angle += correction(radius) * np.pi / 180
            x, y = radius * np.cos(angle), radius * np.sin(angle)
            col = np.floor((x + 60) * 50).astype(int)
            row = np.floor((60 - y) * 50).astype(int)
            inside = (row >= 0) & (row < 6000) & (col >= 0) & (col < 6000)
            label = np.where(inside, design[np.clip(row, 0, 5999), np.clip(col, 0, 5999)], 0)
            if design.dtype.kind == "f":
                # Darkness, dithered per track turn: dark when above frac(turn * golden ratio).
                turn = np.floor(2 * n / (s + u) / (2 * np.pi))
                label = (label > np.mod(turn * ((np.sqrt(5) - 1) / 2), 1)).astype(np.int64)
            want = np.empty(rec.shape, np.uint8)
            for j, t in enumerate(texs):
                m = label == j
                seq = np.frombuffer(bytes.fromhex(t), np.uint8)
                want[m] = seq[(32 * frame[m] + slot[m]) % len(seq)]
            bad = rec != want
            if np.any(bad):
                problems.append(f"LBA {lba[bad][0]} byte {byte[bad][0]}: inverse physical reconstruction differs")
                if len(problems) >= 10:
                    break
            if sums is not None:
                # Actual bytes classified against the texture at this physical phase.
                val = np.zeros(rec.shape)
                for j, t in enumerate(texs):
                    val[rec == texture_bytes(t, frame, slot)] = j
                ix = np.clip(((x + 60) * 10).astype(int), 0, 1199)
                iy = np.clip(((60 - y) * 10).astype(int), 0, 1199)
                np.add.at(sums, (iy, ix), val)
                np.add.at(hits, (iy, ix), 1)
    if sums is not None:
        from PIL import Image
        raster = np.divide(sums, hits, out=np.zeros_like(sums), where=hits > 0)
        scale = 255 / max(1, len(texs) - 1)
        Image.fromarray(np.where(hits > 0, 255 - np.clip(raster * scale, 0, 255), 255).astype(np.uint8)).save(reconstruction)
    return problems
