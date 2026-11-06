"""Picture images, with and without files, checked byte for byte and by an independent ISO walk."""

import os

import numpy as np
import pytest

from cdpicture import design as D, ecma130 as E, hide, iso9660, tonemap, writer
from cdpicture.track import Geometry, data_threshold, picture_threshold

M1 = E.SECTOR_LAYOUTS["mode1"]
TEX = ["AA", "10AA63"]  # a dark sequence, to check its phase follows the physical position
quiet = lambda *_: None


def read_tree(path, lba, joliet):
    """Independent ISO 9660 walk from a volume descriptor: {relative path: bytes}."""
    with open(path, "rb") as f:
        def sector(n, count=1):
            f.seek(n * 2048)
            return f.read(count * 2048)

        vd = sector(lba)
        if vd[1:6] != b"CD001" or vd[0] != (2 if joliet else 1) or (joliet and vd[88:91] != b"%/E"):
            return None
        out, pending = {}, {}

        def walk(rec, prefix):
            ext, size = int.from_bytes(rec[2:6], "little"), int.from_bytes(rec[10:14], "little")
            data, pos = sector(ext, -(-size // 2048)), 0
            while pos < size:
                n = data[pos]
                if n == 0:  # records do not cross sectors; skip to the next one
                    pos = (pos // 2048 + 1) * 2048
                    continue
                r = data[pos:pos + n]
                ident = r[33:33 + r[32]]
                pos += n
                if ident in (b"\x00", b"\x01"):
                    continue
                name = ident.decode("utf-16-be") if joliet else ident.decode("ascii")
                if r[25] & 2:
                    walk(r, prefix + name + "/")
                else:  # a multi-extent file repeats its record, flag 0x80 on all but the last
                    fe, fs = int.from_bytes(r[2:6], "little"), int.from_bytes(r[10:14], "little")
                    part = sector(fe, -(-fs // 2048))[:fs] if fs else b""
                    out[prefix + name] = (out.pop(prefix + name, b"") if pending.get(prefix + name) else b"") + part
                    pending[prefix + name] = bool(r[25] & 0x80)
        walk(vd[156:190], "")
        return out


def _flip(path, pos):
    with open(path, "r+b") as f:
        f.seek(pos)
        b = f.read(1)
        f.seek(-1, 1)
        f.write(bytes([b[0] ^ 1]))


def _source(tmp, want, empty_dirs=()):
    src = os.path.join(tmp, "data")
    for d in empty_dirs:
        os.makedirs(os.path.join(src, d))
    for rel, data in want.items():
        os.makedirs(os.path.dirname(os.path.join(src, rel)), exist_ok=True)
        with open(os.path.join(src, rel), "wb") as f:
            f.write(data)
    return src


@pytest.mark.parametrize("layout", ["mode1", "mode2form1"])
def test_image_round_trip(tmp_path, layout):
    """A two-tone design with boundaries crossing the CIRC footprint, and a correction."""
    offset = E.SECTOR_LAYOUTS[layout]
    design = np.zeros((D.SIZE, D.SIZE), bool)
    design[:3000, :] = True
    design[:, 3000:] ^= True
    x, y = D.grid_coords()
    design[np.hypot(x, y) < 25.03] = False  # leave the ISO metadata footprint outside the drawing
    geo = Geometry.nominal(correction=lambda r: 0.15 * (r - 25))
    tex = TEX
    path = str(tmp_path / "image.iso")
    writer.write_iso(path, design, geo, 512, tex, offset, log=quiet)
    assert not writer.verify_iso(path, design, geo, 512, tex, offset, full=True)
    _flip(path, 123 * 2048 + 42)
    assert writer.verify_iso(path, design, geo, 512, tex, offset, full=True)
    with pytest.raises(ValueError):
        writer.output_paths(path)


def test_darkness_dithered_on_the_track(tmp_path):
    """A grey design is dithered per half-frame; the share of dark ones follows the grey,
    and the independent check reproduces every one of them."""
    x, y = D.grid_coords()
    grey = ((np.arctan2(y, x) + np.pi) / (2 * np.pi)).astype(np.float32)
    design = np.where(np.hypot(x, y) < 25.03, 0, grey).astype(np.float32)
    geo = Geometry.nominal(correction=lambda r: 0.15 * (r - 25))
    path = str(tmp_path / "grey.iso")
    writer.write_iso(path, design, geo, 512, ["AA", "0D"], M1, log=quiet)
    assert not writer.verify_iso(path, design, geo, 512, ["AA", "0D"], M1, full=True)
    _flip(path, 300 * 2048 + 7)
    assert writer.verify_iso(path, design, geo, 512, ["AA", "0D"], M1, full=True)
    frame, slot = np.arange(50 * 98, 500 * 98)[:, None], np.array([0, 16])[None, :]
    assert abs(geo.labels(design, frame, slot).mean() - geo.sample(design, frame, slot).mean()) < 0.01


def test_dither_does_not_follow_data_choice():
    """The dither runs only on turns the data left, so its sequence must not repeat the data one."""
    k = np.arange(200000)
    left = data_threshold(k) >= 0.4  # turns that keep the picture when 40 % of them hold data
    assert abs((picture_threshold(k[left]) < 0.3).mean() - 0.3) < 0.005
    assert abs((data_threshold(k[left]) < 0.3).mean() - 0.3) > 0.1  # the same sequence would not do


def test_wrong_layout_is_detected(tmp_path):
    design = np.zeros((D.SIZE, D.SIZE), bool)
    geo = Geometry.nominal()
    path = str(tmp_path / "m1.iso")
    writer.write_iso(path, design, geo, 64, ["AA", "0D"], M1, log=quiet)
    assert writer.verify_iso(path, design, geo, 64, ["AA", "0D"], E.SECTOR_LAYOUTS["mode2form1"], full=True)


def test_design_over_empty_file_system_refused(tmp_path):
    design = np.ones((D.SIZE, D.SIZE), bool)
    with pytest.raises(ValueError):
        writer.write_iso(str(tmp_path / "x.iso"), design, Geometry.nominal(), 64, ["AA", "0D"], M1, log=quiet)


def test_files_on_inner_zone(tmp_path):
    rng = np.random.default_rng(29)
    want = {"readme.txt": b"hello\n", "empty.txt": b"", "Long name one.txt": b"1", "Long name two.txt": b"2",
            "Fotos ñ 日本/big.bin": rng.integers(0, 256, 5000, dtype=np.uint8).tobytes(),
            "Fotos ñ 日本/sub/deep.md": b"deep"}
    fs = iso9660.FileSystem(_source(str(tmp_path), want, ["empty dir"]), "Test Disc")
    geo = Geometry.nominal()
    r_files = geo.files_radius(fs.end_lba)
    x, y = D.grid_coords()
    r = np.hypot(x, y)
    design = (r > r_files + 0.02) & ((r < 25.5) | (x > 0))
    path = str(tmp_path / "files.iso")
    writer.write_iso(path, design, geo, 600, TEX, M1, files=fs, log=quiet)
    assert not writer.verify_iso(path, design, geo, 600, TEX, M1, full=True, skip=fs.data_sectors(600))
    assert not fs.verify(path)
    assert read_tree(path, 17, True) == want
    pri = read_tree(path, 16, False)
    names = [k.rsplit("/", 1)[-1] for k in pri]
    assert sorted(pri.values()) == sorted(want.values()) and len(set(pri)) == len(want)
    assert all(len(n.split(".")[0]) <= 8 and n.endswith(";1") for n in names)
    with pytest.raises(ValueError):  # a design drawing over the files
        writer.write_iso(str(tmp_path / "bad.iso"), r > 24, geo, 600, TEX, M1, files=fs, log=quiet)


def test_files_woven_into_spiral_mask(tmp_path):
    rng = np.random.default_rng(31)
    want = {"a.bin": rng.integers(0, 256, 9000, dtype=np.uint8).tobytes(), "b.txt": b"short",
            "dir/c.bin": rng.integers(0, 256, 30000, dtype=np.uint8).tobytes()}
    src = _source(str(tmp_path), want)
    geo = Geometry.nominal()
    x, y = D.grid_coords()
    r = np.hypot(x, y)
    design = (r > 25.02) & ((r < 25.5) | (x > 0))
    th = np.arctan2(y, x)
    spiral = np.abs((th - 2 * np.pi * 4 * (r - 25) / 33 + np.pi) % (2 * np.pi) - np.pi) > np.radians(60)
    sec = writer.data_sectors(spiral, geo, 600)
    assert 0.2 < sec.mean() < 0.8 and np.any(np.diff(sec.astype(int)) != 0)
    fs = iso9660.FileSystem(src, "Masked", allowed=np.flatnonzero(sec))
    used = fs.data_sectors(600)
    assert not np.any(used[fs.meta_end:] & ~sec[fs.meta_end:])
    assert max(e["extents"] for e in fs.entries()) > 1
    path = str(tmp_path / "masked.iso")
    writer.write_iso(path, design, geo, 600, TEX, M1, files=fs, log=quiet)
    assert not writer.verify_iso(path, design, geo, 600, TEX, M1, full=True, skip=used)
    assert not fs.verify(path)
    assert read_tree(path, 17, True) == want
    assert sorted(read_tree(path, 16, False).values()) == sorted(want.values())
    with pytest.raises(ValueError):  # files larger than the data region
        iso9660.FileSystem(src, "Masked", allowed=np.flatnonzero(sec)[:5])


def test_hiding_math():
    t = 0.5
    assert np.allclose(hide.limit(np.array([0.0, 0.25, 0.5, 0.75, 1.0]), t), [0, 0.5, 1, 0.5, 0])
    assert np.isclose(hide.sector_limit(np.array([0.3]), np.array([0.6]), t)[0], 0.6)
    geo = Geometry.nominal()
    thr = hide.turn_threshold(geo.u, geo.v, 40000)
    m = hide.choose(np.full(40000, 0.6), thr, 0.5)
    assert abs(m.mean() - 0.3) < 0.01
    assert hide.runs(m) < 0.6 * m.sum()
    tone = np.full((D.SIZE, D.SIZE), 0.7, np.float32)
    frac = np.full((hide.MAP_CELLS,) * 2, 0.4)
    design = hide.compensated_design(tone, frac, t)
    assert abs(0.4 * t + 0.6 * (1 - design).mean() - 0.7) < 0.01


def _limits(n, tone):
    """Limits for n sectors whose stretch of picture is all one tone (data tone 0.5)."""
    geo = Geometry.nominal()
    lim = hide.Limits.__new__(hide.Limits)
    lim.lo = lim.hi = np.full(n, tone)
    lim.data_tone, lim.threshold = 0.5, hide.turn_threshold(geo.u, geo.v, n)
    lim.set_contrast(1.0)
    return lim


def test_place_uses_smallest_headroom_that_fits(tmp_path):
    """Auto placement keeps full contrast and grows the data area only until the files fit."""
    lim = _limits(4000, 0.4)  # limit 0.8
    src = _source(str(tmp_path), {"a.bin": bytes(300 * 2048)})
    fs, mask, h, c = hide.place(lim, lambda allowed: iso9660.FileSystem(src, "T", allowed=allowed))
    assert 0 < h < 1 and c == 1.0
    assert np.all(fs.data_sectors(4000) <= mask | (np.arange(4000) < fs.meta_end))
    smaller = lim.mask(h * 0.9)
    with pytest.raises(ValueError):
        iso9660.FileSystem(src, "T", allowed=np.flatnonzero(smaller))


def test_place_lowers_contrast_only_as_far_as_needed(tmp_path):
    lim = _limits(4000, 0.05)  # nearly black: limit 0.1 at full contrast
    build = lambda allowed: iso9660.FileSystem(_source(str(tmp_path), {"a.bin": bytes(1500 * 2048)}),
                                               "T", allowed=allowed)
    fs, mask, h, c = hide.place(lim, build)
    assert 0 < c < 1 and lim.contrast == c
    lim.set_contrast(c + 0.01)
    with pytest.raises(ValueError):
        build(np.flatnonzero(lim.mask(1.0)))
    with pytest.raises(ValueError):  # a fixed contrast is kept, not lowered
        hide.place(lim, build, contrast=1.0)


def test_contrast_pulls_toward_the_data_tone():
    assert np.allclose(hide.squeeze(np.array([0.0, 0.5, 1.0]), 0.7, 0.5), [0.15, 0.5, 0.85])
    assert np.allclose(hide.squeeze(np.array([0.0, 1.0]), 0.0, 0.4), [0.4, 0.4])


def test_detail_tone_keeps_edges_and_gives_up_shading():
    """At full contrast the picture is untouched; lower, faint fine stripes (texture, not strong
    edges) keep more of their amplitude than an even squeeze leaves them, the broad gradient less."""
    x, _ = D.grid_coords()
    x = np.broadcast_to(x, (D.SIZE, D.SIZE))
    shading = 0.2 + 0.6 * (x + D.HALF_MM) / (2 * D.HALF_MM)
    stripes = 0.03 * np.sign(np.sin(2 * np.pi * x / 0.6))  # 0.3 mm stripes
    tone = np.clip(shading + stripes, 0, 1).astype(np.float32)
    detail = tonemap.Detail(tone)
    assert detail(1.0) is tone
    c = 0.5
    mapped = hide.squeeze(detail(c), c, 0.5)
    even = hide.squeeze(tone, c, 0.5)
    assert mapped.min() >= 0.25 - 1e-6 and mapped.max() <= 0.75 + 1e-6  # still in the band
    row, cols = D.SIZE // 2, np.abs(x[0] - 40) < 3  # a patch on the ring, 37..43 mm
    fine = lambda a: (a[row, cols] - np.convolve(a[row], np.ones(61) / 61, "same")[cols]).std()
    assert fine(mapped) > 1.3 * fine(even)
    broad = lambda a: np.convolve(a[row], np.ones(301) / 301, "same")
    span = lambda a: broad(a)[np.abs(x[0] - 50) < 0.1].mean() - broad(a)[np.abs(x[0] - 30) < 0.1].mean()
    assert 0 < span(mapped) < span(even)


def test_picture_loading(tmp_path):
    """Lightness stretched to full range, transparent parts light, and the shape kept."""
    from PIL import Image
    rgba = np.zeros((100, 200, 4), np.uint8)
    rgba[..., :3] = np.linspace(60, 180, 200).astype(np.uint8)[None, :, None]  # grey ramp, low contrast
    rgba[..., 3] = 255
    rgba[:, :20, 3] = 0  # transparent strip on the left
    path = str(tmp_path / "wide.png")
    Image.fromarray(rgba).save(path)
    tone = D.load_tone(path, 100.0)
    row = tone[D.SIZE // 2]
    x = (np.arange(D.SIZE) + 0.5) * D.CELL_MM - D.HALF_MM
    assert row[np.abs(x + 46) < 3].min() > 0.99  # transparent
    assert row[np.abs(x + 39.4) < 0.1].mean() < 0.05 and row[np.abs(x - 49.6) < 0.1].mean() > 0.95  # stretched
    col = tone[:, D.SIZE // 2]
    assert np.all(col[np.abs(x) > 26] == 1) and np.all(col[np.abs(x) < 24] < 0.9)  # 100 x 50 mm, not squashed
