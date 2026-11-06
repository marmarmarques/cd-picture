"""End to end through the command line, on a nominal geometry and a small disc area."""

import json
import os

import numpy as np
import pytest
from PIL import Image

from cdpicture import ecma130 as E
from cdpicture.__main__ import main
from cdpicture.profile import Profile
from cdpicture.track import Geometry


@pytest.fixture
def disc(tmp_path):
    g = Geometry.nominal()
    geo = tmp_path / "summary.json"
    geo.write_text(json.dumps({"fit": {"media": "cd", "u": g.u, "v": g.v, "bit_nm_assumed": g.bit_nm}}))
    corr = tmp_path / "correction.json"
    corr.write_text(json.dumps({"radius_valid_mm": [25, 56], "poly_in_x": [0.1, 0.2]}))
    out = tmp_path / "disc.json"
    main(["profile", "--geo", str(geo), "--correction", str(corr), "--along-track-frames", "7", "-o", str(out)])
    return str(out)


@pytest.fixture
def picture(tmp_path):
    x = np.linspace(0, 1, 400)
    path = tmp_path / "pic.png"
    Image.fromarray((np.add.outer(x, x) / 2 * 255).astype(np.uint8)).save(path)  # diagonal grey ramp
    return str(path)


@pytest.fixture
def files(tmp_path):
    src = tmp_path / "files"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_bytes(b"hello")
    (src / "sub" / "b.bin").write_bytes(os.urandom(200_000))
    return str(src)


def test_profile_round_trip(disc):
    p = Profile.load(disc)
    assert p.along_track_frames == 7 and p.cd_sector == "mode1" and p.offset == E.SECTOR_LAYOUTS["mode1"]
    assert p.geometry().u < p.geometry(0).u  # along-track shortens u
    assert p.geometry().correction is not None


def test_grid(disc, tmp_path):
    main(["grid", "--disc", disc, "--r-max", "26.5", "-o", str(tmp_path / "grid.iso")])
    assert (tmp_path / "grid.preview.png").exists()


@pytest.mark.parametrize("place", ["auto", "center", "mask"])
def test_make_with_files(disc, picture, files, tmp_path, place):
    if place == "mask":
        place = str(tmp_path / "mask.png")
        Image.fromarray(np.tile(np.r_[np.zeros(200), np.full(200, 255)].astype(np.uint8), (400, 1))).save(place)
    out = str(tmp_path / "disc.iso")
    main(["plan", "--disc", disc, "--picture", picture, "--files", files, "--place", place,
          "--r-max", "27", "-o", out])
    assert not os.path.exists(out)
    main(["make", "--disc", disc, "--picture", picture, "--files", files, "--place", place,
          "--r-max", "27", "-o", out])  # exits non-zero if verification fails
    spec = json.load(open(str(tmp_path / "disc.image.json")))
    assert len(spec["files"]["entries"]) == 2


def test_make_picture_only_threshold(disc, picture, tmp_path):
    main(["make", "--disc", disc, "--picture", picture, "--threshold", "0.5", "--r-max", "26.5",
          "-o", str(tmp_path / "p.iso")])


def test_too_many_files_is_refused(disc, picture, tmp_path):
    big = tmp_path / "big"
    big.mkdir()
    (big / "x.bin").write_bytes(bytes(40_000_000))
    with pytest.raises(SystemExit) as e:
        main(["make", "--disc", disc, "--picture", picture, "--files", str(big), "--r-max", "27",
              "-o", str(tmp_path / "x.iso")])
    assert "can hide up to" in str(e.value)


def test_contrast(disc, picture, files, tmp_path):
    out = str(tmp_path / "c.iso")
    main(["make", "--disc", disc, "--picture", picture, "--files", files, "--contrast", "60",
          "--r-max", "27", "-o", out])
    assert json.load(open(str(tmp_path / "c.image.json")))["contrast"] == 0.6
    with pytest.raises(SystemExit):
        main(["plan", "--disc", disc, "--picture", picture, "--threshold", "0.5", "--contrast", "60",
              "--r-max", "27", "-o", out])


def test_detail_tone(disc, picture, files, tmp_path, capsys):
    out = str(tmp_path / "d.iso")
    main(["make", "--disc", disc, "--picture", picture, "--files", files, "--tone", "detail", "--contrast", "60",
          "--r-max", "27", "-o", out])
    assert json.load(open(str(tmp_path / "d.image.json")))["tone"] == "detail"
    big = tmp_path / "big"
    big.mkdir()
    (big / "x.bin").write_bytes(os.urandom(25_000_000))
    main(["plan", "--disc", disc, "--picture", picture, "--files", str(big), "--tone", "detail",
          "--r-max", "27", "-o", str(tmp_path / "p.iso")])
    assert "lowered so the files fit" in capsys.readouterr().out


def test_files_may_fill_the_last_sector(disc, picture, tmp_path):
    """Lowered contrast can hand the files every sector up to the end of the image."""
    big = tmp_path / "big"
    big.mkdir()
    (big / "x.bin").write_bytes(os.urandom(27_000_000))
    out = tmp_path / "full.iso"
    main(["make", "--disc", disc, "--picture", picture, "--files", str(big), "--r-max", "27", "-o", str(out)])
    spec = json.load(open(str(tmp_path / "full.image.json")))
    assert spec["files"]["end_lba"] == spec["total_sectors"] and os.path.getsize(out) == spec["total_sectors"] * 2048


def test_previews_show_only_the_disc(disc, picture, files, tmp_path):
    """Opaque on the track, transparent in the hole and past the end; the layout shows the data in orange."""
    main(["plan", "--disc", disc, "--picture", picture, "--files", files, "--r-max", "27",
          "-o", str(tmp_path / "v.iso")])
    preview = np.asarray(Image.open(str(tmp_path / "v.preview.png")))
    layout = np.asarray(Image.open(str(tmp_path / "v.layout.png")))
    assert preview.shape == layout.shape == (1200, 1200, 4)
    on, hole, edge = (600, 860), (600, 600), (600, 1150)  # 26 mm, centre, 55 mm (row, column; 10 px/mm)
    for img in (preview, layout):
        assert img[on][3] == 255 and img[hole][3] == 0 and img[edge][3] == 0
    ring = layout[..., 3] == 255
    assert (layout[ring][:, 0].astype(int) - layout[ring][:, 2]).max() > 20  # some orange (red > blue)
    assert os.path.exists(str(tmp_path / "v.sheet.png"))


def test_auto_contrast_makes_room(disc, picture, tmp_path, capsys):
    big = tmp_path / "big"
    big.mkdir()
    (big / "x.bin").write_bytes(os.urandom(27_000_000))
    main(["plan", "--disc", disc, "--picture", picture, "--files", str(big), "--r-max", "27",
          "-o", str(tmp_path / "x.iso")])
    assert "lowered so the files fit" in capsys.readouterr().out
