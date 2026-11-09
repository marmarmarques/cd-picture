"""Audio CD images: byte groups, CUE layout, the independent check, and the CLI end to end."""

import json
import os
import wave

import numpy as np
import pytest

from cdpicture import audio as A, design as D, ecma130 as E
from cdpicture.__main__ import main
from cdpicture.efm import EFM, TRANSITIONS
from cdpicture.track import Geometry

from test_cli import disc, picture  # noqa: F401  (fixtures)


def write_wav(path, seconds, freq, level=0.3, seed=0):
    n = int(seconds * A.RATE)
    t = np.arange(n) / A.RATE
    rng = np.random.default_rng(seed)
    x = level * np.sin(2 * np.pi * freq * t)[:, None] + 0.01 * rng.standard_normal((n, 2))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2), w.setsampwidth(2), w.setframerate(A.RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return str(path)


@pytest.fixture
def album(tmp_path):
    d = tmp_path / "album"
    d.mkdir()
    write_wav(d / "01 one.wav", 1.0, 440)
    write_wav(d / "02 two.wav", 1.5, 220, level=0.99, seed=1)  # clips: tests the full-scale path
    return str(d)


def test_efm_table_is_a_valid_rll_code():
    words = [format(int(w), "014b") for w in EFM]
    assert len(set(words)) == 256
    for b in words:  # 2..10 zeros between ones inside a word
        ones = [i for i, c in enumerate(b) if c == "1"]
        assert all(3 <= q - p <= 11 for p, q in zip(ones, ones[1:]))
    assert TRANSITIONS[0x0D] == 1 and TRANSITIONS[0xAA] == 5


@pytest.mark.parametrize("level", list(A.LEVELS))
def test_quantizer_hits_the_group_with_the_nearest_value(level):
    q = A.Quantizer(level)
    rng = np.random.default_rng(0)
    s = np.r_[rng.integers(-32768, 32768, 200_000), [32767, 32766, -32768, -32767, 0, -1]].astype(np.int16)
    want = rng.integers(1, 3, len(s)).astype(np.uint8)
    out = q(s, want)
    assert np.all(q.group[out.view(np.uint16) & 0xFF] == want)
    d = np.abs(out.astype(np.int32) - s)
    inner = np.abs(s.astype(np.int32)) < 32767 - 256
    assert d[inner].max() <= q.max_step
    # nearest: no closer value with a low byte in the group exists
    for k in rng.choice(np.flatnonzero(inner), 200, replace=False):
        cand = np.arange(int(s[k]) - d[k] + 1, int(s[k]) + d[k])
        assert not np.any(q.group[cand & 0xFF] == want[k])


def test_plan_and_cue(album):
    files = A.track_files([album])
    tracks = A.plan(files, gap_s=2.0)
    assert tracks[0].start == 0 and tracks[0].gap == 0
    assert tracks[1].start == tracks[0].end + 150
    cue = A.cue_text("x.bin", tracks)
    assert "PREGAP 00:02:00" in cue and f"INDEX 00 {A.msf(tracks[0].end)}" in cue
    assert f"INDEX 01 {A.msf(tracks[1].start)}" in cue
    assert A.msf(4500 + 75 + 3) == "01:01:03"


def test_bin_bytes_land_where_the_picture_wants(tmp_path, album):
    geo = Geometry.nominal()
    design = D.grid_design()
    tracks = A.plan(A.track_files([album]))
    path = str(tmp_path / "a.bin")
    A.write_bin(path, tracks, design, geo, A.Quantizer("medium"), log=lambda *_: None)
    total = tracks[-1].end
    assert os.path.getsize(path) == total * E.RAW_BYTES
    assert A.verify_bin(path, design, geo, total, "medium", samples=total) == []
    # every low byte, forward: its physical half-frame label matches its group
    data = np.fromfile(path, np.uint8)
    j = np.arange(0, len(data), 2)
    frame, slot = E.raw_to_physical(j)
    lab = geo.labels(design, frame, slot)
    t = TRANSITIONS[data[j]]
    assert np.all(np.where(lab == 1, t == 1, (t == 4) | (t == 5)))
    # the high bytes are the music's, apart from carries
    src = np.concatenate([A.read_audio(f) for f in A.track_files([album])])
    assert data[1::2].size > src.size  # gap and padding included


def test_independent_check_catches_a_wrong_byte(tmp_path, album):
    geo = Geometry.nominal()
    design = D.grid_design()
    tracks = A.plan(A.track_files([album]))
    path = str(tmp_path / "a.bin")
    A.write_bin(path, tracks, design, geo, A.Quantizer("extreme"), log=lambda *_: None)
    total = tracks[-1].end
    data = np.fromfile(path, np.uint8)
    lba = 100
    j = lba * 2352 + 1000  # a low byte (even index)
    data[j] = 0xAA if TRANSITIONS[data[j]] == 1 else 0x0D  # swap its group
    data.tofile(path)
    assert A.verify_bin(path, design, geo, total, "extreme", samples=total)


def test_cli_audio_end_to_end(tmp_path, disc, picture, album):  # noqa: F811
    out = tmp_path / "iso" / "album.bin"
    main(["audio", "--disc", disc, "--picture", picture, "--tracks", album, "--level", "mild", "-o", str(out)])
    for ext in (".bin", ".cue", ".sheet.png", ".audio.json"):
        assert (tmp_path / "iso" / f"album{ext}").exists()
    spec = json.loads((tmp_path / "iso" / "album.audio.json").read_text())
    assert [t["title"] for t in spec["tracks"]] == ["01 one", "02 two"]
    assert all(t["max_change_lsb"] <= 255 for t in spec["tracks"])
    with pytest.raises(SystemExit):  # refuses to overwrite
        main(["audio", "--disc", disc, "--picture", picture, "--tracks", album, "-o", str(out)])


def test_cli_audio_grid_loud(tmp_path, disc):  # noqa: F811
    out = tmp_path / "grid.bin"
    main(["audio-grid", "--disc", disc, "--loud", "--r-max", "27", "-o", str(out)])
    data = np.fromfile(out, np.uint8)
    assert set(np.unique(data).tolist()) <= {0x0D, 0xAA}


@pytest.mark.parametrize("w", [0, 1, 133, -7])
def test_write_offset_pre_shift_survives_the_burner(tmp_path, album, w):
    """Burn = the stream shifted by w samples. Pre-shifted for w, the disc carries the picture;
    built for 0, it does not (unless w is a whole number of CIRC frames... and even then it moves)."""
    geo = Geometry.nominal()
    x, y = D.grid_coords()
    design = ((np.floor(x / 0.3) + np.floor(y / 0.3)) % 2).astype(bool)  # 0.3 mm checkerboard
    tracks = A.plan(A.track_files([album]))
    total = tracks[-1].end
    path = str(tmp_path / "a.bin")
    A.write_bin(path, tracks, design, geo, A.Quantizer("extreme"), log=lambda *_: None, offset=A.Offset(w))
    data = np.fromfile(path, np.uint8)
    burned = np.roll(data, 4 * w)  # what the disc holds, stream byte by stream byte
    burned.tofile(str(tmp_path / "disc.bin"))
    problems = A.verify_bin(str(tmp_path / "disc.bin"), design, geo, total, "extreme", samples=200)
    assert all(int(p.split()[1]) in (0, 1, total - 2, total - 1) for p in problems)  # only the wrapped ends
    if w % 6:
        A.write_bin(path, tracks, design, geo, A.Quantizer("extreme"), log=lambda *_: None)
        np.roll(np.fromfile(path, np.uint8), 4 * w).tofile(str(tmp_path / "disc.bin"))
        assert len(A.verify_bin(str(tmp_path / "disc.bin"), design, geo, total, "extreme", samples=200)) >= 5


def test_offset_sweep_bands(tmp_path, disc):  # noqa: F811
    out = tmp_path / "sweep.bin"
    main(["audio-grid", "--disc", disc, "--offset-sweep", "--write-offset", "130", "--r-max", "27", "-o", str(out)])
    spec = json.loads((tmp_path / "sweep.audio.json").read_text())
    assert [b[2] for b in spec["sweep_bands_mm"]] == list(range(130, 136))


def test_find_offset(tmp_path, album):
    geo = Geometry.nominal()
    tracks = A.plan(A.track_files([album]))
    path = str(tmp_path / "a.bin")
    A.write_bin(path, tracks, D.grid_design(), geo, A.Quantizer("mild"), log=lambda *_: None)
    pcm = np.fromfile(path, "<i2")
    rip = np.concatenate([np.zeros(150 * 1176 + 2 * 133, np.int16), pcm])  # pregap + 133-sample shift
    with wave.open(str(tmp_path / "rip.wav"), "wb") as wv:
        wv.setnchannels(2), wv.setsampwidth(2), wv.setframerate(A.RATE)
        wv.writeframes(rip.tobytes())
    assert A.find_offset(path, str(tmp_path / "rip.wav"), points=(0, 1)) == (133, True, 2)
