"""Audio CD (CD-DA) picture images: playable music whose sample LOW bytes draw the picture.

An audio track has no scrambler, no sector header and no EDC/ECC: every byte of the 2352-byte
sectors goes through CIRC as it is, in the same order as a data sector's raw bytes (ECMA-130
clause 16 swaps each byte pair, which puts the little-endian sample's high byte first, as
Red Book has it). So BIN byte j lands at E.raw_to_physical(j), exactly like raw data byte j.

Each 16-bit sample is moved to the nearest value whose LOW byte belongs to the dark or light
group the picture wants at that byte's place; the high byte only changes when that step
carries. A group is the set of byte values whose EFM word has a given number of transitions
(0D has 1 and burns dark, AA has 5 and burns light; disc-probe's CD ring scan orders 0D, 10,
random, 63, AA from dark to light). Half the data symbols carry the picture, so it shows with
roughly a quarter to a half of a data disc's contrast; see `LEVELS`.

The output is a BIN (raw little-endian 16-bit stereo, 44.1 kHz) and a CUE for a DAO burn.
Track 1 starts at LBA 0 after the burner's own 2 s pregap, like a data disc, so the profile's
geometry applies. Gaps between tracks are written into the BIN (INDEX 00), so the picture
runs through them. The along-track offset of an AUDIO burn is not measured yet: burn
`audio-grid` first.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import wave
from dataclasses import dataclass, asdict

import numpy as np

from . import design as D, ecma130 as E
from .efm import TRANSITIONS
from .track import Geometry

RATE = 44100
SAMPLES_PER_SECTOR = 588  # stereo samples
PREGAP = 150
CHUNK = 4_000_000  # interleaved samples per processing chunk (multiple of 2)
AUDIO_EXT = (".flac", ".wav", ".mp3", ".m4a", ".ogg", ".opus", ".aiff", ".aif", ".ape", ".wv")

# EFM transitions allowed for (dark, light) low bytes, from mild (small steps, ~27 % of a data
# disc's contrast) to extreme (only the 0D-like and AA-like bytes, ~53 %, steps up to 110 LSB).
LEVELS = {"mild": ((1, 2), (4, 5)), "medium": ((1,), (4, 5)), "extreme": ((1,), (5,))}
DARK, LIGHT = 1, 2


def classes(level: str) -> np.ndarray:
    """group[v] for every byte value v: DARK, LIGHT or 0 (not used)."""
    dark, light = LEVELS[level]
    return np.where(np.isin(TRANSITIONS, dark), DARK,
                    np.where(np.isin(TRANSITIONS, light), LIGHT, 0)).astype(np.uint8)


class Quantizer:
    """Moves samples to the nearest value whose low byte is in the wanted group."""

    def __init__(self, level: str):
        self.level = level
        self.group = classes(level)
        self.step = np.zeros((3, 256), np.int32)
        for c in (DARK, LIGHT):
            members = np.flatnonzero(self.group == c)
            for v in range(256):
                d = ((members - v + 128) % 256) - 128
                self.step[c, v] = d[np.argmin(np.abs(d) * 2 + (d < 0))]
        # Near full scale the nearest member can be past the limit: use the one just inside.
        self.top = np.zeros(3, np.int32)
        self.bottom = np.zeros(3, np.int32)
        for c in (DARK, LIGHT):
            v = np.arange(32767, 32767 - 256, -1)
            self.top[c] = v[self.group[v & 0xFF] == c][0]
            v = np.arange(-32768, -32768 + 256)
            self.bottom[c] = v[self.group[v & 0xFF] == c][0]

    @property
    def max_step(self) -> int:
        return int(np.abs(self.step[1:]).max())

    def __call__(self, s: np.ndarray, want: np.ndarray) -> np.ndarray:
        s = s.astype(np.int32)
        u = s + self.step[want, s & 0xFF]
        u = np.where(u > 32767, self.top[want], u)
        u = np.where(u < -32768, self.bottom[want], u)
        return u.astype(np.int16)


def read_audio(path: str) -> np.ndarray:
    """Interleaved int16 stereo at 44.1 kHz. 16-bit stereo 44.1 kHz WAV is read directly,
    anything else through ffmpeg (which must be installed)."""
    if path.lower().endswith(".wav"):
        with wave.open(path) as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (2, 2, RATE):
                return np.frombuffer(w.readframes(w.getnframes()), "<i2").copy()
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "2", "-ar", str(RATE), "-"],
                             check=True, capture_output=True).stdout
    except FileNotFoundError:
        raise ValueError(f"{path}: reading this format needs ffmpeg (or give a 16-bit 44.1 kHz stereo WAV)")
    except subprocess.CalledProcessError as e:
        raise ValueError(f"{path}: ffmpeg could not read it: {e.stderr.decode(errors='replace').strip()[:200]}")
    return np.frombuffer(raw, "<i2").copy()


def track_files(sources) -> list[str]:
    """Audio files in order: a folder is expanded to its audio files sorted by name."""
    out = []
    for s in sources:
        if os.path.isdir(s):
            out += sorted(os.path.join(s, f) for f in os.listdir(s) if f.lower().endswith(AUDIO_EXT))
        else:
            out.append(s)
    if not out:
        raise ValueError("no audio files given")
    return out


@dataclass
class Track:
    title: str
    path: str | None      # None: silence (the picture-only fill track)
    gap: int              # sectors of INDEX 00 before this track, written into the BIN
    start: int            # LBA of INDEX 01
    samples: int          # stereo samples of audio (the rest of the last sector is silence)

    @property
    def sectors(self) -> int:
        return -(-self.samples // SAMPLES_PER_SECTOR)

    @property
    def end(self) -> int:
        return self.start + self.sectors


def plan(paths, gap_s=2.0, fill_to: int | None = None, lengths=None) -> list[Track]:
    """Lay the tracks out from LBA 0. `lengths` (stereo samples per file) skips decoding.
    With `fill_to`, a silent picture-only track runs to that many sectors."""
    gap = round(gap_s * 75)
    tracks, lba = [], 0
    for i, p in enumerate(paths):
        n = lengths[i] if lengths else len(read_audio(p)) // 2
        if n == 0:
            raise ValueError(f"{p}: no audio")
        g = 0 if i == 0 else gap
        t = Track(os.path.splitext(os.path.basename(p))[0], p, g, lba + g, n)
        tracks.append(t)
        lba = t.end
    if fill_to is not None and fill_to - lba - gap >= 4 * 75:  # Red Book: a track lasts at least 4 s
        tracks.append(Track("picture", None, gap, lba + gap, (fill_to - lba - gap) * SAMPLES_PER_SECTOR))
    end, gaps = tracks[-1].end, sum(t.gap for t in tracks)
    if end > E.CAPACITY:
        spare = E.CAPACITY - (end - gaps)
        hint = (f"; the music alone fits, so shorten the gaps: --gap {max(0, spare // max(1, len(paths) - 1)) // 75}"
                if spare >= 0 else "; the music alone is too long, drop a track")
        raise ValueError(f"the tracks take {msf(end)} ({end} sectors: {end - gaps} music, {gaps} in "
                         f"{gap_s:g} s gaps), more than an 80 min CD's {msf(E.CAPACITY)}{hint}")
    E.check_capacity(end)
    return tracks


def msf(lba: int) -> str:
    return f"{lba // 4500:02d}:{lba // 75 % 60:02d}:{lba % 75:02d}"


def cue_text(bin_name: str, tracks: list[Track]) -> str:
    q = lambda s: s.replace('"', "'")
    lines = [f'FILE "{q(bin_name)}" BINARY']
    for i, t in enumerate(tracks, 1):
        lines += [f"  TRACK {i:02d} AUDIO", f'    TITLE "{q(t.title)}"']
        if i == 1:
            lines.append("    PREGAP 00:02:00")
        elif t.gap:
            lines.append(f"    INDEX 00 {msf(t.start - t.gap)}")
        lines.append(f"    INDEX 01 {msf(t.start)}")
    return "\n".join(lines) + "\n"


class Offset:
    """Where the burner puts BIN sample k: at stream sample k + shift. Burners shift audio by a
    drive-specific write offset; anything but a multiple of 6 samples (one CIRC frame) sends every
    byte down a different interleave path and smears the picture into ghosts. With `sweep`, the
    disc is cut into 6 radial bands that assume offset, offset+1, .. offset+5 (to find it)."""

    BANDS = 6

    def __init__(self, samples: int = 0, sweep: bool = False, total_sectors: int = 1):
        self.samples, self.sweep, self.total = samples, sweep, total_sectors

    def band(self, lba):
        return np.minimum(self.BANDS - 1, np.asarray(lba) * self.BANDS // self.total)

    def at_lba(self, lba):
        """Shift in stereo samples for data around this LBA."""
        lba = np.asarray(lba)
        return self.samples + (self.band(lba) if self.sweep else np.zeros(lba.shape, np.int64))

    def edges(self):
        return [b * self.total // self.BANDS for b in range(1, self.BANDS)] if self.sweep else []


def want_groups(k0: int, n: int, design: np.ndarray, geo: Geometry, byte: int = 0,
                offset: Offset | None = None) -> np.ndarray:
    """Wanted group of the low (byte 0) or high (byte 1) byte of interleaved samples k0 .. k0+n,
    i.e. BIN byte 2k + byte, burned at stream byte 2k + byte + 4 * shift. CIRC delays the two
    bytes of one sample differently."""
    k = np.arange(k0, k0 + n, dtype=np.int64)
    shift = 0 if offset is None else offset.at_lba(k // (2 * SAMPLES_PER_SECTOR))
    frame, slot = E.raw_to_physical(2 * k + byte + 4 * shift)
    return np.where(geo.labels(design, frame, slot) == 1, DARK, LIGHT).astype(np.uint8)


def _db(x):
    return 20 * math.log10(max(x, 1e-12) / 32768)


def write_bin(path: str, tracks: list[Track], design, geo: Geometry, quant: Quantizer, log=print,
              full_bytes: tuple[int, int] | None = None, offset: Offset | None = None) -> list[dict]:
    """Write the BIN; returns per-track change statistics. With full_bytes = (dark, light), every
    byte is set (a loud calibration grid, not music)."""
    stats = []
    k = 0  # interleaved sample index in the BIN

    def emit(fh, pcm, acc=None):
        nonlocal k
        for a in range(0, len(pcm), CHUNK):
            s = pcm[a:a + CHUNK]
            want = want_groups(k + a, len(s), design, geo, offset=offset)
            if full_bytes:
                hi = want_groups(k + a, len(s), design, geo, byte=1, offset=offset)
                pick = lambda w: np.where(w == DARK, full_bytes[0], full_bytes[1]).astype(np.uint16)
                out = (pick(want) | pick(hi) << 8).view(np.int16)
            else:
                out = quant(s, want)
            fh.write(out.astype("<i2").tobytes())
            if acc is not None:
                d = out.astype(np.float64) - s
                acc[0] += np.sum(d * d)
                acc[1] += np.sum(s.astype(np.float64) ** 2)
                acc[2] = max(acc[2], float(np.abs(d).max()))
        k += len(pcm)

    with open(path, "wb") as fh:
        for t in tracks:
            emit(fh, np.zeros(t.gap * SAMPLES_PER_SECTOR * 2, np.int16))
            if t.path is None:
                pcm = np.zeros(t.samples * 2, np.int16)
            else:
                pcm = read_audio(t.path)
                if len(pcm) // 2 != t.samples:
                    raise ValueError(f"{t.path}: length changed since planning")
            pad = (t.sectors * SAMPLES_PER_SECTOR - t.samples) * 2
            pcm = np.concatenate([pcm, np.zeros(pad, np.int16)])
            acc = [0.0, 0.0, 0.0]
            emit(fh, pcm, acc)
            st = {"title": t.title, "start_lba": t.start, "sectors": t.sectors, "max_change_lsb": int(acc[2]),
                  "noise_rms_dbfs": round(_db(math.sqrt(acc[0] / len(pcm))), 1),
                  "music_rms_dbfs": round(_db(math.sqrt(acc[1] / len(pcm))), 1)}
            del pcm
            stats.append(st)
            log(f"  {len(stats):2d} {t.title[:40]:40s} {msf(t.start)}  {geo.end_radius(t.start):5.1f}-"
                f"{geo.end_radius(t.end):5.1f} mm  noise {st['noise_rms_dbfs']:6.1f} dBFS, max step {st['max_change_lsb']}")
    if k * 2 != tracks[-1].end * E.RAW_BYTES:
        raise AssertionError("BIN length does not match the track plan")
    return stats


# --- independent check ------------------------------------------------------------------------
# Nothing below uses the writer's tables or sampling: positions come from an explicit
# enumeration of ECMA-130 figure C.4, groups from counting EFM bits again, and the geometry
# from the closed form written out again (as writer.verify_iso does for data discs).


def verify_bin(path, design, geo: Geometry, total_sectors, level, samples=300, seed=3, full_bytes=None,
               offset: Offset | None = None):
    if os.path.getsize(path) != total_sectors * E.RAW_BYTES:
        return ["BIN size is not the planned number of 2352-byte sectors"]
    from .efm import EFM
    trans = np.array([format(int(w), "014b").count("1") for w in EFM])
    dark_t, light_t = LEVELS[level]
    slots = np.array(list(range(12)) + list(range(16, 28)))
    words = np.repeat([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], 2)
    delays = np.array([3, 6, 11, 14, 19, 22, 27, 30, 35, 38, 43, 46,
                       65, 68, 73, 76, 81, 84, 89, 92, 97, 100, 105, 108])
    u, v, bit_nm, correction = geo.u, geo.v, geo.bit_nm, geo.correction
    picks = sorted({0, total_sectors - 1, *np.random.default_rng(seed).integers(0, total_sectors, samples).tolist()})
    data = np.memmap(path, np.uint8, "r")
    problems = []
    offset = offset or Offset()
    for lba in picks:
        if any(abs(lba - e) <= 2 for e in offset.edges()):
            continue  # a sweep band edge: bytes there may belong to either band
        sh = 4 * int(offset.at_lba(lba))  # bytes: BIN byte j is burned at stream byte j + sh
        frames = np.arange(lba * 98, lba * 98 + 98 + 108)[:, None]
        raw = ((frames - delays) * 24 + 2 * words + np.tile([0, 1], 12)) ^ 1  # stream byte
        ok = (raw - sh >= lba * 2352) & (raw - sh < (lba + 1) * 2352) & (full_bytes is not None or raw % 2 == 0)
        raw, frame = raw[ok], np.broadcast_to(frames, ok.shape)[ok]
        slot = np.broadcast_to(slots, ok.shape)[ok]
        n = frame * 588 + np.where(slot < 12, 144.5, 416.5)
        s = np.sqrt(u ** 2 + v * n / np.pi)
        angle = -2 * n / (s + u)
        radius = bit_nm / 1e6 * s
        if correction is not None:
            angle += correction(radius) * np.pi / 180
        x, y = radius * np.cos(angle), radius * np.sin(angle)
        col, row = np.floor((x + 60) * 50).astype(int), np.floor((60 - y) * 50).astype(int)
        inside = (row >= 0) & (row < 6000) & (col >= 0) & (col < 6000)
        label = np.where(inside, design[np.clip(row, 0, 5999), np.clip(col, 0, 5999)], 0)
        if design.dtype.kind == "f":
            turn = np.floor(2 * n / (s + u) / (2 * np.pi))
            label = (label > np.mod(turn * ((np.sqrt(5) - 1) / 2), 1)).astype(np.int64)
        got = np.asarray(data[raw - sh])
        if full_bytes is not None:
            bad = got != np.where(label == 1, full_bytes[0], full_bytes[1])
        else:
            t = trans[got]
            bad = np.where(label == 1, ~np.isin(t, dark_t), ~np.isin(t, light_t))
        if bad.any():
            problems.append(f"LBA {lba} BIN byte {raw[bad][0] - sh}: wrong group for the picture there")
            if len(problems) >= 10:
                break
    return problems


def check_cue(cue_path, bin_path, tracks: list[Track]) -> list[str]:
    """Re-read the CUE and check every INDEX against the plan and the BIN size."""
    problems, idx = [], []
    for line in open(cue_path, encoding="utf-8"):
        p = line.split()
        if p[:1] == ["INDEX"]:
            m, s_, f = (int(x) for x in p[2].split(":"))
            idx.append((int(p[1]), (m * 60 + s_) * 75 + f))
    want = []
    for i, t in enumerate(tracks):
        if i and t.gap:
            want.append((0, t.start - t.gap))
        want.append((1, t.start))
    if idx != want:
        problems.append("CUE indices differ from the track plan")
    if os.path.getsize(bin_path) != tracks[-1].end * E.RAW_BYTES:
        problems.append("BIN size differs from the CUE's last track end")
    return problems


# --- what the disc should look like -----------------------------------------------------------
# A model, not a photo: brightness ~ mean EFM transitions of the data symbols, which orders the
# measured CD rings (0D, 10, random, 63, AA) correctly. Parity, subcode and merging bits are left
# out; they are the same on average everywhere and only dilute the contrast further.


def disc_views(bin_path, design, geo: Geometry, total_sectors, size=900, every=4, offset: Offset | None = None):
    """(audio, data): simulated brightness rasters (mean EFM transitions) from the BIN's actual
    bytes, and from the 0D/AA bytes a data disc would carry at the same places; NaN off-track."""
    data = np.memmap(bin_path, np.uint8, "r")
    acc = np.zeros((2, size, size))
    cnt = np.zeros((size, size))
    last = total_sectors * 98 + 108
    for a in range(0, last, 2_000_000):
        frames = np.arange(a, min(a + 2_000_000, last), every)
        for half, sl in ((0, slice(0, 12)), (1, slice(12, 24))):
            sl_slots = E.DATA_SLOTS[sl]
            raw = E.physical_to_raw(frames[:, None], sl_slots[None, :])
            if offset is not None:  # stream byte -> the BIN byte the burner put there
                raw = raw - 4 * offset.at_lba(frames // 98)[:, None]
            ok = (raw >= 0) & (raw < len(data))
            b = np.asarray(data[np.clip(raw, 0, len(data) - 1)])
            t = np.where(ok, TRANSITIONS[b], 0).sum(1) / np.maximum(ok.sum(1), 1)
            keep = ok.all(1)
            lab = geo.labels(design, frames[keep], np.full(keep.sum(), sl_slots[0]))
            n = Geometry.half_frame_bit(frames[keep], np.full(keep.sum(), sl_slots[0]))
            ix, iy = geo.raster_index(n, size)
            np.add.at(acc[0], (iy, ix), t[keep])
            np.add.at(acc[1], (iy, ix), np.where(lab == 1, TRANSITIONS[0x0D], TRANSITIONS[0xAA]))
            np.add.at(cnt, (iy, ix), 1)
    with np.errstate(invalid="ignore"):
        return acc[0] / cnt, acc[1] / cnt


def write_sheet(path, tone, design, bin_path, geo, total_sectors, title, size=900, offset: Offset | None = None):
    """picture | simulated audio disc (from the BIN) | same picture as a 0D/AA data disc, all
    drawn with one brightness scale so the contrasts compare directly."""
    from PIL import Image, ImageDraw
    audio, data = disc_views(bin_path, design, geo, total_sectors, size, offset=offset)
    mid = np.nanmean(audio)
    span = TRANSITIONS[0xAA] - TRANSITIONS[0x0D]
    grey = lambda t: np.where(np.isnan(t), 255, np.clip(128 + (t - mid) / span * 220, 0, 255)).astype(np.uint8)
    pic = np.asarray(Image.fromarray((np.clip(tone, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.BOX))
    out = Image.new("RGB", (3 * size + 40, size + 70), "white")
    d = ImageDraw.Draw(out)
    small = design_small(design, size)
    def cont(t):
        lt, dk = t[(small == 0) & ~np.isnan(t)], t[(small == 1) & ~np.isnan(t)]
        return float(lt.mean() - dk.mean()) if len(lt) and len(dk) else float("nan")
    labels = ("picture", f"audio disc, simulated from the BIN (contrast {cont(audio) / span:.0%} of 0D/AA)",
              "same picture as a 0D/AA data disc (100 %)")
    for i, (p, label) in enumerate(zip((pic, grey(audio), grey(data)), labels)):
        out.paste(Image.fromarray(p), (10 + i * (size + 10), 50))
        d.text((10 + i * (size + 10), 30), label, fill="black")
    d.text((10, 8), title + "   (model: mean EFM transitions of the data symbols; parity and merging bits left out)",
           fill="black")
    out.save(path)
    return cont(audio) / span


def design_small(design, size):
    """Design box-sampled to size x size: 1 where mostly dark, 0 mostly light, -1 in between."""
    from PIL import Image
    d = np.asarray(Image.fromarray(design.astype(np.float32)).resize((size, size), Image.BOX))
    return np.where(d > 0.7, 1, np.where(d < 0.3, 0, -1))


def write_spec(path, meta: dict, tracks: list[Track], stats: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        rows = [{**asdict(t), "path": t.path and os.path.abspath(t.path), **s} for t, s in zip(tracks, stats)]
        json.dump({**meta, "tracks": rows}, f, indent=1,
                  ensure_ascii=False)


def read_pcm_memmap(path):
    """int16 memmap of a BIN, or of a WAV's data chunk."""
    if path.lower().endswith(".wav"):
        with open(path, "rb") as f:
            head = f.read(1 << 16)
        pos = 12
        while pos + 8 <= len(head):
            cid, size = head[pos:pos + 4], int.from_bytes(head[pos + 4:pos + 8], "little")
            if cid == b"data":
                return np.memmap(path, "<i2", "r", offset=pos + 8, shape=(size // 2,))
            pos += 8 + size + (size & 1)
        raise ValueError(f"{path}: no WAV data chunk")
    return np.memmap(path, "<i2", "r")


def find_offset(bin_path, rip_path, points=(60, 600, 1800, 3600), block=20000, search=6000):
    """Where the BIN's samples sit in a rip of the burned disc, in stereo samples, relative to
    the BIN's own start and to the same start after a 150-sector pregap (EAC's 'range' rip keeps
    track 1's pregap). Returns (shift, rip_has_pregap, matched points)."""
    b, r = read_pcm_memmap(bin_path), read_pcm_memmap(rip_path)
    results = []
    for sec in points:
        k = sec * RATE * 2
        if k + block > len(b):
            continue
        seg = np.asarray(b[k:k + block])
        for base, pregap in ((0, False), (PREGAP * SAMPLES_PER_SECTOR * 2, True)):
            hit = None
            for sh in range(-search, search + 1):
                a = base + k + 2 * sh
                if 0 <= a and a + block <= len(r) and r[a] == seg[0] and np.array_equal(np.asarray(r[a:a + block]), seg):
                    hit = sh
                    break
            if hit is not None:
                results.append((hit, pregap))
                break
    if not results:
        raise ValueError("no stretch of the BIN was found bit-exact in the rip (different disc, byte-swapped, "
                         "or read errors)")
    if len(set(results)) != 1:
        raise ValueError(f"the rip matches at different shifts {sorted(set(results))}: not one consistent offset")
    return results[0][0], results[0][1], len(results)
