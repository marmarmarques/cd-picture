"""Burnable CD picture images that can also carry files.

  profile  save one drive + disc + burner's measurements (from disc-probe) as a disc profile
  grid     test disc: burn it to check the geometry
  make     picture disc, optionally with files
  plan     same as make, but only reports what fits and writes the previews
  audio       audio CD (BIN/CUE) whose music draws the picture in its sample low bytes
  audio-grid  audio test disc to check the geometry and along-track value of an audio burn
  audio-offset  measure a burner's audio write offset from a rip of a burned audio image
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import audio as A, design as D, ecma130 as E, hide, iso9660, tonemap, writer
from .profile import Profile, build as build_profile


def _fail(e):
    sys.exit(str(e))


def _profile(args):
    try:
        return Profile.load(args.disc)
    except (OSError, ValueError, KeyError) as e:
        _fail(f"{args.disc}: {e}")


def _total(geo, r_max: float) -> int:
    total = geo.sectors_to_radius(r_max + 0.6)
    try:
        E.check_capacity(total)
    except ValueError as e:
        _fail(f"{e} (--r-max {r_max} with this geometry)")
    return total


def _mb(sectors) -> str:
    return f"{sectors * 2048 / 1e6:.1f} MB"


def cmd_profile(args):
    try:
        prof = build_profile(os.path.abspath(args.geo), os.path.abspath(args.correction) if args.correction else None,
                             args.along_track_frames, args.cd_sector, args.dark, args.light, args.data_tone,
                             args.allow_stopped)
    except (OSError, ValueError, KeyError) as e:
        _fail(e)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(prof, f, indent=2)
    print(f"Disc profile: {os.path.abspath(args.output)}")
    if not args.correction:
        print("  no --correction: lines may bend by a few tenths of a degree")
    print(f"  {args.cd_sector} sectors, along-track {args.along_track_frames} frames "
          "(burn a grid to check it, then edit along_track_frames in the file)")


def cmd_grid(args):
    prof = _profile(args)
    geo = prof.geometry(args.along_track_frames)
    total = _total(geo, args.r_max)
    texs = [prof.light, prof.dark]
    try:
        paths = writer.output_paths(args.output)
    except ValueError as e:
        _fail(e)
    design = D.grid_design(r_max=args.r_max)
    a = prof.along_track_frames if args.along_track_frames is None else args.along_track_frames
    print(f"grid, {total} sectors, along-track {a} frames")
    spec = writer.write_iso(args.output, design, geo, total, texs, prof.offset,
                            meta={"disc": os.path.abspath(args.disc), "design": "grid", "along_track_frames": a})
    hide.previews(paths, 1 - D.darkness(design), design, np.zeros((hide.MAP_CELLS,) * 2), prof.data_tone,
                  geo, total, "grid", layout=False)
    _verify_and_report(args.output, design, geo, total, texs, prof.offset, None, None, paths, spec)


def _contrast(text):
    if text == "auto":
        return None
    pct = float(text)
    if not 0 <= pct <= 100:
        raise argparse.ArgumentTypeError("give a percentage from 0 to 100, or auto")
    return pct / 100


def _room(lim) -> str:
    """Room for files at a few contrasts, e.g. '61 MB at 100 %, 150 MB at 85 %, ...'."""
    parts = []
    for pct in (100, 85, 70, 50):
        lim.set_contrast(pct / 100)
        parts.append(f"{lim.capacity_mb(1.0):.0f} MB at {pct} %")
    return ", ".join(parts) + " contrast"


RING_GAP = 30  # fewest picture sectors between two rings


def _ring_starts(sizes, geo, first, total):
    """Start LBAs for rings of these sizes (in order, inside out) between LBA `first` and `total`,
    with the same radial gap g between rings and half a gap at either end. Returns (starts, g mm)."""
    if not sizes:
        raise ValueError("--place rings needs at least one non-empty file")
    r0, r1 = geo.end_radius(first), geo.end_radius(total)

    def lay(g):
        starts, r = [], r0 + g / 2
        for k in sizes:
            s = max(geo.sectors_to_radius(r), starts[-1] + sizes[len(starts) - 1] + RING_GAP if starts else first)
            starts.append(s)
            r = geo.end_radius(s + k) + g
        return starts, starts[-1] + sizes[-1] <= total

    starts, ok = lay(0.0)
    if not ok:
        raise ValueError(f"files need {sum(sizes)} sectors as rings with picture between them, "
                         f"the picture image has {total - first} after the directories")
    lo, hi = 0.0, r1 - r0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if lay(mid)[1] else (lo, mid)
    return lay(lo)[0], lo


def _layout(args, prof, geo, total, report_room=False):
    """Decide picture vs data sectors. Returns a dict with design, files, region, fill, notes."""
    tone = D.load_tone(args.picture, args.diameter_mm)
    t, contrast = prof.data_tone, args.contrast
    if contrast is not None and args.threshold is not None:
        _fail("--contrast changes a picture's greys; it doesn't work with --threshold")

    toned = tonemap.MODES[args.tone](tone)  # the picture's tone at a contrast, before the squeeze

    def picture(c):
        return D.picture_design(hide.squeeze(toned(c), c, t), args.threshold)

    c = 1.0 if contrast is None else contrast  # without hidden files, auto keeps it all
    out = {"texs": [prof.light, prof.dark], "files": None, "region": None, "fill": None, "tone": tone,
           "frac": None, "contrast": c if args.threshold is None else None}
    place = args.place
    if not args.files:
        out["design"] = picture(c)
        # Keep the 5 file-system sectors blank (plus a few cells, which are sampled by their corners).
        D.clear_inside(out["design"], geo.files_radius(21) + 0.05)
        if report_room:
            lim = hide.Limits(tone, geo, total, t, tone_at=toned)
            out["notes"] = ["room to hide files in this picture: " + _room(lim)]
        return out

    def build(allowed=None, by_size=False):
        return iso9660.FileSystem(args.files, args.label, allowed=allowed, by_size=by_size)

    try:
        if place == "center":
            fs = build()
            r = geo.files_radius(fs.end_lba) + 0.3
            out["design"] = picture(c)
            D.clear_inside(out["design"], r)
            notes = [f"files in the middle out to {r - 0.3:.2f} mm; picture starts at {r:.2f} mm"]
        elif place == "outer":
            # The directories stay at LBA 16 (every reader looks there); only the file data moves.
            # One contiguous run at the end of the image keeps one extent per file, so the
            # directory sizes, and with them the sector count, match the plain layout.
            plain = build()
            start = total - (plain.end_lba - plain.meta_end)
            if start < plain.meta_end:
                raise ValueError(f"files need {plain.end_lba} sectors, the picture image only has {total}")
            fs = build(np.arange(start, total))
            r_meta = geo.files_radius(fs.meta_end) + 0.05
            r_data = geo.end_radius(start) - 0.3
            out["design"] = picture(c)
            D.clear_inside(out["design"], r_meta)
            D.clear_outside(out["design"], max(r_data, r_meta))
            notes = [f"files in the outer ring from {r_data + 0.3:.2f} mm to {geo.end_radius(total):.2f} mm; "
                     f"picture from {r_meta:.2f} to {r_data:.2f} mm"]
            if r_data <= r_meta:
                notes.append("the files take the whole disc: no room is left for the picture")
        elif place == "rings":
            # One extent per file, so each file is one ring of the spiral. Smallest file innermost,
            # biggest outermost: a ring of k sectors is ~k/r wide, so this keeps their widths closer.
            plain = build()
            sizes = sorted(-(-f.size // 2048) for f in plain.files if f.size)
            starts, gap_mm = _ring_starts(sizes, geo, plain.meta_end, total)
            fs = build(np.concatenate([np.arange(s, s + k) for s, k in zip(starts, sizes)]), by_size=True)
            out["design"] = picture(c)
            D.clear_inside(out["design"], geo.files_radius(fs.meta_end) + 0.05)
            notes = [f"{len(sizes)} rings from {geo.end_radius(starts[0]):.2f} to "
                     f"{geo.end_radius(starts[-1] + sizes[-1]):.2f} mm, {gap_mm:.2f} mm of picture between "
                     f"them; smallest file innermost, biggest outermost"]
        elif place == "auto":
            if args.threshold is not None:
                _fail("--place auto always dithers the picture; drop --threshold or use --place center / a mask")
            lim = hide.Limits(tone, geo, total, t, tone_at=toned)
            try:
                fs, mask, h, c = hide.place(lim, build, contrast)
            except ValueError as e:
                if contrast is None:
                    _fail(f"{e}. This picture can hide up to {lim.capacity_mb(1.0):.0f} MB even then; "
                          f"use fewer files or --place center, outer or rings")
                _fail(f"{e}. This picture can hide up to {lim.capacity_mb(1.0):.0f} MB at {contrast:.0%} "
                      f"contrast; lower --contrast or leave it out (auto), use fewer files, or --place center, outer or rings")
            out["contrast"] = c
            out["design"], out["frac"] = hide.retone(hide.squeeze(toned(c), c, t), mask, geo, t)
            out["region"], out["fill"] = mask, mask & ~fs.data_sectors(total)
            notes = [f"contrast {c:.0%}" + (" (lowered so the files fit)" if contrast is None and c < 1 else "")
                     + (": the picture will look faint" if c < 0.5 else ""),
                     f"files hidden in the picture: {100 * mask.mean():.0f} % of sectors, "
                     f"{100 * h:.0f} % of what it could hide at this contrast (up to {lim.capacity_mb(1.0):.0f} MB)"]
            if report_room:
                notes.append("room for files: " + _room(lim))
        else:
            if not os.path.isfile(place):
                _fail(f"--place is auto, center, outer, rings or a mask PNG; {place!r} is not a file")
            share = writer.footprint_share(D.mask_region(place, args.diameter_mm), geo, total)
            fs, t = writer.fit_to_mask(share, build)
            region = share >= t
            out["design"], out["region"] = picture(c), region
            notes = [f"mask {os.path.basename(place)}: {_mb(region.sum())} of data area, "
                     f"{100 * (fs.data_sectors(total) & region).sum() / max(1, region.sum()):.0f} % used",
                     "strict: no file data reaches into the black" if t >= 1 else
                     f"files don't fit strictly ({_mb((share >= 1).sum())}), so sectors {t:.0%} in the white "
                     f"are used too: data reaches into the black at the mask's edges"]
    except ValueError as e:
        _fail(e)
    size = sum(f["bytes"] for f in fs.entries())
    extents = sum(e["extents"] for e in fs.entries())
    out["files"] = fs
    out["notes"] = [f"{len(fs.files)} files, {size / 1e6:.1f} MB in {extents} pieces"] + notes
    return out


def cmd_make(args, write=True):
    prof = _profile(args)
    geo = prof.geometry()
    total = _total(geo, args.r_max)
    try:
        paths = writer.output_paths(args.output, image=write)
    except ValueError as e:
        _fail(e)
    lay = _layout(args, prof, geo, total, report_room=not write)
    design, fs = lay["design"], lay["files"]
    what = os.path.basename(args.picture)
    print(f"{what}: {total} sectors to {args.r_max} mm, {prof.cd_sector}, place {os.path.basename(args.place) if fs else 'no files'}")
    for n in lay.get("notes", []):
        print("  " + n)
    frac = lay["frac"]
    if frac is None:  # files in the middle or in a mask: data exactly in their own sectors
        frac = hide.realised_fraction(fs.data_sectors(total), geo)[0] if fs else np.zeros((hide.MAP_CELLS,) * 2)
    hide.previews(paths, lay["tone"], design, frac, prof.data_tone, geo, total, what, layout=fs is not None)
    if not write:
        print("Previews (data-side view): " + ", ".join(os.path.abspath(paths[k]) for k in ("preview", "layout", "sheet")
                                                       if os.path.exists(paths[k])))
        return
    meta = {"disc": os.path.abspath(args.disc), "design": what, "place": args.place if fs else None,
            "contrast": lay["contrast"], "tone": args.tone,
            "sector_layout": prof.cd_sector, "along_track_frames": prof.along_track_frames}
    try:
        spec = writer.write_iso(args.output, design, geo, total, lay["texs"], prof.offset, files=fs, meta=meta)
    except ValueError as e:
        _fail(e)
    skip = fs.data_sectors(total) if fs else None
    if lay["fill"] is not None:
        # The picture was re-toned for grey in every data sector, so unused ones get noise too.
        writer.fill_random(args.output, np.flatnonzero(lay["fill"]))
        skip = skip | lay["region"]
    _verify_and_report(args.output, design, geo, total, lay["texs"], prof.offset, fs, skip, paths, spec)


def _verify_and_report(path, design, geo, total, texs, offset, fs, skip, paths, spec):
    problems = writer.verify_iso(path, design, geo, total, texs, offset, skip=skip)
    if fs:
        problems += fs.verify(path)
    print("Verified: sampled sectors match the design" + (", every file matches its source" if fs else "")
          if not problems else "VERIFY FAILED, don't burn it: " + "; ".join(problems[:5]))
    print(f"Image: {os.path.abspath(path)}\nSpec: {os.path.abspath(spec)}")
    print("Previews (data-side view): " + ", ".join(os.path.abspath(paths[k]) for k in ("preview", "layout", "sheet")
                                                   if os.path.exists(paths[k])))
    if problems:
        sys.exit(1)


def _audio_paths(out):
    base = os.path.splitext(out)[0]
    paths = {"bin": base + ".bin", "cue": base + ".cue", "spec": base + ".audio.json", "sheet": base + ".sheet.png"}
    if any(os.path.exists(paths[k]) for k in ("bin", "cue")):
        _fail(f"output already exists: {paths['bin']} or {paths['cue']}; choose a new filename")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    return paths


def _audio_write(args, prof, geo, design, tone, tracks, level, title, meta, full_bytes=None, sweep=False):
    paths = _audio_paths(args.output)
    total = tracks[-1].end
    a = prof.along_track_frames if args.along_track_frames is None else args.along_track_frames
    print(f"{title}: {len(tracks)} tracks, {A.msf(total)} ({total} sectors), to {geo.end_radius(total):.1f} mm, "
          f"level {level}, along-track {a} frames")
    offset = A.Offset(args.write_offset, sweep, total)
    if sweep:
        print(f"  offset sweep: 6 bands from the centre out assume write offsets {args.write_offset} .. "
              f"{args.write_offset + 5} samples; the sharp band gives the offset (mod 6)")
    elif args.write_offset:
        print(f"  pre-shifted for a burner write offset of {args.write_offset} samples")
    stats = A.write_bin(paths["bin"], tracks, design, geo, A.Quantizer(level), full_bytes=full_bytes, offset=offset)
    with open(paths["cue"], "w", encoding="utf-8") as f:
        f.write(A.cue_text(os.path.basename(paths["bin"]), tracks))
    meta = {**meta, "level": level, "along_track_frames": a, "write_offset": args.write_offset,
            "offset_sweep": sweep, "sectors": total}
    if sweep:
        meta["sweep_bands_mm"] = [[round(geo.end_radius(b * total // 6), 2), round(geo.end_radius((b + 1) * total // 6), 2),
                                   args.write_offset + b] for b in range(6)]
    A.write_spec(paths["spec"], meta, tracks, stats)
    problems = A.verify_bin(paths["bin"], design, geo, total, level, full_bytes=full_bytes, offset=offset)
    problems += A.check_cue(paths["cue"], paths["bin"], tracks)
    contrast = A.write_sheet(paths["sheet"], tone, design, paths["bin"], geo, total, title, offset=offset)
    print("Verified: sampled sectors carry the right byte groups, CUE matches the BIN" if not problems
          else "VERIFY FAILED, don't burn it: " + "; ".join(problems[:5]))
    print(f"  simulated contrast {contrast:.0%} of a 0D/AA data disc (model, see the sheet)")
    for k in ("cue", "bin", "sheet", "spec"):
        print(f"{k.upper():5s} {os.path.abspath(paths[k])}")
    print("Burn the CUE in DAO / disc-at-once (ImgBurn: Write image file to disc).")
    if problems:
        sys.exit(1)


def cmd_audio(args):
    prof = _profile(args)
    geo = prof.geometry(args.along_track_frames)
    try:
        files = A.track_files(args.tracks)
        fill = geo.sectors_to_radius(args.r_max) if args.fill else None
        tracks = A.plan(files, args.gap, fill)
    except ValueError as e:
        _fail(e)
    tone = D.load_tone(args.picture, args.diameter_mm)
    design = D.picture_design(tone, args.threshold)
    meta = {"disc": os.path.abspath(args.disc), "design": os.path.basename(args.picture), "gap_s": args.gap}
    _audio_write(args, prof, geo, design, tone, tracks, args.level, os.path.basename(args.picture), meta)


def cmd_audio_grid(args):
    prof = _profile(args)
    geo = prof.geometry(args.along_track_frames)
    total = _total(geo, args.r_max)
    design = D.grid_design(r_max=args.r_max)
    tracks = [A.Track("grid", None, 0, 0, total * A.SAMPLES_PER_SECTOR)]
    full = (int(prof.dark[:2], 16), int(prof.light[:2], 16)) if args.loud else None
    if full:
        print("--loud: every byte draws the grid; it plays as a LOUD buzz, don't play it")
    meta = {"disc": os.path.abspath(args.disc), "design": "grid", "loud": args.loud}
    _audio_write(args, prof, geo, design, 1 - D.darkness(design), tracks, args.level, "audio grid", meta, full,
                 sweep=args.offset_sweep)


def cmd_audio_offset(args):
    try:
        shift, pregap, n = A.find_offset(args.bin, args.rip)
    except (OSError, ValueError) as e:
        _fail(e)
    print(f"The rip holds the BIN shifted by {shift:+d} samples ({n} stretches matched bit for bit"
          f"{', rip includes the 2 s pregap' if pregap else ''}).")
    print(f"  {shift % 6} samples past a whole CIRC frame" + ("" if shift % 6 == 0 else
          ": the picture on that disc is smeared into ghosts"))
    print(f"  If the ripper's read offset was set correctly, this is the burner's write offset: rebuild with "
          f"--write-offset {shift}")


def main(argv=None):
    # File names can be any Unicode; don't crash printing them to a Windows code-page console or log.
    sys.stdout.reconfigure(errors="replace")
    p = argparse.ArgumentParser(prog="cdpicture", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("profile", help="save disc-probe's measurements as a disc profile")
    sp.add_argument("--geo", required=True, help="summary.json of a disc-probe CD geo run")
    sp.add_argument("--correction", help="correction.json from disc-probe's geo_polyfit")
    sp.add_argument("--along-track-frames", type=float, default=0.0,
                    help="read-vs-write offset of this drive and burner, found with a grid disc")
    sp.add_argument("--cd-sector", choices=tuple(E.SECTOR_LAYOUTS), default="mode1",
                    help="what the burner writes: mode1 (ImgBurn, cdrecord) or mode2form1 (Windows' own burner)")
    sp.add_argument("--dark", default="0D", help="byte pattern that burns darkest")
    sp.add_argument("--light", default="AA", help="byte pattern that burns lightest")
    sp.add_argument("--data-tone", type=float, default=0.5,
                    help="how grey file data looks, 0 = like --dark, 1 = like --light")
    sp.add_argument("--allow-stopped", action="store_true",
                    help="use a geo run that stopped early (it only covers the radii it reached)")
    sp.add_argument("--output", "-o", default="disc.json")
    sp.set_defaults(func=cmd_profile)

    sp = sub.add_parser("grid", help="test disc to check the geometry and the along-track value")
    sp.add_argument("--disc", required=True, help="disc profile")
    sp.add_argument("--along-track-frames", type=float, help="try this value instead of the profile's")
    sp.add_argument("--r-max", type=float, default=56.0, help="outer edge of the drawing, mm")
    sp.add_argument("--output", "-o", default="iso/grid.iso")
    sp.set_defaults(func=cmd_grid)

    for name, func, text in (("make", cmd_make, "write a picture disc image, optionally with files"),
                             ("plan", lambda a: cmd_make(a, write=False),
                              "like make, but only report what fits and write the previews")):
        sp = sub.add_parser(name, help=text)
        sp.add_argument("--disc", required=True, help="disc profile")
        sp.add_argument("--picture", required=True, help="the picture (any image; colour is turned to grey)")
        sp.add_argument("--files", help="folder to put on the disc")
        sp.add_argument("--place", default="auto",
                        help="where the files go: auto (hidden in the picture, default), center (a block in "
                             "the middle, picture outside it), outer (a ring at the outer edge, picture "
                             "inside it), rings (each file one ring, spread evenly from the middle out, biggest "
                             "files outermost), or a mask PNG (white = files; kept strictly "
                             "inside the white unless the files only fit by using its edges)")
        sp.add_argument("--contrast", type=_contrast, default=None, metavar="PERCENT|auto",
                        help="how much of the picture's contrast to keep (default auto: all of it, or with "
                             "--place auto as much as still lets the files fit)")
        sp.add_argument("--tone", choices=tuple(tonemap.MODES), default="even",
                        help="how the picture loses contrast: even (all tones alike) or detail (broad "
                             "shading first, edges kept); only matters below 100 %% contrast")
        sp.add_argument("--label", default="CDPICTURE", help="disc name shown by the computer, up to 16 characters")
        sp.add_argument("--output", "-o", default="iso/disc.iso")
        sp.add_argument("--diameter-mm", type=float, default=116.0, help="size of the picture on the disc")
        sp.add_argument("--r-max", type=float, default=56.0, help="outer edge of the drawing, mm")
        sp.add_argument("--threshold", type=float,
                        help="cut the picture to black/white at this grey (0-1) instead of dithering; for line art")
        sp.set_defaults(func=func)

    sp = sub.add_parser("audio", help="audio CD (BIN/CUE): the picture rides in the low byte of every sample")
    sp.add_argument("--disc", required=True, help="disc profile")
    sp.add_argument("--picture", required=True, help="the picture (any image; colour is turned to grey)")
    sp.add_argument("--tracks", required=True, nargs="+",
                    help="audio files in order, or folders (their audio files sorted by name)")
    sp.add_argument("--level", choices=tuple(A.LEVELS), default="medium",
                    help="mild: ~27 %% of a data disc's contrast, about inaudible; medium: ~39 %%; "
                         "extreme: ~53 %%, steps up to 110 LSB (audible in quiet passages)")
    sp.add_argument("--gap", type=float, default=2.0, help="seconds between tracks (written into the BIN)")
    sp.add_argument("--fill", action="store_true",
                    help="add a silent picture-only last track so the picture reaches --r-max")
    sp.add_argument("--along-track-frames", type=float,
                    help="along-track value of an AUDIO burn (find it with audio-grid); default: the profile's")
    sp.add_argument("--write-offset", type=int, default=0,
                    help="the burner's audio write offset in samples: where BIN sample 0 really lands. Measure it by "
                         "ripping a burned disc with EAC (read offset set) and comparing to the BIN, or with "
                         "audio-grid --offset-sweep")
    sp.add_argument("--output", "-o", default="iso/audio.bin")
    sp.add_argument("--diameter-mm", type=float, default=116.0, help="size of the picture on the disc")
    sp.add_argument("--r-max", type=float, default=56.0, help="outer edge for --fill, mm")
    sp.add_argument("--threshold", type=float,
                    help="cut the picture to black/white at this grey (0-1) instead of dithering")
    sp.set_defaults(func=cmd_audio)

    sp = sub.add_parser("audio-grid", help="audio test disc: a grid to check an audio burn's geometry")
    sp.add_argument("--disc", required=True, help="disc profile")
    sp.add_argument("--along-track-frames", type=float, help="try this value instead of the profile's")
    sp.add_argument("--level", choices=tuple(A.LEVELS), default="extreme")
    sp.add_argument("--loud", action="store_true",
                    help="set every byte (profile's dark/light), for full contrast; plays as a loud buzz")
    sp.add_argument("--r-max", type=float, default=56.0, help="outer edge of the drawing, mm")
    sp.add_argument("--write-offset", type=int, default=0,
                    help="the burner's audio write offset in samples: where BIN sample 0 really lands. Measure it by "
                         "ripping a burned disc with EAC (read offset set) and comparing to the BIN, or with "
                         "audio-grid --offset-sweep")
    sp.add_argument("--offset-sweep", action="store_true",
                    help="6 radial bands assuming write offsets W .. W+5 (W = --write-offset); the sharp one wins")
    sp.add_argument("--output", "-o", default="iso/audio-grid.bin")
    sp.set_defaults(func=cmd_audio_grid)

    sp = sub.add_parser("audio-offset", help="measure the burner's write offset: compare a rip with the BIN")
    sp.add_argument("--bin", required=True, help="the BIN that was burned")
    sp.add_argument("--rip", required=True, help="a rip of the burned disc (WAV of the whole disc, or a BIN)")
    sp.set_defaults(func=cmd_audio_offset)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
