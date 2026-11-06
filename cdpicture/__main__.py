"""Burnable CD picture images that can also carry files.

  profile  save one drive + disc + burner's measurements (from disc-probe) as a disc profile
  grid     test disc: burn it to check the geometry
  make     picture disc, optionally with files
  plan     same as make, but only reports what fits and writes the previews
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import design as D, ecma130 as E, hide, iso9660, tonemap, writer
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
                             args.along_track_frames, args.cd_sector, args.dark, args.light, args.data_tone)
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

    def build(allowed=None):
        return iso9660.FileSystem(args.files, args.label, allowed=allowed)

    try:
        if place == "center":
            fs = build()
            r = geo.files_radius(fs.end_lba) + 0.3
            out["design"] = picture(c)
            D.clear_inside(out["design"], r)
            notes = [f"files in the middle out to {r - 0.3:.2f} mm; picture starts at {r:.2f} mm"]
        elif place == "auto":
            if args.threshold is not None:
                _fail("--place auto always dithers the picture; drop --threshold or use --place center / a mask")
            lim = hide.Limits(tone, geo, total, t, tone_at=toned)
            try:
                fs, mask, h, c = hide.place(lim, build, contrast)
            except ValueError as e:
                if contrast is None:
                    _fail(f"{e}. This picture can hide up to {lim.capacity_mb(1.0):.0f} MB even then; "
                          f"use fewer files or --place center")
                _fail(f"{e}. This picture can hide up to {lim.capacity_mb(1.0):.0f} MB at {contrast:.0%} "
                      f"contrast; lower --contrast or leave it out (auto), use fewer files, or --place center")
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
                _fail(f"--place is auto, center or a mask PNG; {place!r} is not a file")
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
                             "the middle, picture outside it), or a mask PNG (white = files; kept strictly "
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

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
