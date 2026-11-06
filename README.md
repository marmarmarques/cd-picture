# cd-picture

Makes a CD image (`.iso`) that shows a picture on the data side of the disc when burned.
It can also carry real files, which a computer can read like on any data CD.

It needs your disc's exact spiral geometry. [disc-probe](https://github.com/Ashley-QwQ/disc-probe) measures that, this tool only builds the image. It's CD *only*.

Python 3.10+ on Windows or Linux:

```bash
pip install numpy pillow scipy
```

## 1. Measure the disc (disc-probe)

On Windows, in an administrator terminal, inside the disc-probe folder.
For Linux, use [bczhc's port](https://github.com/bczhc/disc-probe).
`H:` is your drive. Use blank discs from the same batch for every step.

Make a random-data disc and burn it: (You can try any burned iso but it may create problems.)

```bash
python -m discprobe random --media cd --output iso/cd-random.iso
```

Check the drive first. This takes a few minutes and tells you whether the drive can be used at all:

```bash
python -m discprobe timing --media cd --drive H: --speed 1411
```

Measure the spiral. This writes `results/geo-H-.../summary.json`:

```bash
python -m discprobe geo --media cd --drive H: --speed 1411 --host-delay-ms 55 --ranges 2000-356000
```

Fit the angle correction. Run it once without `--write-correction` to see the table, then pick
the degree (see below):

```bash
python tools/geo_polyfit.py results/geo-H-... --write-correction correction.json --phi-degree 4
```

### Options worth knowing

**Must** = set it right or the result is wrong. **Check** = the default usually works, but look at it.
**Rarely** = leave it alone unless something goes wrong.

| Option | Importance | Used by | Default (CD) | What it does |
|---|---|---|---|---|
| `--drive` | **Must** | `timing`, `geo` | none | Drive letter, e.g. `H:`. |
| `--media cd` | **Must** | `random`, `timing`, `geo` | `dvd` | Without it disc-probe assumes a DVD. |
| `--speed` | **Must** | `timing`, `geo` | `1411` | Read speed in kB/s. CD 1x = 176, so `1411` = 8x, `2822` = 16x, `4234` = 24x. Use the **lowest speed at which `timing` still says `CAV`**. Slower usually means less jitter in degrees, but some drives switch to CLV when slow. Use the same value for `timing` and `geo`. |
| `--host-delay-ms` | **Must** | `timing`, `geo` | `35` | Random pause before each read. It **must be longer than one turn**: one turn = 60000 / rpm ms. `timing` prints the rpm. At 2080 rpm a turn is 29 ms, so 35 is fine; at 1241 rpm it's 48 ms, so use 55 or more. |
| `--phi-degree` | **Must** | `geo_polyfit` | none | Polynomial degree of the correction. Pick it from the leave-one-out table. |
| `--ranges` | **Check** | `geo` | 3–85 % of the disc | Which LBAs to measure. Without it `geo` stays between 3 % and 85 % of the burned area. The correction is only fitted where it was measured and is held flat outside, so the outer edge would be guessed. `2000-356000` covers 25.6 mm to the end. |
| `--reps` | Rarely | `geo` | `32` | Reads per rung. More is slower but less noisy: the error goes down with √reps. Raise it if jitter is high. |
| `--max-targets` | Rarely | `geo` | `36` | Number of rungs (measured radii). More rungs give a better leave-one-out table. |
| `--chi2-abort` | Rarely | `geo` | `25` | Stops when the fit goes bad. Raise it only if a disc with a wavy pitch keeps tripping it. |
| `--cav` | Rarely | `timing` | off | Asks the drive for constant rotation. Not every drive listens; `timing` shows whether it worked. |
| `--transport imapi` | Rarely | `timing`, `geo` | `auto` | Use this if you can't run as administrator. `auto` falls back to it by itself. |

### Is it going well?

**`timing`:**

| Line | Good | Bad |
|---|---|---|
| `Cache:` strategy | `fua` or `plain` | `evict` works but is slower and noisier |
| `Rotation:` | `CAV` | `CLV`: the results are not disc angles. Try another `--speed` |
| `Spindle steady: T = ... ms (... rpm)` | present | `did not reach a steady speed` |
| `jitter` per read | under ~1° | over ~2° is poor |

**`geo`:** each rung prints a line, and the summary comes at the end:

| Value | Good | Bad |
|---|---|---|
| `residual` per rung | well under 1° | tens of degrees means a whole turn was miscounted. It aborts at 90° |
| `chi2/dof` | about 0.3 to 2 | over 2: the pitch varies. It tells you to run `geo_polyfit`, and the correction handles it. It aborts at 25 |
| `residual rms` | 0.2 to 0.5° | over ~0.8° |
| `Angle uncertainty across the measured span` | ~0.1–0.2° | over ~0.5° |
| `skipped (no stable measurement)` | rare | many of them means the drive is too noisy or the disc is bad |

**`geo_polyfit`:**

| Value | How to read it |
|---|---|
| `constant / quadratic / cubic chi2/dof` | If quadratic is much lower than constant, the pitch drifts and the correction helps. |
| `fit` column | Lower is better, always drops as the degree goes up, so don't use it to choose. |
| `leave-one-out` column | Lower is better, **pick the degree with the lowest value here.** |
| `phi peak to peak` | How much the correction moves the angle in total. Under ~1° is typical. |

## 2. Make the image (this tool)

Run these from this folder. Every disc is either **picture only** or **picture + files**; the
only choice with files is where they go.

**Save your disc profile once.** It keeps the measurements and burner settings, so later
commands only need `--disc`:

```bash
python -m cdpicture profile --geo summary.json --correction correction.json -o disc.json
```

**Burn a grid to check it.** If the lines come out twisted, like in a spiral, the drive reads a little ahead of or behind where it wrote.
Try `--along-track-frames N` until they're straight, then put that N in `disc.json` (`along_track_frames`). It varies a lot by drive and burner:
about 206 for a GH70N with the Windows burner, about 7 for one drive with cdrdao in mode1. disc-probe's manual explains
how to estimate it from the twist.

```bash
python -m cdpicture grid --disc disc.json --along-track-frames 0 -o iso/grid.iso
```

**Draw your picture on the template.** `cd-mask/` has a template that lines up with
`--picture` at the default `--diameter-mm 116`: 3200 x 3200 px over 116 mm (27.6 px/mm). It shows the disc
from the data side, which is how the picture will look on the disc.

- `cd-mask-overlay.png`: put it as the top layer in your editor. The ring you can draw in is clear, the
  area outside it is tinted red, the hole and the area past the edge are black, and blue lines mark the edges. Hide it before you export.
- `cd-mask.png`: white = drawable, black = not. Use it as a selection or a layer mask.

Export a square PNG at the same size, without the overlay. Anything outside the ring isn't burned.
Colour is fine: the picture is turned to perceptual lightness (CIE L*) and stretched to full contrast,
and transparent parts count as light.
The ring is 25.6–56 mm, which fits the default `--r-max 56` and a track that starts around 25.3 mm. If your
disc or `--r-max` is different, change `R_IN` / `R_OUT` at the top of the script and run it again:

```bash
python cd-mask/make_mask.py
```

With `--place center` the files take up the inner part of the ring, so the picture only starts outside
them. `plan` shows where.

**Make a picture disc**, optionally with files:

```bash
python -m cdpicture make --disc disc.json --picture pic.png --files filedir -o iso/disc.iso
```

`--place` decides where the files go:

| `--place` | Files go | Picture |
|---|---|---|
| `auto` (default) | hidden inside the picture | re-toned so the files don't show. Contrast is lowered only if the files need the room (below) |
| `center` | files are packed in the inner section | image starts just outside the files |
| `mask.png` | where the PNG is white | unchanged, so the files show as grey there |

**More data means less contrast.** With `auto`, the contrast is lowered until the files fit, this makes the image grayer overall. To choose it yourself, use `--contrast 70`.

| Contrast | Room for files (685 MB disc) |
|---|---|
| 100 % | about 60 MB |
| 85 % | about 150 MB |
| 70 % | about 240 MB |
| 40 % | about 415 MB |

**Not sure it fits?** `plan` takes the same options, reports how much fits, and writes the previews
without making the image:

```bash
python -m cdpicture plan --disc disc.json --picture pic.png --files filedir -o iso/disc.iso
```

### Options worth knowing

| Option | Used by | Importance | Default | What it does |
|---|---|---|---|---|
| `--geo`, `--correction` | `profile` | **Must** | none | From disc-probe's `geo` and `geo_polyfit`. Without the correction, lines bend by up to its `phi peak to peak`. |
| `--cd-sector` | `profile` | **Must** | `mode1` | Must match the burner (see Burn). |
| `--along-track-frames` | `profile`, `grid` | **Must** | `0` | Find it with a grid disc (above). |
| `--data-tone` | `profile` | **Check** | `0.5` | Estimates how file data looks between dark (0) and light (1). If it's wrong, `auto` file areas show as lighter or darker patches. |
| `--place` | `make`, `plan` | **Check** | `auto` | Where the files go (table above). |
| `--contrast` | `make`, `plan` | **Check** | `auto` | How much of the picture's contrast to keep, in %. `auto` keeps all of it, or with `--place auto` as much as the files allow. Lower means more room for files and a fainter picture. |
| `--tone` | `make`, `plan` | Rarely | `even` | How the picture loses contrast. `even` squeezes all tones alike; `detail` gives up broad shading first and keeps edges and texture, for the same room for files. Only matters below 100 %. Experimental, not yet burned. |
| `--label` | `make`, `plan` | **Check** | `CDPICTURE` | Disc name the computer shows, up to 16 characters. |
| `--dark` / `--light` | `profile` | Rarely | `0D` / `AA` | Byte patterns for dark and light. Which look darkest and lightest depends on the disc; disc-probe's `tiles` disc measures that. |
| `--threshold` | `make`, `plan` | Rarely | off | Cut the picture to black/white at this grey (0–1) instead of dithering. For line art; not with `auto` files. |
| `--diameter-mm` | `make`, `plan` | Rarely | `116` | Size of the picture on the disc. |
| `--r-max` | all three | Rarely | `56` | Outer edge of the drawing, mm. It errors if the disc can't hold that many sectors. |

Edit `disc.json` by hand to change a profile setting, or run `profile` again.

### Is it going well?

| Line | Meaning |
|---|---|
| `Verified: ... every file matches its source` | The image is right. On `VERIFY FAILED` it exits with an error; don't burn that image. |
| `contrast N %` | `auto`: how much contrast the picture keeps. `(lowered so the files fit)` means the files took some; below 50 % it warns that the picture will look faint. |
| `files hidden in the picture: N % of sectors, M % of what it could hide` | `auto`: M under 100 % means the data is spread thinner than it could be, which hides a slightly wrong `--data-tone` better. It doesn't change the look. |
| `room for files: ... MB at 100 %, ...` | `plan`: how much fits at a few contrasts, to pick `--contrast` or check what `auto` will do. |
| `This picture can hide up to N MB` | The files don't fit. Use fewer files, a lower `--contrast` (or `auto`), or `center`. |
| `N files, X MB in K pieces` | Files are split around the picture. Thousands of pieces is fine. |
| `.preview.png`, `.layout.png`, `.sheet.png` | Look before burning. `layout` shows files in yellow; `sheet` simulates the burned disc. |

## 3. Burn

Burn the `.iso` as a disc image at a low speed (8x). A GH70N reset its connection at 40x.

- **ImgBurn / cdrecord** write Mode 1, which is the default here.
- **Windows' own burner** writes Mode 2 Form 1, so make the profile with `--cd-sector mode2form1`.

The wrong mode turns the picture into noise. The Windows burner also needs some spare room to
close the disc: on a 359,847-sector disc it took a 359,600-sector image but refused one of 359,800.

## Tests

```bash
python -m pytest
```

GPL-3.0-or-later. Derived from disc-probe, commit `f060089`.
