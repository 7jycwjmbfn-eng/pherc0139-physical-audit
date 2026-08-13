#!/usr/bin/env python3
"""Final figure. Rows are the two longest stretches the harness scored as fully
missed, then the longest one it scored as covered, all from the same slice.
Same selection rule for both, so neither row is hand-picked.

Columns: raw CT -> CT with the model's output -> CT with the score marked.

Usage: render4.py <z> <out> [half]
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render2 import arcs_with_pixels, get_ct, clean   # noqa: E402
from fetch_util import SCROLLS as SC                  # noqa: E402

OY, OX = 576, 480
UM = 18.724
HERE = Path(os.environ.get("SLICES",
                           Path(__file__).resolve().parent))
GREEN = np.array([0.15, 0.95, 0.25])
RED = np.array([1.0, 0.12, 0.12])
BLUE = np.array([0.25, 0.6, 1.0])


def base(ct):
    v = ct[ct > 0]
    lo, hi = np.percentile(v, (1, 99.5)) if v.size else (0, 1)
    g = np.clip((ct.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    return np.dstack([g, g, g])


def add_pred(rgb, pb, box):
    y0, y1, x0, x1 = box
    sub = pb[y0:y1, x0:x1]
    out = rgb.copy()
    out[sub] = 0.55 * out[sub] + 0.45 * GREEN
    return out


def _in(a, box):
    y0, y1, x0, x1 = box
    return ((a["ys"] >= y0) & (a["ys"] < y1)
            & (a["xs"] >= x0) & (a["xs"] < x1))


def add_scored(rgb, arcs, box, focus):
    y0, y1, x0, x1 = box
    out = rgb.copy()
    for a in arcs:
        col = RED if a["gone"] else (BLUE if a is focus else None)
        if col is None:
            continue
        m = _in(a, box)
        if not m.any():
            continue
        out[a["ys"][m] - y0, a["xs"][m] - x0] = \
            0.35 * out[a["ys"][m] - y0, a["xs"][m] - x0] + 0.65 * col
    return out


def bracket(ax, arcs, box, focus, pad=9):
    y0, y1, x0, x1 = box
    for a in arcs:
        if not (a["gone"] or a is focus):
            continue
        m = _in(a, box)
        if not m.any():
            continue
        yy, xx = a["ys"][m] - y0, a["xs"][m] - x0
        # only box stretches that lie wholly inside the crop
        if m.sum() < len(a["ys"]):
            continue
        ax.add_patch(plt.Rectangle(
            (xx.min() - pad, yy.min() - pad),
            xx.max() - xx.min() + 2 * pad, yy.max() - yy.min() + 2 * pad,
            fill=False, ec="#ffe14d", lw=1.2, ls=(0, (4, 3))))


def scalebar(ax, h, w, um_len=500):
    n = um_len / UM
    ax.plot([w * 0.05, w * 0.05 + n], [h * 0.955] * 2, "-", c="w", lw=2.6)
    ax.text(w * 0.05, h * 0.935, f"{um_len/1000:g} mm", color="w",
            fontsize=8.5, va="bottom")


def main(gz, out, half=70, scroll="0139"):
    arcs, lab, pb = arcs_with_pixels(gz, scroll)
    gone = sorted([a for a in arcs if a["gone"]], key=lambda a: -a["n"])
    hits = [a for a in arcs if a["hit"]]
    rows = []
    for g in gone:
        if len(rows) == 2:
            break
        box = (int(g["cy"]) - half, int(g["cy"]) + half,
               int(g["cx"]) - half, int(g["cx"]) + half)
        # longest covered stretch lying wholly inside the same crop, so the
        # comparison is same tissue, same slice, same field of view. PHerc1203
        # carries about a tenth the stretches per slice that PHerc0139 does,
        # so fall back to one that is mostly in view rather than show none.
        inside = [h for h in hits if _in(h, box).sum() == len(h["ys"])]
        if not inside:
            inside = [h for h in hits
                      if _in(h, box).sum() >= 0.8 * len(h["ys"])]
        if not inside:
            continue          # no covered stretch to compare against here
        ct = get_ct(gz, *box, scroll=scroll)
        if (ct > 0).mean() < 0.6:
            continue          # crop straddles the edge of the scanned volume
        rows.append((g, max(inside, key=lambda a: a["n"]), ct))
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(11.4, 3.9 * n), dpi=170,
                             squeeze=False)
    titles = ["CT the model was given",
              "the model's own output (green)",
              "what the harness scored"]
    for r, (a, ctrl, ct) in enumerate(rows):
        cy, cx = int(a["cy"]), int(a["cx"])
        box = (cy - half, cy + half, cx - half, cx + half)
        b = base(ct)
        p = add_pred(b, pb, box)
        s = add_scored(p, arcs, box, ctrl)
        for c, img in enumerate((b, p, s)):
            ax = axes[r][c]
            ax.imshow(img, interpolation="nearest")
            bracket(ax, arcs, box, ctrl)
            scalebar(ax, 2 * half, 2 * half)
            clean(ax)
            if r == 0:
                ax.set_title(titles[c], fontsize=11)
        cl = (f"blue: covered {100*ctrl['cov']:.0f}%, {ctrl['n']} px"
              if ctrl is not None else "no covered stretch wholly in view")
        oy, ox = SC[scroll]["origin"][1:]
        axes[r][0].set_ylabel(
            f"red: covered {100*a['cov']:.0f}%, {a['n']} px\n{cl}\n"
            f"z={gz} y={cy+oy} x={cx+ox}", fontsize=8.5)
        axes[r][0].yaxis.set_visible(True)
        axes[r][0].set_yticks([])
    fig.text(0.5, -0.03,
             "Red = sheet stretches the harness scored as fully missed. "
             "Blue = the longest one it scored as covered, in the same crop, "
             "same slice, same tissue.\nEvery scored stretch lying wholly in "
             "view is boxed, in all three columns. Rows are the two longest "
             "missed stretches in this slice that have a covered stretch in "
             "the same field of view,\nso the two are always shown under the "
             f"same conditions.  1 voxel = {UM:g} um.",
             ha="center", fontsize=9)
    fig.tight_layout(pad=0.35)
    fig.savefig(HERE / f"{out}.png", bbox_inches="tight", pad_inches=0.06)
    print("wrote", HERE / f"{out}.png", flush=True)
    oy, ox = SC[scroll]["origin"][1:]
    for a, c, _ in rows:
        print(f"  missed n={a['n']} cov={a['cov']:.3f} "
              f"y={int(a['cy'])+oy} x={int(a['cx'])+ox}"
              + (f"   | control n={c['n']} cov={c['cov']:.3f}"
                 if c is not None else "   | no control in view"), flush=True)


if __name__ == "__main__":
    # render4.py <scroll> <z> <out> [half]
    main(int(sys.argv[2]), sys.argv[3],
         int(sys.argv[4]) if len(sys.argv) > 4 else 70, sys.argv[1])
