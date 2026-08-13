#!/usr/bin/env python3
"""What the reference is, in one row.

  1. the scan the model works from
  2. the same, with the model's output in green and the stretch the check
     scored as fully missed in red
  3. the second scan of the same scroll, four times finer, put into the same
     frame by the published registration, with the same red stretch on it

Panel 3 is the whole answer to "there is no ground truth for this scroll".
Nobody drew it and no model produced it. It is the object, measured again.

Usage: render5.py <scroll> <z> <local_cy> <local_cx> <half> <out>
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
from fine import fine_crop, HI, UP                    # noqa: E402
from fetch_util import SCROLLS as SC                  # noqa: E402

UM = 18.724
HERE = Path(os.environ.get("SLICES",
                           Path(__file__).resolve().parent))
GREEN = np.array([0.15, 0.95, 0.25])
RED = np.array([1.0, 0.12, 0.12])


def norm(a):
    v = a[a > 0]
    lo, hi = np.percentile(v, (1, 99.5)) if v.size else (0, 1)
    g = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    return np.dstack([g, g, g])


def mark(rgb, arcs, box, up=1):
    y0, y1, x0, x1 = box
    out = rgb.copy()
    for a in arcs:
        if not a["gone"]:
            continue
        m = ((a["ys"] >= y0) & (a["ys"] < y1)
             & (a["xs"] >= x0) & (a["xs"] < x1))
        if not m.any():
            continue
        yy, xx = a["ys"][m] - y0, a["xs"][m] - x0
        if up == 1:
            out[yy, xx] = 0.35 * out[yy, xx] + 0.65 * RED
        else:
            for dy in range(up):
                for dx in range(up):
                    ry, rx = yy * up + dy, xx * up + dx
                    out[ry, rx] = 0.35 * out[ry, rx] + 0.65 * RED
    return out


def bar(ax, w, h, px_per_um, um_len=500):
    n = um_len / px_per_um
    ax.plot([w * 0.05, w * 0.05 + n], [h * 0.955] * 2, "-", c="w", lw=2.8)
    ax.text(w * 0.05, h * 0.932, f"{um_len/1000:g} mm", color="w",
            fontsize=9, va="bottom")


def main(scroll, gz, cy, cx, half, out):
    arcs, lab, pb = arcs_with_pixels(gz, scroll)
    box = (cy - half, cy + half, cx - half, cx + half)
    lo = get_ct(gz, *box, scroll=scroll)
    fi = fine_crop(scroll, gz, *box, verbose=True)
    native, shown = HI[scroll]["native"], HI[scroll]["um"]

    a = norm(lo)
    b = a.copy()
    sub = pb[box[0]:box[1], box[2]:box[3]]
    b[sub] = 0.55 * b[sub] + 0.45 * GREEN
    b = mark(b, arcs, box)
    c = mark(norm(fi), arcs, box, up=UP)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.6), dpi=170)
    titles = [
        f"the scan the model works from\n9.362 um, scored on an "
        f"{UM:g} um grid",
        "the model's output (green)\nand what the check scored missed (red)",
        f"the second scan of the same scroll\n{native} um native, shown at "
        f"{shown} um, same frame"]
    for ax, img, t, ppu in zip(axes, (a, b, c), titles, (UM, UM, UM / UP)):
        ax.imshow(img, interpolation="nearest")
        bar(ax, img.shape[1], img.shape[0], ppu)
        clean(ax)
        ax.set_title(t, fontsize=10.5)
    oy, ox = SC[scroll]["origin"][1:]
    err = {"0139": 4.09, "1203": 2.38}[scroll]
    fig.text(0.5, -0.045,
             f"PHerc{scroll}, z={gz} y={cy+oy} x={cx+ox}. The right panel is "
             f"not a drawing and not a model output. It is the same object "
             f"scanned again at {native} um and put into the left panel's "
             f"frame\nby the registration published with the labels, which "
             f"lands within {err} um on points held out of the fit. The check "
             f"scores against sheets found in the right panel, at "
             f"tolerances of 19 to 56 um.", ha="center", fontsize=9.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(HERE / f"{out}.png", bbox_inches="tight", pad_inches=0.05)
    print("wrote", HERE / f"{out}.png", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], *(int(v) for v in sys.argv[2:6]), sys.argv[6])
