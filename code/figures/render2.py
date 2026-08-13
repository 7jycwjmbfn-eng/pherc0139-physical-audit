#!/usr/bin/env python3
"""Shared pieces for the figure scripts: recover each scored stretch with its
own centerline pixels, and read a working-grid crop of the coarse scan.

Not an entry point. See render4.py and render5.py.
"""
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_util import SCROLLS, read_box     # noqa: E402

UM = 18.724
HERE = Path(os.environ.get("SLICES",
                           Path(__file__).resolve().parent))
GREEN = np.array([0.15, 0.95, 0.25])
RED = np.array([1.0, 0.15, 0.15])


def _dist(band):
    if not band.any():
        return np.full(band.shape, np.inf, np.float32)
    return ndi.distance_transform_edt(~band).astype(np.float32)


def arcs_with_pixels(gz, scroll="0139"):
    d = np.load(HERE / "slices" / f"{scroll}_z{gz}.npz", allow_pickle=True)
    lab, pred = d["lab"], d["pred"]
    valid = (lab & 1) > 0
    ctr = (lab & 4) > 0
    pb = pred & valid
    dpred = _dist(pb)
    ys, xs = np.nonzero(ctr)
    cd = dpred[ys, xs]
    lab2, _ = ndi.label(ctr, structure=np.ones((3, 3), int))
    lv = lab2[ys, xs]
    aid = (lv.astype(np.int64) * 10_000_000
           + (ys // 64).astype(np.int64) * 2000 + xs // 64)
    _, inv = np.unique(aid, return_inverse=True)
    cnt = np.bincount(inv)
    cov = np.bincount(inv, weights=(cd <= 2)) / cnt
    out = []
    for i in np.where(cnt >= 20)[0]:
        m = inv == i
        out.append(dict(n=int(cnt[i]), cov=float(cov[i]),
                        ys=ys[m], xs=xs[m],
                        cy=float(ys[m].mean()), cx=float(xs[m].mean()),
                        gone=bool(cov[i] < 0.1), hit=bool(cov[i] >= 0.5)))
    return out, lab, pb


def paint(ct, pb, arcs, y0, y1, x0, x1, red=True):
    v = ct[ct > 0]
    lo, hi = np.percentile(v, (1, 99.5)) if v.size else (0, 1)
    g = np.clip((ct.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    rgb = np.dstack([g, g, g])
    sub = pb[y0:y1, x0:x1]
    rgb[sub] = 0.5 * rgb[sub] + 0.5 * GREEN
    if red:
        for a in arcs:
            if not a["gone"]:
                continue
            m = ((a["ys"] >= y0) & (a["ys"] < y1)
                 & (a["xs"] >= x0) & (a["xs"] < x1))
            if not m.any():
                continue
            yy, xx = a["ys"][m] - y0, a["xs"][m] - x0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ry = np.clip(yy + dy, 0, y1 - y0 - 1)
                    rx = np.clip(xx + dx, 0, x1 - x0 - 1)
                    rgb[ry, rx] = RED
    return rgb


def get_ct(gz, y0, y1, x0, x1, scroll="0139"):
    cfg = SCROLLS[scroll]
    _, oy, ox = cfg["origin"]
    _, ly, lx = cfg["lo_l1"]
    # the label box can reach past the volume (1203 is 3456 wide, the volume
    # 3422); clamp so the request stays inside and pad the rest with zeros
    a0, a1 = oy + y0, min(oy + y1, ly)
    b0, b1 = ox + x0, min(ox + x1, lx)
    out = np.zeros((y1 - y0, x1 - x0), np.uint8)
    if a1 > a0 and b1 > b0:
        out[:a1 - a0, :b1 - b0] = read_box(
            cfg["lo"], 1, 128, False, gz, gz + 1, a0, a1, b0, b1)[0]
    return out


def scalebar(ax, h, w, um_len=500):
    n = um_len / UM
    ax.plot([w * 0.04, w * 0.04 + n], [h * 0.955] * 2, "-", c="w", lw=3)
    ax.text(w * 0.04, h * 0.935, f"{um_len/1000:g} mm", color="w",
            fontsize=9, va="bottom")


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
