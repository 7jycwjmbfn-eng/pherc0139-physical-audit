#!/usr/bin/env python3
"""Resample the second (fine) scan into the same frame as a working-grid crop.

The registration is the one shipped in results/pass3_final.npz:
    p_L1 = M2 @ p_L4 + off,   off = 2*t2 + 0.5
so going the other way, and dropping to the finer level L2 (4.516 um on
PHerc0139, four times the working grid):
    p_L2 = 4 * M2inv @ (p_L1 - off) + 1.5

Nothing here re-fits anything. It reads the published transform and applies it.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_util import SCROLLS, read_box            # noqa: E402

# prefix, the pyramid level the published transform is written against, the
# level to display, and that level's voxel size. The two scrolls differ:
# PHerc0139's M2 maps hi L4 -> lo L1, PHerc1203's maps hi L3 -> lo L1
# (pass5b_1203.py lines 135-137). Getting this wrong misplaces the overlay.
HI = {
    "0139": dict(prefix="PHerc0139/volumes/"
                 "20260413113053-1.129um-0.2m-59keV-masked.zarr",
                 base=4, show=2, um=4.516, native=1.129,
                 shape=(4825, 8993, 8080)),
    "1203": dict(prefix="PHerc1203/volumes/"
                 "20260319130212-2.403um-0.2m-77keV-masked.zarr",
                 base=3, show=1, um=4.806, native=2.403,
                 shape=(7569, 13247, 13247)),
}
NPZ = {"0139": "results/pass3_final.npz",
       "1203": "results/1203/pass3_final.npz"}
UP = 4                     # output pixels per working-grid voxel


def transform(scroll):
    f = np.load(NPZ[scroll])
    M2, t2 = f["M2"], f["t2"]
    return np.linalg.inv(M2), 2.0 * t2 + 0.5


def fine_crop(scroll, gz, y0, y1, x0, x1, level=None, verbose=True):
    """y0..x1 are label-local working-grid indices, same convention as
    render2.get_ct. The registration lives in the volume's global frame, so
    the origin is added here rather than by the caller."""
    _, oy, ox = SCROLLS[scroll]["origin"]
    y0, y1, x0, x1 = y0 + oy, y1 + oy, x0 + ox, x1 + ox
    Minv, off = transform(scroll)
    h = HI[scroll]
    prefix, shape = h["prefix"], h["shape"]
    level = h["show"] if level is None else level
    s = 2 ** (h["base"] - level)         # transform level -> displayed level
    H, W = y1 - y0, x1 - x0

    # map the output corners into the fine level to size the fetch
    corners = []
    for dz in (-1, 1):
        for yy in (y0, y1):
            for xx in (x0, x1):
                p1 = np.array([gz + dz, yy, xx], float)
                corners.append(s * (Minv @ (p1 - off)) + (s - 1) / 2.0)
    c = np.array(corners)
    z0, y0f, x0f = np.floor(c.min(0)).astype(int) - 2
    z1, y1f, x1f = np.ceil(c.max(0)).astype(int) + 3
    z0, y0f, x0f = max(z0, 0), max(y0f, 0), max(x0f, 0)
    if shape:
        z1, y1f, x1f = (min(z1, shape[0]), min(y1f, shape[1]),
                        min(x1f, shape[2]))
    if verbose:
        print(f"  fine L{level} box z{z0}:{z1} y{y0f}:{y1f} x{x0f}:{x1f}",
              flush=True)
    vol = read_box(prefix, level, 128, False, z0, z1, y0f, y1f, x0f, x1f,
                   verbose=verbose)

    # output index o = (k, i, j) at UP samples per working voxel
    #   p_L1 = [gz, y0, x0] + o / UP
    #   p_Lx = s * Minv @ (p_L1 - off) + (s-1)/2
    A = (s / UP) * Minv
    base = np.array([gz, y0, x0], float) - off
    cvec = s * (Minv @ base) + (s - 1) / 2.0 - np.array([z0, y0f, x0f], float)
    out = ndi.affine_transform(vol.astype(np.float32), A, offset=cvec,
                               output_shape=(1, H * UP, W * UP), order=1)
    return out[0]


if __name__ == "__main__":
    scroll, gz, cy, cx, half = (sys.argv[1], *(int(v) for v in sys.argv[2:6]))
    a = fine_crop(scroll, gz, cy - half, cy + half, cx - half, cx + half)
    print("fine crop", a.shape, "mean", float(a.mean()),
          "nonzero", float((a > 0).mean()))
    np.save(Path(__file__).resolve().parent / "fine_test.npy", a)
