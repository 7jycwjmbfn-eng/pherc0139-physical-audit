#!/usr/bin/env python3
"""QC overlays: hi-truth (grayscale) + m7 prediction (red) + centerline (green).

Confirms visually what m7 predicts (thin band vs full material) and produces
report figures. One slab, three crops.
"""
import numpy as np
from pathlib import Path
from scipy import ndimage as ndi
from PIL import Image

import sys
sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
from pass5b import (load_pred_slab, maxpool2, Y1_0, X1_0, ERODE, HI4, OUT,
                    read_zrange, zmeta)

IZ = 24
TH = 65

f3 = np.load(OUT / "pass3_final.npz")
M2, t2 = f3["M2"], f3["t2"]
Minv = np.linalg.inv(M2)
off = 2.0 * t2 + 0.5

m4 = zmeta(HI4)
hi = read_zrange(HI4, 0, m4["shape"][0])
slab, _ = load_pred_slab(IZ)
pred = maxpool2(slab) > 0
del slab
origin = np.array([IZ * 96, Y1_0, X1_0], float)
hiT = ndi.affine_transform(hi, Minv, offset=Minv @ (origin - off),
                           output_shape=pred.shape, order=1)
k = 48
tb = (hiT[k] > TH)
valid = ndi.binary_erosion(hiT[k] > 0, iterations=ERODE)
tb &= valid
pb = pred[k] & valid
dt = ndi.distance_transform_edt(tb)
ctr = tb & (dt >= ndi.maximum_filter(dt, 3)) & (dt >= 1)

g = np.clip(hiT[k].astype(np.float32) / 170, 0, 1)
for name, (y0, x0) in dict(center=(1000, 900), east=(900, 1600),
                           south=(1700, 900)).items():
    cy, cx = slice(y0, y0 + 500), slice(x0, x0 + 500)
    G = g[cy, cx]
    P = pb[cy, cx]
    C = ctr[cy, cx]
    rgb = np.stack([G * 0.85, G * 0.85, G * 0.85], -1)
    rgb[P] = rgb[P] * 0.4 + np.array([0.9, 0.15, 0.1]) * 0.6
    rgb[C] = [0.1, 1.0, 0.2]
    Image.fromarray((rgb * 255).astype(np.uint8)).resize((1000, 1000),
        Image.NEAREST).save(OUT / f"qc_overlay_{name}.png")
    print(name, "pred frac", float(P.mean()), "ctr px", int(C.sum()))
print("done")
