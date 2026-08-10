#!/usr/bin/env python3
"""Zone calibration, part 1 (cheap instruments):

A. Threshold x boundary scan: on 3 hi_L3 slices, how do material-voxel zone
   shares move across TH in {41..71} and mf boundaries; where does the
   'fused' share stabilize?
B. Per-zone registration residual: bin the pass3 block-match displacements
   by the material fraction at each block center. If matching degrades in
   dense zones, label casting there inherits that error.
C. Gap-visibility probe: within each zone, the fraction of material voxels
   whose distance to the nearest air voxel is < 3 vox (57 um) - a direct
   physical measure of 'how far is the nearest resolvable boundary',
   threshold-dependent but boundary-shape independent.
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

import sys
sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
from pass1 import read_zrange

BASE = Path("/root/fl_probe/reg1203")
HI3 = BASE / "hi_L3"
TILE = 64

out = {}

# ---- A + C
for zc in (448, 832, 1344):
    sl = read_zrange(HI3, zc, zc + 1)[0]
    row = {}
    for th in (41, 48, 56, 64, 71):
        tb = sl > th
        mf = ndi.uniform_filter(tb.astype(np.float32), TILE)
        zone = np.where(mf >= 0.80, 2, np.where(mf >= 0.45, 1, 0))
        zm = zone[tb]
        shares = [round(float((zm == k).mean()), 3) for k in (0, 1, 2)]
        # gap visibility: distance from material voxel to nearest air
        dair = ndi.distance_transform_edt(tb)
        gv = [round(float((dair[tb & (zone == k)] < 3).mean()), 3)
              if (tb & (zone == k)).sum() > 1000 else None
              for k in (0, 1, 2)]
        row[th] = dict(shares_LCF=shares, gapvis_LCF=gv)
    out[f"z{zc}"] = row
    print(zc, json.dumps(row[56]), flush=True)

# ---- B: per-zone registration residual from pass3 field
f = np.load(BASE / "reg" / "pass3_final.npz")
P, D, NC = f["P"], f["D"], f["NCC"]
# material fraction at block centers, from a matching hi_L3 slice per block z
# use the L2-frame hiT proxy: reuse truth at TH=56 on nearest surveyed slice
res = {}
sl_cache = {}
lo1_to_hi3 = None
# cheap proxy: block z (L2) -> hi_L3 z ~ (z_L2*2*18.724/19.226) adjusted; use
# the mf of the nearest of our three slices by scroll-z fraction
zL2 = P[:, 0]
zfrac = (zL2 - zL2.min()) / max(zL2.max() - zL2.min(), 1)
pick = np.clip((zfrac * 3).astype(int), 0, 2)
slices = [448, 832, 1344]
dn = np.linalg.norm(D - D.mean(0), axis=1)
for k, zc in enumerate(slices):
    sl = read_zrange(HI3, zc, zc + 1)[0]
    tb = sl > 56
    mf = ndi.uniform_filter(tb.astype(np.float32), TILE)
    sel = pick == k
    # map block yx (L2) -> hi_L3 yx: scale ~ 37.448/19.226 within hi frame
    # after removing the offset; approximate via linear rescale of extents
    yy = np.clip(((P[sel, 1] - P[:, 1].min())
                  / max(P[:, 1].max() - P[:, 1].min(), 1)
                  * (mf.shape[0] - 1)).astype(int), 0, mf.shape[0] - 1)
    xx = np.clip(((P[sel, 2] - P[:, 2].min())
                  / max(P[:, 2].max() - P[:, 2].min(), 1)
                  * (mf.shape[1] - 1)).astype(int), 0, mf.shape[1] - 1)
    z = np.where(mf[yy, xx] >= 0.80, 2, np.where(mf[yy, xx] >= 0.45, 1, 0))
    for zi, zn in enumerate("LCF"):
        m = z == zi
        if m.sum() > 50:
            key = f"zone_{zn}"
            d = res.setdefault(key, [])
            d.extend((dn[sel][m] * 37.448).tolist())
regres = {k: dict(n=len(v), med_um=round(float(np.median(v)), 2),
                  p95_um=round(float(np.percentile(v, 95)), 2),
                  ncc_med=None)
          for k, v in res.items()}
out["per_zone_registration_residual"] = regres
print(json.dumps(regres, indent=1), flush=True)
json.dump(out, open(BASE / "reg" / "calib1.json", "w"), indent=1)
