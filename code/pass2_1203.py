#!/usr/bin/env python3
"""PHerc1203 block-matching refinement, reusing the proven pass2 machinery.

Forward map (theta=0 makes it pure scale+shift):
  p_L2 = 1.0267050 * p_hiL4 + (1986.53, 5.7, -3.5)
derived from the pass1 lock (zoff 74.4 mm, dy 2.6, dx -2.0 on the 74.896 um
grid) and voxel sizes hi_L4 = 38.448 um, lo_L2 = 37.448 um.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

import sys
sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
import pass2
from pass1 import read_zrange, zmeta

BASE = Path("/root/fl_probe/reg1203")
OUT = BASE / "reg"
LO2_VOX = 37.448
HI4_VOX = 2.403 * 16                      # 38.448
Z0, Z1 = 1920, 3072                       # downloaded lo_L2 window
XY_MAX = 1711

R = HI4_VOX / LO2_VOX
T = np.array([74.4 * 1000.0 / LO2_VOX, 2 * 2.6 + 0.5, 2 * -2.0 + 0.5])


def main():
    t0 = time.time()
    print("loading lo_L2 window...", flush=True)
    lo = read_zrange(BASE / "lo_L2", Z0, Z1)
    print("loading hi_L4...", flush=True)
    hi = read_zrange(BASE / "hi_L4", 0, zmeta(BASE / "hi_L4")["shape"][0])

    M = np.eye(3) * R
    corners = np.array([[z, y, x] for z in (0, hi.shape[0])
                        for y in (0, hi.shape[1])
                        for x in (0, hi.shape[2])], float)
    cl2 = (M @ corners.T).T + T
    b0 = np.maximum(np.floor(cl2.min(0)).astype(int) - 2, [Z0, 0, 0])
    b1 = np.minimum(np.ceil(cl2.max(0)).astype(int) + 2,
                    [Z1, XY_MAX, XY_MAX])
    shape = tuple((b1 - b0).astype(int))
    Minv = np.linalg.inv(M)
    off = Minv @ (b0 - T)
    print(f"resampling hi into L2 box {b0}..{b1} {shape}", flush=True)
    # stay uint8 end to end: the box is ~2.8G voxels and float32 would need
    # >20 GB across input+output; u1 rounding is irrelevant to block NCC
    hiT = ndi.affine_transform(hi, Minv, offset=off,
                               output_shape=shape, order=1)
    msk = ndi.affine_transform((hi > 0).astype(np.uint8) * 255, Minv,
                               offset=off, output_shape=shape,
                               order=1) > 180
    del hi

    pass2.HIT, pass2.MSK, pass2.LO, pass2.BOX0 = hiT, msk, lo, b0
    pass2.LO2_Z0 = Z0
    pass2.SRCH = 14        # pass1 z grid is 0.8 mm = 21 L2 vox; widen margin
    jobs = []
    for bz in range(0, shape[0] - pass2.BLK, pass2.STRIDE):
        for by in range(0, shape[1] - pass2.BLK, pass2.STRIDE):
            for bx in range(0, shape[2] - pass2.BLK, pass2.STRIDE):
                jobs.append((bz, by, bx))
    print(f"{len(jobs)} candidate blocks", flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for r in ex.map(pass2.match_block, jobs, chunksize=16):
            if r is not None:
                res.append(r)
    res = np.array(res, np.float64)
    print(f"{len(res)} matched blocks in {time.time()-t0:.0f}s", flush=True)

    P, D, NC = res[:, :3], res[:, 3:6], res[:, 6]
    ctr = P.mean(0)
    sol, keep = pass2.robust_fit(P - ctr, D)
    r_ = D - pass2.predict(P - ctr, sol)
    rn = np.linalg.norm(r_[keep], axis=1)
    stats = dict(
        n_blocks=int(len(res)), n_kept=int(keep.sum()),
        ncc_median=float(np.median(NC)),
        disp_median_vox=float(np.median(np.linalg.norm(D, axis=1))),
        resid_median_um=float(np.median(rn) * LO2_VOX),
        resid_p95_um=float(np.percentile(rn, 95) * LO2_VOX),
        sol=dict(tz=float(sol[0]), ty=float(sol[1]), tx=float(sol[2]),
                 rz_rad=float(sol[3]), ry_rad=float(sol[4]),
                 rx_rad=float(sol[5]), scale=float(sol[6])))
    print(json.dumps(stats, indent=1), flush=True)
    np.savez_compressed(OUT / "pass2_field.npz", P=P, D=D, NCC=NC,
                        keep=keep, sol=sol, ctr=ctr, M=M, t=T)
    json.dump(stats, open(OUT / "pass2_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
