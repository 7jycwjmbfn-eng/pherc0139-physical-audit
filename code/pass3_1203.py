#!/usr/bin/env python3
"""PHerc1203 pass 3: rematch under corrected pose, split-half CV, field."""
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
Z0, Z1 = 1920, 3072
XY_MAX = 1711
LAT = 96


def skew(sol):
    tz, ty, tx, rz, ry, rx, s = sol
    return (np.array([[s, -rx, ry], [rx, s, -rz], [-ry, rz, s]]),
            np.array([tz, ty, tx]))


def main():
    t0 = time.time()
    f = np.load(OUT / "pass2_field.npz")
    sol, ctr, M, T = f["sol"], f["ctr"], f["M"], f["t"]
    Wm, tsol = skew(sol)
    W = np.eye(3) + Wm
    M2 = W @ M
    t2 = W @ T + tsol - Wm @ ctr

    print("loading volumes...", flush=True)
    lo = read_zrange(BASE / "lo_L2", Z0, Z1)
    hi = read_zrange(BASE / "hi_L4", 0, zmeta(BASE / "hi_L4")["shape"][0])
    corners = np.array([[z, y, x] for z in (0, hi.shape[0])
                        for y in (0, hi.shape[1])
                        for x in (0, hi.shape[2])], float)
    cl2 = (M2 @ corners.T).T + t2
    b0 = np.maximum(np.floor(cl2.min(0)).astype(int) - 2, [Z0, 0, 0])
    b1 = np.minimum(np.ceil(cl2.max(0)).astype(int) + 2,
                    [Z1, XY_MAX, XY_MAX])
    shape = tuple((b1 - b0).astype(int))
    Minv = np.linalg.inv(M2)
    off = Minv @ (b0 - t2)
    print(f"resampling {shape}", flush=True)
    hiT = ndi.affine_transform(hi, Minv, offset=off, output_shape=shape,
                               order=1)
    msk = ndi.affine_transform((hi > 0).astype(np.uint8) * 255, Minv,
                               offset=off, output_shape=shape,
                               order=1) > 180
    del hi
    pass2.HIT, pass2.MSK, pass2.LO, pass2.BOX0 = hiT, msk, lo, b0
    pass2.LO2_Z0 = Z0
    pass2.SRCH = 10
    jobs = []
    for bz in range(0, shape[0] - pass2.BLK, pass2.STRIDE):
        for by in range(0, shape[1] - pass2.BLK, pass2.STRIDE):
            for bx in range(0, shape[2] - pass2.BLK, pass2.STRIDE):
                jobs.append((bz, by, bx))
    res = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for r in ex.map(pass2.match_block, jobs, chunksize=16):
            if r is not None:
                res.append(r)
    res = np.array(res)
    P, D, NC = res[:, :3], res[:, 3:6], res[:, 6]
    dn = np.linalg.norm(D, axis=1)
    print(f"rematch {len(res)} blocks ncc={np.median(NC):.3f} "
          f"|d| med={np.median(dn):.3f} t={time.time()-t0:.0f}s", flush=True)
    sol2, _ = pass2.robust_fit(P - P.mean(0), D)
    print("residual rigid (should be ~0):",
          np.round(sol2, 5).tolist(), flush=True)

    cell = np.floor(P / LAT).astype(int)
    cmin = cell.min(0)
    cidx = cell - cmin
    csh = cidx.max(0) + 1

    def build(maskrows):
        s_ = np.zeros((*csh, 3))
        w_ = np.zeros(csh)
        for i in np.where(maskrows)[0]:
            c = tuple(cidx[i])
            s_[c] += D[i] * NC[i]
            w_[c] += NC[i]
        lat = np.full((*csh, 3), np.nan)
        ok = w_ > 1.0
        lat[ok] = s_[ok] / w_[ok, None]
        return lat, ok

    def interp(lat, ok, pts):
        filled = lat.copy()
        if (~ok).any():
            ind = ndi.distance_transform_edt(~ok, return_distances=False,
                                             return_indices=True)
            filled = lat[tuple(ind)]
        g = pts / LAT - cmin - 0.5
        out = np.empty((len(pts), 3))
        from scipy.ndimage import map_coordinates
        for a in range(3):
            out[:, a] = map_coordinates(filled[..., a], g.T, order=1,
                                        mode="nearest")
        return out

    zc = cidx[:, 0]
    A, B = (zc % 2 == 0), (zc % 2 == 1)
    latA, okA = build(A)
    pred = interp(latA, okA, P[B])
    err = np.linalg.norm(D[B] - pred, axis=1)
    base = np.linalg.norm(D[B], axis=1)
    latF, okF = build(np.ones(len(P), bool))
    stats = dict(
        n_blocks=int(len(res)),
        ncc_med=float(np.median(NC)),
        rigid_only_med_um=float(np.median(base) * LO2_VOX),
        heldout_med_um=float(np.median(err) * LO2_VOX),
        heldout_p95_um=float(np.percentile(err, 95) * LO2_VOX),
        residual_rigid=np.round(sol2, 6).tolist())
    print(json.dumps(stats, indent=1), flush=True)
    np.savez_compressed(OUT / "pass3_final.npz", P=P, D=D, NCC=NC,
                        M2=M2, t2=t2, lattice=latF, lattice_ok=okF,
                        lat_spacing=LAT, lat_cmin=cmin, box0=b0)
    json.dump(stats, open(OUT / "pass3_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
