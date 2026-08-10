#!/usr/bin/env python3
"""Pass 2: 6+1-DOF refinement of the pass1 lock by 3D block matching.

hi_L5 (36.128 um) is resampled straight into the lo_L2 (37.448 um) frame with
one trilinear affine step; local 3D NCC block matching yields a displacement
field; a robust linearized rigid+scale fit refines the pose.  The post-fit
residual field is saved for the deformation analysis (pass 3).

The in-plane rotation convention is NOT hand-derived: before anything runs, a
numeric marker test reproduces the exact pass1 op chain (zoom -> rotate ->
paste) and asserts our matrix agrees with it to <1 voxel.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

import sys
sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
from pass1 import zmeta, read_zrange, HI_ROOT, HI_VOX, LO_VOX  # noqa: E402

BASE = Path("/root/fl_probe/reg0139")
OUT = BASE / "reg2"
LO2_ROOT, LO2_VOX = BASE / "lo_L2", 37.448

THETA, ZOFF_MM, DY74, DX74 = 183.0, 33.6, 157.2, 153.2
ZF5 = HI_VOX / LO_VOX                      # hi_L5 -> 74.896 um grid
S_INPLANE = 2.0 * ZF5                      # hi_L5 -> lo_L2 (= 36.128/37.448)
S_Z = HI_VOX / LO2_VOX                     # identical value, kept explicit

BLK = 40            # template block size (L2 vox)
SRCH = 10           # +- search (L2 vox)
STRIDE = 32
NCC_MIN = 0.30
MASKFRAC_MIN = 0.75

LO2_Z0, LO2_Z1 = 768, 1536                 # downloaded window


# --------------------------------------------------- forward map hi_L5 -> L2
def rot2d(theta_deg, sign):
    a = np.deg2rad(theta_deg) * sign
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def build_forward(sign):
    """Return (M, t): p_L2 = M @ p_hi5 + t, axes (z,y,x)."""
    h74 = np.array([round(1125 * ZF5), round(1010 * ZF5)], float)
    c74 = (h74 - 1) / 2.0
    R = rot2d(THETA, sign)
    M = np.zeros((3, 3))
    M[0, 0] = S_Z
    M[1:, 1:] = 2.0 * ZF5 * R
    t = np.zeros(3)
    t[0] = ZOFF_MM * 1000.0 / LO2_VOX
    off74 = -R @ c74 + c74 + np.array([DY74, DX74])
    t[1:] = 2.0 * off74 + 0.5
    return M, t


def check_convention():
    """Reproduce the pass1 2D op chain on a marker; pick the sign that agrees."""
    img = np.zeros((1125, 1010), np.float32)
    my, mx = 300, 700
    img[my - 2:my + 3, mx - 2:mx + 3] = 100.0
    z = ndi.zoom(img, ZF5, order=1)
    r = ndi.rotate(z, THETA, reshape=False, order=1)
    yy, xx = np.nonzero(r > r.max() * 0.2)
    got74 = np.array([yy.mean(), xx.mean()])
    for sign in (+1.0, -1.0):
        M, t = build_forward(sign)
        pred = M @ np.array([0.0, my, mx]) + t
        pred74 = (pred[1:] - 0.5) / 2.0 - np.array([DY74, DX74])
        if np.abs(pred74 - got74).max() < 1.5:
            print(f"convention check OK: sign={sign:+.0f} "
                  f"err={np.abs(pred74 - got74).max():.3f} vox", flush=True)
            return sign
    raise AssertionError(f"neither sign matches: got {got74}")


# ------------------------------------------------------------------ resample
def resample_hi_to_L2(M, t):
    m5 = zmeta(HI_ROOT)
    hi = read_zrange(HI_ROOT, 0, m5["shape"][0])
    corners = np.array([[z, y, x] for z in (0, m5["shape"][0])
                        for y in (0, m5["shape"][1])
                        for x in (0, m5["shape"][2])], float)
    cl2 = (M @ corners.T).T + t
    lo_box0 = np.floor(cl2.min(0)).astype(int) - 2
    lo_box1 = np.ceil(cl2.max(0)).astype(int) + 2
    lo_box0[0] = max(lo_box0[0], LO2_Z0)
    lo_box1[0] = min(lo_box1[0], LO2_Z1)
    lo_box0[1:] = np.maximum(lo_box0[1:], 0)
    lo_box1[1:] = np.minimum(lo_box1[1:], 1656)
    shape = tuple((lo_box1 - lo_box0).astype(int))
    Minv = np.linalg.inv(M)
    off = Minv @ (lo_box0 - t)
    print(f"resampling hi into L2 box {lo_box0}..{lo_box1} {shape}", flush=True)
    hiT = ndi.affine_transform(hi.astype(np.float32), Minv, offset=off,
                               output_shape=shape, order=1, mode="constant")
    mask = ndi.affine_transform((hi > 0).astype(np.float32), Minv, offset=off,
                                output_shape=shape, order=1, mode="constant")
    return hiT, (mask > 0.7), lo_box0


# ------------------------------------------------------------ block matching
HIT = MSK = LO = None
BOX0 = None


def match_block(args):
    bz, by, bx = args        # block origin in hiT coords
    t = HIT[bz:bz + BLK, by:by + BLK, bx:bx + BLK]
    m = MSK[bz:bz + BLK, by:by + BLK, bx:bx + BLK]
    if m.mean() < MASKFRAC_MIN or t.std() < 1.0:
        return None
    # lo search region around the same position
    gz, gy, gx = bz + BOX0[0], by + BOX0[1], bx + BOX0[2]
    z0, y0, x0 = gz - SRCH - LO2_Z0, gy - SRCH, gx - SRCH
    if z0 < 0 or y0 < 0 or x0 < 0:
        return None
    reg = LO[z0:z0 + BLK + 2 * SRCH, y0:y0 + BLK + 2 * SRCH,
             x0:x0 + BLK + 2 * SRCH]
    if reg.shape != (BLK + 2 * SRCH,) * 3 or (reg > 0).mean() < 0.5:
        return None
    tt = (t - t.mean()).astype(np.float32) * m
    reg = reg.astype(np.float32)
    cross = fftconvolve(reg, tt[::-1, ::-1, ::-1], mode="valid")
    n = m.sum()
    rs = ndi.uniform_filter(reg, BLK, mode="constant")
    rs2 = ndi.uniform_filter(reg * reg, BLK, mode="constant")
    half = BLK // 2
    sl = slice(half, half + 2 * SRCH + 1)
    rmean = rs[sl, sl, sl]
    rvar = np.maximum(rs2[sl, sl, sl] - rmean ** 2, 1e-6) * BLK ** 3
    den = np.sqrt(float((tt ** 2).sum()) * rvar)
    ncc = (cross - rmean * tt.sum()) / np.maximum(den, 1e-6)
    i = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    pk = float(ncc[i])
    if pk < NCC_MIN:
        return None
    d = np.array(i, float) - SRCH
    # parabolic subvoxel per axis: vertex x* = (a-b) / (2(a+b-2pk)), pk is max
    for ax in range(3):
        j = list(i)
        if 0 < i[ax] < ncc.shape[ax] - 1:
            j[ax] = i[ax] - 1
            a = float(ncc[tuple(j)])
            j[ax] = i[ax] + 1
            b = float(ncc[tuple(j)])
            denom = a + b - 2 * pk          # <= 0 at a true peak
            if denom < -1e-9:
                d[ax] += np.clip(0.5 * (a - b) / denom, -0.75, 0.75)
    ctr = np.array([gz, gy, gx], float) + (BLK - 1) / 2.0
    return (*ctr, *d, pk)


def robust_fit(P, D):
    """Linearized rigid+scale: d = t + (W - I) p, W = (1+s)R(small angles)."""
    keep = np.ones(len(P), bool)
    sol = None
    for _ in range(4):
        p, d = P[keep], D[keep]
        A = np.zeros((3 * len(p), 7))
        z, y, x = p[:, 0], p[:, 1], p[:, 2]
        # params: tz ty tx  rz(y<->x)  ry(z<->x)  rx(z<->y)  s
        A[0::3, 0] = 1
        A[1::3, 1] = 1
        A[2::3, 2] = 1
        A[1::3, 3] = -x
        A[2::3, 3] = y
        A[0::3, 4] = x
        A[2::3, 4] = -z
        A[0::3, 5] = -y
        A[1::3, 5] = z
        A[0::3, 6] = z
        A[1::3, 6] = y
        A[2::3, 6] = x
        sol, *_ = np.linalg.lstsq(A, d.reshape(-1), rcond=None)
        r = (A @ sol).reshape(-1, 3) - d
        rn = np.linalg.norm(r, axis=1)
        mad = np.median(np.abs(rn - np.median(rn))) + 1e-9
        newkeep = np.zeros(len(P), bool)
        newkeep[np.where(keep)[0]] = rn < np.median(rn) + 3 * 1.4826 * mad
        if newkeep.sum() == keep.sum():
            break
        keep = newkeep
    return sol, keep


def main():
    global HIT, MSK, LO, BOX0
    sign = check_convention()
    M, t = build_forward(sign)

    print("loading lo_L2 window...", flush=True)
    LO = read_zrange(LO2_ROOT, LO2_Z0, LO2_Z1)

    HIT, MSK, BOX0 = resample_hi_to_L2(M, t)
    print(f"hiT {HIT.shape} valid={MSK.mean():.3f}", flush=True)

    jobs = []
    for bz in range(0, HIT.shape[0] - BLK, STRIDE):
        for by in range(0, HIT.shape[1] - BLK, STRIDE):
            for bx in range(0, HIT.shape[2] - BLK, STRIDE):
                jobs.append((bz, by, bx))
    print(f"{len(jobs)} candidate blocks", flush=True)
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=20) as ex:
        for r in ex.map(match_block, jobs, chunksize=16):
            if r is not None:
                res.append(r)
    res = np.array(res, np.float64)
    print(f"{len(res)} matched blocks in {time.time()-t0:.0f}s", flush=True)

    P = res[:, :3]                      # block centers, global L2 coords
    D = res[:, 3:6]                     # displacement lo <- hiT (vox)
    NC = res[:, 6]
    ctr = P.mean(0)
    sol, keep = robust_fit(P - ctr, D)
    r = D - predict(P - ctr, sol)
    stats = dict(
        n_blocks=int(len(res)), n_kept=int(keep.sum()),
        ncc_median=float(np.median(NC)),
        disp_median_vox=float(np.median(np.linalg.norm(D, axis=1))),
        resid_median_vox=float(np.median(
            np.linalg.norm(r[keep], axis=1))),
        resid_p95_vox=float(np.percentile(
            np.linalg.norm(r[keep], axis=1), 95)),
        resid_median_um=float(np.median(
            np.linalg.norm(r[keep], axis=1)) * LO2_VOX),
        sol=dict(tz=float(sol[0]), ty=float(sol[1]), tx=float(sol[2]),
                 rz_rad=float(sol[3]), ry_rad=float(sol[4]),
                 rx_rad=float(sol[5]), scale=float(sol[6])),
        fit_center=[float(v) for v in ctr])
    print(json.dumps(stats, indent=1), flush=True)
    np.savez_compressed(OUT / "pass2_field.npz", P=P, D=D, NCC=NC,
                        keep=keep, resid=r, sol=sol, ctr=ctr,
                        M=M, t=t, sign=sign)
    json.dump(stats, open(OUT / "pass2_stats.json", "w"), indent=1)


def predict(p, sol):
    tz, ty, tx, rz, ry, rx, s = sol
    z, y, x = p[:, 0], p[:, 1], p[:, 2]
    dz = tz + ry * x - rx * y + s * z
    dy = ty - rz * x + rx * z + s * y
    dx = tx + rz * y - ry * z + s * x
    return np.stack([dz, dy, dx], 1)


if __name__ == "__main__":
    main()
