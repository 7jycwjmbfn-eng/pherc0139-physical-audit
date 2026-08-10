#!/usr/bin/env python3
"""Pass 7: side-of-sheet audit — does m7 predict the RECTO (inward) side?

m7 is documented as recto-only (inward-facing surface).  With model-free
truth we can measure this: at each truth centerline pixel that has a nearby
prediction, sample the distance-to-prediction field at +-2 vox along the
inward radial direction (toward the slice centroid of material).  If the
prediction band sits on the recto side, the inward sample is closer for most
pixels.  Wrong-side fraction = pixels where the outward sample is closer.

Controls: the same statistic under a y-shifted null must sit near 50/50.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

import sys
sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
from pass5b import (load_pred_slab, maxpool2, Y1_0, X1_0, ERODE, HI4, OUT,
                    read_zrange, zmeta, ZCH, NULL_SHIFT)

TH = 65
STEP = 2.0          # vox offset along the sheet normal
COH_MIN = 0.3
HI = MINV = OFFV = None


def normal_field(g):
    """Per-pixel sheet normal from the structure tensor (2D)."""
    gy, gx = ndi.sobel(g, 0), ndi.sobel(g, 1)
    Jyy = ndi.gaussian_filter(gy * gy, 6)
    Jyx = ndi.gaussian_filter(gy * gx, 6)
    Jxx = ndi.gaussian_filter(gx * gx, 6)
    # major eigenvector of [[Jyy,Jyx],[Jyx,Jxx]] = max-gradient dir = normal
    phi = 0.5 * np.arctan2(2 * Jyx, Jyy - Jxx)
    tr, det = Jyy + Jxx, Jyy * Jxx - Jyx * Jyx
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    lam1, lam2 = tr / 2 + disc, tr / 2 - disc
    coher = (lam1 - lam2) / np.maximum(lam1 + lam2, 1e-9)
    return np.cos(phi), np.sin(phi), coher


def _selftest_normal():
    yy, xx = np.mgrid[0:200, 0:200].astype(np.float32)
    for ang in (0.0, 30.0, 77.0, 120.0):
        a = np.deg2rad(ang)
        stripes = np.sin((yy * np.cos(a) + xx * np.sin(a)) * 2.0)
        ny, nx, coh = normal_field(stripes)
        cy, cx = ny[100, 100], nx[100, 100]
        dot = abs(cy * np.cos(a) + cx * np.sin(a))
        assert dot > 0.97, (ang, cy, cx, dot)
        assert coh[100, 100] > 0.9, (ang, coh[100, 100])
    print("normal_field selftest OK", flush=True)


def slab_work(iz):
    slab, _ = load_pred_slab(iz)
    pred = maxpool2(slab) > 0
    del slab
    origin = np.array([iz * 96, Y1_0, X1_0], float)
    aoff = MINV @ (origin - OFFV)
    hiT = ndi.affine_transform(HI, MINV, offset=aoff,
                               output_shape=pred.shape, order=1)
    valid = ndi.binary_erosion(hiT > 0, iterations=ERODE)
    truth = (hiT > TH) & valid
    part = dict(n=0, inward=0, outward=0, tie=0,
                n_null=0, inward_null=0, outward_null=0,
                n_ideal=0, inward_ideal=0, outward_ideal=0,
                off_hist=np.zeros(17, np.int64),
                off_hist_ideal=np.zeros(17, np.int64),
                tile_sum=np.zeros(64 * 64), tile_n=np.zeros(64 * 64))
    for k in range(0, pred.shape[0], 4):
        tb, pb = truth[k], pred[k] & valid[k]
        if tb.sum() < 500 or pb.sum() < 100:
            continue
        dt = ndi.distance_transform_edt(tb)
        ctr = tb & (dt >= ndi.maximum_filter(dt, 3)) & (dt >= 1)
        dpred = ndi.distance_transform_edt(~pb)
        pbs = np.zeros_like(pb)
        pbs[NULL_SHIFT:] = pb[:-NULL_SHIFT]
        dnull = ndi.distance_transform_edt(~pbs)
        ys, xs = np.nonzero(ctr)
        near = dpred[ys, xs] <= 3
        if near.sum() < 100:
            continue
        ys, xs = ys[near], xs[near]
        # sheet normal from the structure tensor, oriented outward via the
        # radial direction (centroid only sets the sign, not the axis)
        nyf, nxf, cohf = normal_field(
            ndi.gaussian_filter(hiT[k].astype(np.float32), 1.5))
        ny, nx = nyf[ys, xs], nxf[ys, xs]
        coh = cohf[ys, xs]
        m = np.nonzero(tb)
        cy, cx = m[0].mean(), m[1].mean()
        ry0, rx0 = ys - cy, xs - cx
        flip = (ny * ry0 + nx * rx0) < 0
        ny = np.where(flip, -ny, ny)
        nx = np.where(flip, -nx, nx)
        ok = coh > COH_MIN
        if ok.sum() < 100:
            continue
        ys, xs, ry, rx = ys[ok], xs[ok], ny[ok], nx[ok]
        # ideal recto band from truth: inward-facing material boundary
        YY, XX = np.mgrid[0:tb.shape[0], 0:tb.shape[1]].astype(np.float32)
        sgn = np.sign(nyf * (YY - cy) + nxf * (XX - cx))
        sgn[sgn == 0] = 1.0
        NYo, NXo = nyf * sgn, nxf * sgn
        tbf = tb.astype(np.float32)
        in_air = map_coordinates(tbf, [YY - 1.5 * NYo, XX - 1.5 * NXo],
                                 order=1) < 0.5
        ideal = ndi.binary_dilation(tb & in_air, iterations=1)
        dideal = ndi.distance_transform_edt(~ideal)

        # signed offset of the nearest band along the outward normal:
        # sample dpred / dideal at offsets -4..+4 (outward positive)
        offs = np.arange(-4.0, 4.01, 0.5)
        cy_s = ys[:, None] + offs[None, :] * ry[:, None]
        cx_s = xs[:, None] + offs[None, :] * rx[:, None]
        prof_p = map_coordinates(dpred, [cy_s.ravel(), cx_s.ravel()],
                                 order=1).reshape(len(ys), len(offs))
        prof_i = map_coordinates(dideal, [cy_s.ravel(), cx_s.ravel()],
                                 order=1).reshape(len(ys), len(offs))
        am_p = np.argmin(prof_p, axis=1)
        am_i = np.argmin(prof_i, axis=1)
        on_p = prof_p[np.arange(len(ys)), am_p] <= 1.0
        on_i = prof_i[np.arange(len(ys)), am_i] <= 1.0
        part["off_hist"] += np.bincount(am_p[on_p], minlength=len(offs))
        part["off_hist_ideal"] += np.bincount(am_i[on_i],
                                              minlength=len(offs))
        # spatial coherence of side choice, 64-vox tiles
        so = offs[am_p]
        side = np.where(so < -0.25, -1, np.where(so > 0.25, 1, 0))
        tile = ((ys // 64).astype(np.int64) * 64 + (xs // 64))[on_p]
        part["tile_sum"] += np.bincount(tile, weights=side[on_p],
                                        minlength=64 * 64)
        part["tile_n"] += np.bincount(tile, minlength=64 * 64)
        din = map_coordinates(dpred, [ys - STEP * ry, xs - STEP * rx],
                              order=1)
        dout = map_coordinates(dpred, [ys + STEP * ry, xs + STEP * rx],
                               order=1)
        part["n"] += len(ys)
        part["inward"] += int((din < dout - 0.25).sum())
        part["outward"] += int((dout < din - 0.25).sum())
        part["tie"] += int((np.abs(din - dout) <= 0.25).sum())
        inear = dideal[ys, xs] <= 3
        if inear.sum() > 100:
            yi, xi = ys[inear], xs[inear]
            ryi, rxi = ry[inear], rx[inear]
            dii = map_coordinates(dideal, [yi - STEP * ryi, xi - STEP * rxi],
                                  order=1)
            dio = map_coordinates(dideal, [yi + STEP * ryi, xi + STEP * rxi],
                                  order=1)
            part["n_ideal"] += len(yi)
            part["inward_ideal"] += int((dii < dio - 0.25).sum())
            part["outward_ideal"] += int((dio < dii - 0.25).sum())
        nnear = dnull[ys, xs] <= 3
        if nnear.sum() > 100:
            yn, xn = ys[nnear], xs[nnear]
            ryn, rxn = ry[nnear], rx[nnear]
            di = map_coordinates(dnull, [yn - STEP * ryn, xn - STEP * rxn],
                                 order=1)
            do = map_coordinates(dnull, [yn + STEP * ryn, xn + STEP * rxn],
                                 order=1)
            part["n_null"] += len(yn)
            part["inward_null"] += int((di < do - 0.25).sum())
            part["outward_null"] += int((do < di - 0.25).sum())
    return part


def main():
    global HI, MINV, OFFV
    _selftest_normal()
    t0 = time.time()
    f3 = np.load(OUT / "pass3_final.npz")
    M2, t2 = f3["M2"], f3["t2"]
    MINV = np.linalg.inv(M2)
    OFFV = 2.0 * t2 + 0.5
    m4 = zmeta(HI4)
    HI = read_zrange(HI4, 0, m4["shape"][0])
    print(f"hi loaded t={time.time()-t0:.0f}s", flush=True)
    acc = dict(n=0, inward=0, outward=0, tie=0,
               n_null=0, inward_null=0, outward_null=0,
               n_ideal=0, inward_ideal=0, outward_ideal=0,
               off_hist=np.zeros(17, np.int64),
               off_hist_ideal=np.zeros(17, np.int64),
               tile_sum=np.zeros(64 * 64), tile_n=np.zeros(64 * 64))
    with ProcessPoolExecutor(max_workers=4) as ex:
        for part in ex.map(slab_work, list(ZCH)):
            for k in acc:
                acc[k] += part[k]
            print(f"cum n={acc['n']} inward={acc['inward']} "
                  f"outward={acc['outward']} t={time.time()-t0:.0f}s",
                  flush=True)
    decided = max(acc["inward"] + acc["outward"], 1)
    decided_null = max(acc["inward_null"] + acc["outward_null"], 1)
    decided_ideal = max(acc["inward_ideal"] + acc["outward_ideal"], 1)
    offs = np.arange(-4.0, 4.01, 0.5)
    oh, oi = acc["off_hist"], acc["off_hist_ideal"]
    tn = np.maximum(acc["tile_n"], 1)
    tmean = acc["tile_sum"] / tn
    big = acc["tile_n"] > 500
    stats = dict(
        n_pts=int(acc["n"]),
        inward_frac_of_decided=acc["inward"] / decided,
        outward_frac_of_decided=acc["outward"] / decided,
        tie_frac=acc["tie"] / max(acc["n"], 1),
        null_inward_frac_of_decided=acc["inward_null"] / decided_null,
        ideal_inward_frac_of_decided=acc["inward_ideal"] / decided_ideal,
        n_ideal=int(acc["n_ideal"]),
        n_null=int(acc["n_null"]),
        offsets_vox=[float(v) for v in offs],
        off_hist=[int(v) for v in oh],
        off_hist_ideal=[int(v) for v in oi],
        off_frac_inward=float(oh[offs < -0.25].sum() / max(oh.sum(), 1)),
        off_frac_center=float(oh[np.abs(offs) <= 0.25].sum()
                              / max(oh.sum(), 1)),
        off_frac_outward=float(oh[offs > 0.25].sum() / max(oh.sum(), 1)),
        ideal_frac_inward=float(oi[offs < -0.25].sum() / max(oi.sum(), 1)),
        n_tiles_big=int(big.sum()),
        tiles_strong_inward=float((tmean[big] < -0.5).mean()),
        tiles_strong_outward=float((tmean[big] > 0.5).mean()),
        tiles_mixed=float((np.abs(tmean[big]) <= 0.5).mean()))
    print(json.dumps({k: v for k, v in stats.items()
                      if not k.startswith("off_hist")}, indent=1), flush=True)
    json.dump(stats, open(OUT / "pass8_stats.json", "w"), indent=1)
    np.savez_compressed(OUT / "pass8_tiles.npz", tile_mean=tmean,
                        tile_n=acc["tile_n"])


if __name__ == "__main__":
    main()
