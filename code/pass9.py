#!/usr/bin/env python3
"""Pass 9: identical audit of the older surface-recto-090 prediction.

Same analysis box, truth, and instruments as passes 5b/6/7b, so the two
production models can be compared row by row.  Metrics per slab:
point recall at 1/2/3 vox + null, arc-level recall/gone + null, and the
side test with the ideal ceiling.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

import sys
sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
from pass5b import (maxpool2, Y1_0, X1_0, ERODE, HI4, OUT, read_zrange,
                    zmeta, ZCH, NULL_SHIFT)
from pass7b import normal_field, COH_MIN, STEP

P090 = Path("/root/fl_probe/reg0139/pred090_L0")
BLOSC = numcodecs.Blosc()
TH = 65
CH = 128
Y0_L0, Y1_L0 = 1152, 5760
X0_L0, X1_L0 = 960, 5376
HI = MINV = OFFV = None


def load_090_slab(z0, z1):
    """Assemble [z0,z1) x [Y0,Y1) x [X0,X1) from 128^3 blosc chunks."""
    ny, nx = Y1_L0 - Y0_L0, X1_L0 - X0_L0
    out = np.zeros((z1 - z0, ny, nx), np.uint8)
    for iz in range(z0 // CH, -(-z1 // CH)):
        for iy in range(Y0_L0 // CH, -(-Y1_L0 // CH)):
            for ix in range(X0_L0 // CH, -(-X1_L0 // CH)):
                f = P090 / str(iz) / str(iy) / str(ix)
                if not f.is_file():
                    continue
                a = np.frombuffer(BLOSC.decode(f.read_bytes()),
                                  np.uint8).reshape(CH, CH, CH)
                az, ay, ax = iz * CH, iy * CH, ix * CH
                sz0, sy0, sx0 = max(az, z0), max(ay, Y0_L0), max(ax, X0_L0)
                sz1 = min(az + CH, z1)
                sy1, sx1 = min(ay + CH, Y1_L0), min(ax + CH, X1_L0)
                if sz1 <= sz0 or sy1 <= sy0 or sx1 <= sx0:
                    continue
                out[sz0 - z0:sz1 - z0, sy0 - Y0_L0:sy1 - Y0_L0,
                    sx0 - X0_L0:sx1 - X0_L0] = \
                    a[sz0 - az:sz1 - az, sy0 - ay:sy1 - ay, sx0 - ax:sx1 - ax]
    return out


def slab_work(iz):
    z0 = iz * 192
    slab = load_090_slab(z0, z0 + 192)
    pred = maxpool2(slab) > 0
    del slab
    origin = np.array([iz * 96, Y1_0, X1_0], float)
    aoff = MINV @ (origin - OFFV)
    hiT = ndi.affine_transform(HI, MINV, offset=aoff,
                               output_shape=pred.shape, order=1)
    valid = ndi.binary_erosion(hiT > 0, iterations=ERODE)
    truth = (hiT > TH) & valid
    part = dict(n_center=0, hits={1: 0, 2: 0, 3: 0},
                null_hits_r={1: 0, 2: 0, 3: 0},
                n_pred=0, pred_far2=0,
                n_arc=0, arc_hit=0, arc_gone=0, null_arc_hit=0,
                null_arc_gone=0,
                n_side=0, inward=0, outward=0,
                pred_frac=float((pred & valid).mean()))
    TILE = 64
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
        cd = dpred[ys, xs]
        nd = dnull[ys, xs]
        part["n_center"] += len(ys)
        for r in (1, 2, 3):
            part["hits"][r] += int((cd <= r).sum())
            part["null_hits_r"][r] += int((nd <= r).sum())
        dtruth = ndi.distance_transform_edt(~tb)
        pd = dtruth[pb]
        part["n_pred"] += int(pb.sum())
        part["pred_far2"] += int((pd > 2).sum())
        # arcs
        lab, nlab = ndi.label(ctr, structure=np.ones((3, 3), int))
        lv = lab[ys, xs]
        aid = (lv.astype(np.int64) * 10_000_000
               + (ys // TILE).astype(np.int64) * 2000 + xs // TILE)
        uid, inv = np.unique(aid, return_inverse=True)
        cnt = np.bincount(inv)
        cov = np.bincount(inv, weights=(cd <= 2)) / cnt
        ncov = np.bincount(inv, weights=(nd <= 2)) / cnt
        big = cnt >= 20
        part["n_arc"] += int(big.sum())
        part["arc_hit"] += int((cov[big] >= 0.5).sum())
        part["arc_gone"] += int((big & (cov < 0.1)).sum())
        part["null_arc_hit"] += int((ncov[big] >= 0.5).sum())
        part["null_arc_gone"] += int((big & (ncov < 0.1)).sum())
        # side
        near = cd <= 3
        if near.sum() < 100:
            continue
        yn, xn = ys[near], xs[near]
        nyf, nxf, cohf = normal_field(
            ndi.gaussian_filter(hiT[k].astype(np.float32), 1.5))
        ny, nx = nyf[yn, xn], nxf[yn, xn]
        coh = cohf[yn, xn]
        m = np.nonzero(tb)
        cy, cx = m[0].mean(), m[1].mean()
        flip = (ny * (yn - cy) + nx * (xn - cx)) < 0
        ny = np.where(flip, -ny, ny)
        nx = np.where(flip, -nx, nx)
        ok = coh > COH_MIN
        if ok.sum() < 100:
            continue
        yn, xn, ny, nx = yn[ok], xn[ok], ny[ok], nx[ok]
        din = map_coordinates(dpred, [yn - STEP * ny, xn - STEP * nx],
                              order=1)
        dout = map_coordinates(dpred, [yn + STEP * ny, xn + STEP * nx],
                               order=1)
        part["n_side"] += len(yn)
        part["inward"] += int((din < dout - 0.25).sum())
        part["outward"] += int((dout < din - 0.25).sum())
    return part


def main():
    global HI, MINV, OFFV
    t0 = time.time()
    f3 = np.load(OUT / "pass3_final.npz")
    M2, t2 = f3["M2"], f3["t2"]
    MINV = np.linalg.inv(M2)
    OFFV = 2.0 * t2 + 0.5
    HI = read_zrange(HI4, 0, zmeta(HI4)["shape"][0])
    print(f"hi loaded t={time.time()-t0:.0f}s", flush=True)
    acc = None
    with ProcessPoolExecutor(max_workers=4) as ex:
        for part in ex.map(slab_work, list(ZCH)):
            if acc is None:
                acc = part
            else:
                for k, v in part.items():
                    if isinstance(v, dict):
                        for r in v:
                            acc[k][r] += v[r]
                    else:
                        acc[k] += v
            print(f"cum n_center={acc['n_center']} arcs={acc['n_arc']} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    n = max(acc["n_center"], 1)
    na = max(acc["n_arc"], 1)
    dec = max(acc["inward"] + acc["outward"], 1)
    stats = dict(
        model="surface-recto-090",
        n_centerline=int(acc["n_center"]),
        recall_19um=acc["hits"][1] / n,
        recall_37um=acc["hits"][2] / n,
        recall_56um=acc["hits"][3] / n,
        null_recall_37um=acc["null_hits_r"][2] / n,
        pred_beyond_37um=acc["pred_far2"] / max(acc["n_pred"], 1),
        n_arcs=int(acc["n_arc"]),
        arc_recall_cov50=acc["arc_hit"] / na,
        arc_gone_cov10=acc["arc_gone"] / na,
        null_arc_recall_cov50=acc["null_arc_hit"] / na,
        null_arc_gone_cov10=acc["null_arc_gone"] / na,
        inward_frac_of_decided=acc["inward"] / dec,
        n_side=int(acc["n_side"]))
    print(json.dumps(stats, indent=1), flush=True)
    json.dump(stats, open(OUT / "pass9_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
