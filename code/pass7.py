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
STEP = 2.0          # vox offset along radial direction
HI = MINV = OFFV = None


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
                n_null=0, inward_null=0, outward_null=0)
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
        # inward = toward material centroid of this slice
        m = np.nonzero(tb)
        cy, cx = m[0].mean(), m[1].mean()
        ry, rx = ys - cy, xs - cx
        rn = np.sqrt(ry ** 2 + rx ** 2) + 1e-9
        ry, rx = ry / rn, rx / rn
        din = map_coordinates(dpred, [ys - STEP * ry, xs - STEP * rx],
                              order=1)
        dout = map_coordinates(dpred, [ys + STEP * ry, xs + STEP * rx],
                               order=1)
        part["n"] += len(ys)
        part["inward"] += int((din < dout - 0.25).sum())
        part["outward"] += int((dout < din - 0.25).sum())
        part["tie"] += int((np.abs(din - dout) <= 0.25).sum())
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
    t0 = time.time()
    f3 = np.load(OUT / "pass3_final.npz")
    M2, t2 = f3["M2"], f3["t2"]
    MINV = np.linalg.inv(M2)
    OFFV = 2.0 * t2 + 0.5
    m4 = zmeta(HI4)
    HI = read_zrange(HI4, 0, m4["shape"][0])
    print(f"hi loaded t={time.time()-t0:.0f}s", flush=True)
    acc = dict(n=0, inward=0, outward=0, tie=0,
               n_null=0, inward_null=0, outward_null=0)
    with ProcessPoolExecutor(max_workers=4) as ex:
        for part in ex.map(slab_work, list(ZCH)):
            for k in acc:
                acc[k] += part[k]
            print(f"cum n={acc['n']} inward={acc['inward']} "
                  f"outward={acc['outward']} t={time.time()-t0:.0f}s",
                  flush=True)
    decided = max(acc["inward"] + acc["outward"], 1)
    decided_null = max(acc["inward_null"] + acc["outward_null"], 1)
    stats = dict(
        n_pts=int(acc["n"]),
        inward_frac_of_decided=acc["inward"] / decided,
        outward_frac_of_decided=acc["outward"] / decided,
        tie_frac=acc["tie"] / max(acc["n"], 1),
        null_inward_frac_of_decided=acc["inward_null"] / decided_null,
        n_null=int(acc["n_null"]))
    print(json.dumps(stats, indent=1), flush=True)
    json.dump(stats, open(OUT / "pass7_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
