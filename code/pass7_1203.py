#!/usr/bin/env python3
"""PHerc1203 side-of-sheet audit: replicate the 0139 finding on the GP scroll.

Same instrument as 0139 pass7b (structure-tensor normal, ideal-recto ceiling,
shifted null), streamed per slab like pass5b_1203.
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
from pass1 import read_zrange
from pass7b import normal_field, COH_MIN, STEP
sys.path.insert(0, "/root/fl_probe/reg1203")
import pass5b_1203 as S

OUT = Path("/root/fl_probe/reg1203/reg")


def slab_work(iz):
    pred = S.load_pred_slab_L1(iz)
    z1_0 = iz * 96
    pts = np.array([[z1_0, 0, 0], [z1_0 + 96, 3400, 3400]], float)
    hz = (S.Minv1 @ pts.T).T[:, 0] + S.off1[0]
    hz0 = max(int(min(hz)) - 4, 0)
    hi = read_zrange(S.HI3, hz0, int(max(hz)) + 5)
    aoff = (S.off1 + np.array([z1_0, 0, 0]) @ S.Minv1.T
            - np.array([hz0, 0, 0]))
    hiT = ndi.affine_transform(hi, S.Minv1, offset=aoff,
                               output_shape=pred.shape, order=1)
    del hi
    valid = ndi.binary_erosion(hiT > 0, iterations=S.ERODE)
    truth = (hiT > S.TH) & valid
    part = dict(n=0, inward=0, outward=0,
                n_null=0, inward_null=0, outward_null=0,
                n_ideal=0, inward_ideal=0, outward_ideal=0)
    for k in range(0, pred.shape[0], 8):
        tb, pb = truth[k], pred[k] & valid[k]
        if tb.sum() < 500 or pb.sum() < 100:
            continue
        dt = ndi.distance_transform_edt(tb)
        ctr = tb & (dt >= ndi.maximum_filter(dt, 3)) & (dt >= 1)
        dpred = ndi.distance_transform_edt(~pb)
        pbs = np.zeros_like(pb)
        pbs[S.NULL_SHIFT:] = pb[:-S.NULL_SHIFT]
        dnull = ndi.distance_transform_edt(~pbs)
        ys, xs = np.nonzero(ctr)
        near = dpred[ys, xs] <= 3
        if near.sum() < 100:
            continue
        ys, xs = ys[near], xs[near]
        nyf, nxf, cohf = normal_field(
            ndi.gaussian_filter(hiT[k].astype(np.float32), 1.5))
        ny, nx = nyf[ys, xs], nxf[ys, xs]
        coh = cohf[ys, xs]
        m = np.nonzero(tb)
        cy, cx = m[0].mean(), m[1].mean()
        flip = (ny * (ys - cy) + nx * (xs - cx)) < 0
        ny = np.where(flip, -ny, ny)
        nx = np.where(flip, -nx, nx)
        ok = coh > COH_MIN
        if ok.sum() < 100:
            continue
        ys, xs, ny, nx = ys[ok], xs[ok], ny[ok], nx[ok]
        din = map_coordinates(dpred, [ys - STEP * ny, xs - STEP * nx],
                              order=1)
        dout = map_coordinates(dpred, [ys + STEP * ny, xs + STEP * nx],
                               order=1)
        part["n"] += len(ys)
        part["inward"] += int((din < dout - 0.25).sum())
        part["outward"] += int((dout < din - 0.25).sum())
        nn = dnull[ys, xs] <= 3
        if nn.sum() > 100:
            di = map_coordinates(dnull, [ys[nn] - STEP * ny[nn],
                                         xs[nn] - STEP * nx[nn]], order=1)
            do = map_coordinates(dnull, [ys[nn] + STEP * ny[nn],
                                         xs[nn] + STEP * nx[nn]], order=1)
            part["n_null"] += int(nn.sum())
            part["inward_null"] += int((di < do - 0.25).sum())
            part["outward_null"] += int((do < di - 0.25).sum())
        # ideal recto band ceiling
        YY, XX = np.mgrid[0:tb.shape[0], 0:tb.shape[1]].astype(np.float32)
        sgn = np.sign(nyf * (YY - cy) + nxf * (XX - cx))
        sgn[sgn == 0] = 1.0
        in_air = map_coordinates(tb.astype(np.float32),
                                 [YY - 1.5 * nyf * sgn,
                                  XX - 1.5 * nxf * sgn], order=1) < 0.5
        ideal = ndi.binary_dilation(tb & in_air, iterations=1)
        dideal = ndi.distance_transform_edt(~ideal)
        ni = dideal[ys, xs] <= 3
        if ni.sum() > 100:
            di = map_coordinates(dideal, [ys[ni] - STEP * ny[ni],
                                          xs[ni] - STEP * nx[ni]], order=1)
            do = map_coordinates(dideal, [ys[ni] + STEP * ny[ni],
                                          xs[ni] + STEP * nx[ni]], order=1)
            part["n_ideal"] += int(ni.sum())
            part["inward_ideal"] += int((di < do - 0.25).sum())
            part["outward_ideal"] += int((do < di - 0.25).sum())
    return part


def main():
    t0 = time.time()
    # transform + threshold setup (mirrors pass5b_1203.main)
    f = np.load(OUT / "pass3_final.npz")
    S.M2 = f["M2"]
    S.t2 = 2.0 * f["t2"] + 0.5
    S.Minv1 = np.linalg.inv(S.M2)
    S.off1 = -(S.Minv1 @ S.t2)
    mid = read_zrange(S.HI3, 832, 833)[0]
    v = mid[mid > 0]
    h = ndi.gaussian_filter1d(
        np.bincount(v, minlength=256).astype(float), 3)
    lo_pk = int(np.argmax(h[:100]))
    hi_pk = int(100 + np.argmax(h[100:]))
    S.TH = int(lo_pk + np.argmin(h[lo_pk:hi_pk]))
    print(f"threshold {S.TH}", flush=True)
    acc = None
    with ProcessPoolExecutor(max_workers=3) as ex:
        for part in ex.map(slab_work, list(S.ZCH)):
            if acc is None:
                acc = part
            else:
                for k in part:
                    acc[k] += part[k]
            print(f"cum n={acc['n']} t={time.time()-t0:.0f}s", flush=True)
    dec = max(acc["inward"] + acc["outward"], 1)
    dec_n = max(acc["inward_null"] + acc["outward_null"], 1)
    dec_i = max(acc["inward_ideal"] + acc["outward_ideal"], 1)
    stats = dict(
        n_pts=int(acc["n"]),
        inward_frac=round(acc["inward"] / dec, 4),
        null_inward_frac=round(acc["inward_null"] / dec_n, 4),
        ideal_inward_frac=round(acc["inward_ideal"] / dec_i, 4),
        n_ideal=int(acc["n_ideal"]), n_null=int(acc["n_null"]))
    print(json.dumps(stats, indent=1), flush=True)
    json.dump(stats, open(OUT / "pass7_side.json", "w"), indent=1)


if __name__ == "__main__":
    main()
