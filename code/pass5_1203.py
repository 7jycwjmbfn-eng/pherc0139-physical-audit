#!/usr/bin/env python3
"""PHerc1203 stratified audit: m7 vs physical truth, split by zone type.

Zones (per 1.2 mm tile on the truth grid, physical and model-free):
  L loose       material fraction < 0.45
  C compressed  material fraction 0.45 - 0.80
  F fused       material fraction >= 0.80  (near-solid block)

Metrics per zone, each with the shifted-null control:
  centerline recall @37um, arc recall (>=50% cov), arcs fully missed.

Truth: hi_L3 (19.226 um) resampled into the lo L1 grid (18.724 um) via the
validated transform. Predictions: m7 L0 max-pooled to L1.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np
from scipy import ndimage as ndi

import sys
sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
from pass1 import read_zrange, zmeta

BASE = Path("/root/fl_probe/reg1203")
OUT = BASE / "reg"
PRED = BASE / "pred_L0"
HI3 = BASE / "hi_L3"
BLOSC = numcodecs.Blosc()

LO1_VOX, HI3_VOX = 18.724, 19.2264
CH = 192
ZCH = range(41, 62)                # lo L0 chunk rows of the overlap
YCH = range(0, 36)
XCH = range(0, 36)
TH = None                          # truth threshold from histogram
NULL_SHIFT = 64
TILE = 64
ERODE = 3

M2 = t2 = Minv1 = off1 = None      # hi_L3 vox -> lo_L1 vox


def load_pred_slab_L1(iz):
    """Decode chunk-by-chunk and max-pool straight to L1 (memory-safe)."""
    h = CH // 2
    ny, nx = len(YCH) * h, len(XCH) * h
    slab = np.zeros((h, ny, nx), bool)
    for jy, iy in enumerate(YCH):
        for jx, ix in enumerate(XCH):
            f = PRED / str(iz) / str(iy) / str(ix)
            if not f.is_file():
                continue
            a = np.frombuffer(BLOSC.decode(f.read_bytes()),
                              np.uint8).reshape(CH, CH, CH)
            p = a.reshape(h, 2, h, 2, h, 2).max((1, 3, 5)) > 0
            slab[:, jy * h:(jy + 1) * h, jx * h:(jx + 1) * h] = p
    return slab


def slab_work(iz):
    pred = load_pred_slab_L1(iz)
    z1_0 = iz * 96                       # L1 slab origin (global L1 coords)
    # hi_L3 z window for this slab (theta~0: contiguous); include off1!
    pts = np.array([[z1_0, 0, 0], [z1_0 + 96, 3400, 3400]], float)
    hz = (Minv1 @ pts.T).T[:, 0] + off1[0]
    hz0, hz1 = int(min(hz)) - 4, int(max(hz)) + 5
    hz0 = max(hz0, 0)
    hi = read_zrange(HI3, hz0, hz1)
    out_shape = pred.shape
    # affine: p_hi = Minv1 @ (p_L1 - t) ; adjust offset for hi z base
    aoff = off1 + np.array([z1_0, 0, 0]) @ Minv1.T - np.array([hz0, 0, 0])
    hiT = ndi.affine_transform(hi, Minv1, offset=aoff,
                               output_shape=out_shape, order=1)
    del hi
    valid = ndi.binary_erosion(hiT > 0, iterations=ERODE)
    truth = (hiT > TH) & valid
    part = {z: dict(n_ctr=0, hit2=0, null2=0, n_arc=0, arc_hit=0,
                    arc_gone=0, narc_hit=0, narc_gone=0)
            for z in "LCF"}
    part["tiles"] = {z: 0 for z in "LCF"}
    for k in range(0, out_shape[0], 4):
        tb, pb = truth[k], pred[k] & valid[k]
        if tb.sum() < 500 or pb.sum() < 100:
            continue
        # zone map for this slice: material fraction per TILE
        mf = ndi.uniform_filter(tb.astype(np.float32), TILE)
        dt = ndi.distance_transform_edt(tb)
        ctr = tb & (dt >= ndi.maximum_filter(dt, 3)) & (dt >= 1)
        dpred = ndi.distance_transform_edt(~pb)
        pbs = np.zeros_like(pb)
        pbs[NULL_SHIFT:] = pb[:-NULL_SHIFT]
        dnull = ndi.distance_transform_edt(~pbs)
        ys, xs = np.nonzero(ctr)
        cd = dpred[ys, xs]
        nd = dnull[ys, xs]
        zf = mf[ys, xs]
        zone = np.where(zf >= 0.80, 2, np.where(zf >= 0.45, 1, 0))
        lab, _ = ndi.label(ctr, structure=np.ones((3, 3), int))
        lv = lab[ys, xs]
        aid = (lv.astype(np.int64) * 10_000_000
               + (ys // TILE).astype(np.int64) * 2000 + xs // TILE)
        uid, inv = np.unique(aid, return_inverse=True)
        cnt = np.bincount(inv)
        cov = np.bincount(inv, weights=(cd <= 2)) / cnt
        ncov = np.bincount(inv, weights=(nd <= 2)) / cnt
        azone = np.round(np.bincount(inv, weights=zone) / cnt).astype(int)
        big = cnt >= 20
        for zi, zn in enumerate("LCF"):
            pm = zone == zi
            p = part[zn]
            p["n_ctr"] += int(pm.sum())
            p["hit2"] += int((cd[pm] <= 2).sum())
            p["null2"] += int((nd[pm] <= 2).sum())
            am = big & (azone == zi)
            p["n_arc"] += int(am.sum())
            p["arc_hit"] += int((cov[am] >= 0.5).sum())
            p["arc_gone"] += int((am & (cov < 0.1)).sum())
            p["narc_hit"] += int((ncov[am] >= 0.5).sum())
            p["narc_gone"] += int((am & (ncov < 0.1)).sum())
    return part


def main():
    global M2, t2, Minv1, off1, TH
    t0 = time.time()
    f = np.load(OUT / "pass3_final.npz")
    M2h, t2h = f["M2"], f["t2"]          # hi_L4 vox -> lo_L2 vox
    # convert to hi_L3 vox -> lo_L1 vox:
    # p_L4 = p_L3/2 ; p_L1 = 2*p_L2 + 0.5
    M2 = M2h                              # (2 and 1/2 cancel in linear part)
    t2 = 2.0 * t2h + 0.5
    Minv1 = np.linalg.inv(M2)
    off1 = -(Minv1 @ t2)
    # threshold from a mid hi_L3 slice
    mid = read_zrange(HI3, 832, 833)[0]
    v = mid[mid > 0]
    h = np.bincount(v, minlength=256).astype(float)
    h = ndi.gaussian_filter1d(h, 3)
    lo_pk = int(np.argmax(h[:100]))
    hi_pk = int(100 + np.argmax(h[100:]))
    TH = int(lo_pk + np.argmin(h[lo_pk:hi_pk]))
    print(f"threshold {TH} (air {lo_pk}, sheet {hi_pk})", flush=True)

    acc = None
    with ProcessPoolExecutor(max_workers=3) as ex:
        for part in ex.map(slab_work, list(ZCH)):
            if acc is None:
                acc = part
            else:
                for zn in "LCF":
                    for k in part[zn]:
                        acc[zn][k] += part[zn][k]
            print(f"slab done t={time.time()-t0:.0f}s "
                  f"L={acc['L']['n_ctr']} C={acc['C']['n_ctr']} "
                  f"F={acc['F']['n_ctr']}", flush=True)
    stats = dict(threshold=TH)
    for zn, name in zip("LCF", ("loose", "compressed", "fused")):
        p = acc[zn]
        n = max(p["n_ctr"], 1)
        na = max(p["n_arc"], 1)
        stats[name] = dict(
            n_centerline=int(p["n_ctr"]), n_arcs=int(p["n_arc"]),
            recall_37um=round(p["hit2"] / n, 4),
            null_recall_37um=round(p["null2"] / n, 4),
            arc_recall=round(p["arc_hit"] / na, 4),
            arc_gone=round(p["arc_gone"] / na, 4),
            null_arc_recall=round(p["narc_hit"] / na, 4),
            null_arc_gone=round(p["narc_gone"] / na, 4))
    print(json.dumps(stats, indent=1), flush=True)
    json.dump(stats, open(OUT / "pass5_stratified.json", "w"), indent=1)


if __name__ == "__main__":
    main()
