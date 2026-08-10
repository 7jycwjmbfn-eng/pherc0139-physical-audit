#!/usr/bin/env python3
"""Pass 5: model-free physical audit of surface-m7 predictions on PHerc0139.

Truth = 1.129 um scan (hi_L4, 18.064 um) mapped into the lo frame with the
pass2/3 transform (rigid+scale, held-out 4.1 um).  Predictions = official
20260413222639-surface-m7-L0-th0.2 (binary), max-pooled L0 -> L1 (18.724 um).

Per axial slice (2D; sheets are near-vertical so 2D distance ~ 3D):
  recall:  fraction of truth sheet centerline pixels with a predicted-positive
           within r in {1,2,3} L1 vox (19/37/56 um)
  localization: distance from predicted positives to nearest truth material
  miss physics: hi intensity at hit vs missed centerline pixels
  null control: same recall with predictions shifted +64 vox in y
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

BASE = Path("/root/fl_probe/reg0139")
OUT = BASE / "reg2"
PRED = BASE / "pred_L0"
HI4 = BASE / "hi_L4"

CHUNK = 192
ZCH = range(18, 31)
YCH = range(6, 30)          # L0 vox 1152..5760
XCH = range(5, 28)          # L0 vox  960..5376
Y1_0, X1_0 = 576, 480       # L1 origin of the slab box
TH_SHEET = None             # set from histogram
ERODE = 3
NULL_SHIFT = 64
BLOSC = numcodecs.Blosc()


def load_pred_slab(iz):
    ny, nx = len(YCH) * CHUNK, len(XCH) * CHUNK
    slab = np.zeros((CHUNK, ny, nx), np.uint8)
    miss = 0
    for jy, iy in enumerate(YCH):
        for jx, ix in enumerate(XCH):
            f = PRED / str(iz) / str(iy) / str(ix)
            if not f.is_file():
                miss += 1
                continue
            a = np.frombuffer(BLOSC.decode(f.read_bytes()), np.uint8)
            slab[:, jy * CHUNK:(jy + 1) * CHUNK,
                 jx * CHUNK:(jx + 1) * CHUNK] = a.reshape(CHUNK, CHUNK, CHUNK)
    return slab, miss


def maxpool2(a):
    z, y, x = a.shape
    return a[:z - z % 2, :y - y % 2, :x - x % 2].reshape(
        z // 2, 2, y // 2, 2, x // 2, 2).max((1, 3, 5))


HI = MINV = OFFV = None


def slab_work(iz):
    slab, missc = load_pred_slab(iz)
    pred = maxpool2(slab) > 0
    del slab
    origin = np.array([iz * 96, Y1_0, X1_0], float)
    aoff = MINV @ (origin - OFFV)
    hiT = ndi.affine_transform(HI, MINV, offset=aoff,
                               output_shape=pred.shape, order=1)
    valid = ndi.binary_erosion(hiT > 0, iterations=ERODE)
    truth = (hiT > TH_SHEET) & valid
    part = dict(n_arc=0, arc_hit=0, arc_gone=0,
                null_arc_hit=0, null_arc_gone=0,
                gone_pts=[], gone_int_sum=0.0,
                arc_int_hit_sum=0.0, arc_int_gone_sum=0.0,
                sl_stats=dict(iz=iz, missing_chunks=missc,
                              truth_frac=float(truth.mean()),
                              pred_frac=float((pred & valid).mean())))
    TILE = 64
    for k in range(0, pred.shape[0], 4):        # every 4th slice suffices
        tb, pb = truth[k], pred[k] & valid[k]
        if tb.sum() < 500 or pb.sum() < 100:
            continue
        dt = ndi.distance_transform_edt(tb)
        ctr = tb & (dt >= ndi.maximum_filter(dt, 3)) & (dt >= 1)
        lab, nlab = ndi.label(ctr, structure=np.ones((3, 3), int))
        if nlab == 0:
            continue
        dpred = ndi.distance_transform_edt(~pb)
        pbs = np.zeros_like(pb)
        pbs[NULL_SHIFT:] = pb[:-NULL_SHIFT]
        dnull = ndi.distance_transform_edt(~pbs)
        ys, xs = np.nonzero(ctr)
        lv = lab[ys, xs]
        # arc id = (component, 1.2mm tile)
        aid = (lv.astype(np.int64) * 10_000_000
               + (ys // TILE).astype(np.int64) * 2000 + xs // TILE)
        uid, inv = np.unique(aid, return_inverse=True)
        cnt = np.bincount(inv)
        covered = np.bincount(inv, weights=(dpred[ys, xs] <= 2)) / cnt
        ncov = np.bincount(inv, weights=(dnull[ys, xs] <= 2)) / cnt
        ints = np.bincount(inv, weights=hiT[k][ys, xs]) / cnt
        big = cnt >= 20
        part["n_arc"] += int(big.sum())
        part["arc_hit"] += int((covered[big] >= 0.5).sum())
        gone = big & (covered < 0.1)
        part["arc_gone"] += int(gone.sum())
        part["null_arc_hit"] += int((ncov[big] >= 0.5).sum())
        part["null_arc_gone"] += int((big & (ncov < 0.1)).sum())
        part["arc_int_hit_sum"] += float(ints[big & (covered >= 0.5)].sum())
        part["arc_int_gone_sum"] += float(ints[gone].sum())
        # record locations of fully-missed arcs (global L1 coords)
        if gone.any():
            gy = np.bincount(inv, weights=ys) / cnt
            gx = np.bincount(inv, weights=xs) / cnt
            for gi in np.where(gone)[0]:
                part["gone_pts"].append(
                    (iz * 96 + k, gy[gi] + Y1_0, gx[gi] + X1_0,
                     float(ints[gi]), int(cnt[gi])))
    return part


import os
SENS_TH = int(os.environ.get("SENS_TH", "0"))
SENS_SLABS = [22, 26]


def main():
    global TH_SHEET
    t0 = time.time()
    f3 = np.load(OUT / "pass3_final.npz")
    M2, t2 = f3["M2"], f3["t2"]
    Minv = np.linalg.inv(M2)
    off = 2.0 * t2 + 0.5                    # p_L1 = M2 @ p_L4 + off

    print("loading hi_L4...", flush=True)
    m4 = zmeta(HI4)
    hi = read_zrange(HI4, 0, m4["shape"][0])
    print(f"hi {hi.shape} {hi.nbytes/1e9:.1f}GB t={time.time()-t0:.0f}s",
          flush=True)

    # threshold from a middle L4 slice histogram (in-mask, bimodal air/sheet)
    mid = hi[hi.shape[0] // 2]
    v = mid[mid > 0]
    hist = np.bincount(v, minlength=256).astype(float)
    hist = ndi.gaussian_filter1d(hist, 3)
    lo_pk = int(np.argmax(hist[:100]))
    hi_pk = int(100 + np.argmax(hist[100:]))
    TH_SHEET = int(lo_pk + np.argmin(hist[lo_pk:hi_pk]))
    print(f"threshold: air peak {lo_pk}, sheet peak {hi_pk}, "
          f"valley TH={TH_SHEET}", flush=True)

    global HI, MINV, OFFV
    HI, MINV, OFFV = hi, Minv, off
    acc = dict(n_arc=0, arc_hit=0, arc_gone=0,
               null_arc_hit=0, null_arc_gone=0,
               arc_int_hit_sum=0.0, arc_int_gone_sum=0.0,
               gone_pts=[], per_slab=[])
    if SENS_TH:
        TH_SHEET = SENS_TH
        print(f"sensitivity override TH={TH_SHEET}", flush=True)
    with ProcessPoolExecutor(max_workers=2) as ex:
        for part in ex.map(slab_work, SENS_SLABS):
            for k in ("n_arc", "arc_hit", "arc_gone", "null_arc_hit",
                      "null_arc_gone", "arc_int_hit_sum",
                      "arc_int_gone_sum"):
                acc[k] += part[k]
            acc["gone_pts"].extend(part["gone_pts"])
            acc["per_slab"].append(part["sl_stats"])
            print(f"slab {part['sl_stats']['iz']} done "
                  f"t={time.time()-t0:.0f}s arcs={acc['n_arc']} "
                  f"gone={acc['arc_gone']}", flush=True)

    n = max(acc["n_arc"], 1)
    stats = dict(
        threshold=TH_SHEET,
        n_arcs=int(acc["n_arc"]),
        arc_recall_cov50=acc["arc_hit"] / n,
        arc_gone_cov10=acc["arc_gone"] / n,
        null_arc_recall_cov50=acc["null_arc_hit"] / n,
        null_arc_gone_cov10=acc["null_arc_gone"] / n,
        n_gone=len(acc["gone_pts"]),
        hit_arc_intensity=acc["arc_int_hit_sum"] / max(acc["arc_hit"], 1),
        gone_arc_intensity=acc["arc_int_gone_sum"] / max(acc["arc_gone"], 1),
        per_slab=acc["per_slab"])
    print(json.dumps({k: v for k, v in stats.items() if k != "per_slab"},
                     indent=1), flush=True)
    json.dump(stats, open(OUT / f"pass6s_th{TH_SHEET}.json", "w"), indent=1)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
