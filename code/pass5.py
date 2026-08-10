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
    part = dict(n_center=0, hits={1: 0, 2: 0, 3: 0}, null_hits=0,
                n_pred=0, pred_far2=0, pred_far4=0,
                hit_int_sum=0.0, hit_int_n=0, miss_int_sum=0.0, miss_int_n=0,
                dist_hist=np.zeros(21, np.int64),
                sl_stats=dict(iz=iz, missing_chunks=missc,
                              truth_frac=float(truth.mean()),
                              pred_frac=float((pred & valid).mean())))
    for k in range(pred.shape[0]):
        tb, pb = truth[k], pred[k] & valid[k]
        if tb.sum() < 500 or pb.sum() < 100:
            continue
        dt = ndi.distance_transform_edt(tb)
        ctr = tb & (dt >= ndi.maximum_filter(dt, 3)) & (dt >= 1)
        dpred = ndi.distance_transform_edt(~pb)
        cd = dpred[ctr]
        part["n_center"] += int(ctr.sum())
        for r in (1, 2, 3):
            part["hits"][r] += int((cd <= r).sum())
        hit2 = cd <= 2
        ints = hiT[k][ctr]
        part["hit_int_sum"] += float(ints[hit2].sum())
        part["hit_int_n"] += int(hit2.sum())
        part["miss_int_sum"] += float(ints[~hit2].sum())
        part["miss_int_n"] += int((~hit2).sum())
        pbs = np.zeros_like(pb)
        pbs[NULL_SHIFT:] = pb[:-NULL_SHIFT]
        dnull = ndi.distance_transform_edt(~pbs)
        part["null_hits"] += int((dnull[ctr] <= 2).sum())
        dtruth = ndi.distance_transform_edt(~tb)
        pd = dtruth[pb]
        part["n_pred"] += int(pb.sum())
        part["pred_far2"] += int((pd > 2).sum())
        part["pred_far4"] += int((pd > 4).sum())
        part["dist_hist"] += np.bincount(
            np.clip(pd.astype(int), 0, 20), minlength=21)
    return part


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
    R = [1, 2, 3]
    acc = dict(n_center=0, hits={r: 0 for r in R}, null_hits=0,
               n_pred=0, pred_far2=0, pred_far4=0,
               hit_int_sum=0.0, hit_int_n=0, miss_int_sum=0.0, miss_int_n=0,
               dist_hist=np.zeros(21, np.int64), per_slab=[])
    with ProcessPoolExecutor(max_workers=4) as ex:
        for part in ex.map(slab_work, list(ZCH)):
            for r in R:
                acc["hits"][r] += part["hits"][r]
            for k in ("n_center", "null_hits", "n_pred", "pred_far2",
                      "pred_far4", "hit_int_sum", "hit_int_n",
                      "miss_int_sum", "miss_int_n"):
                acc[k] += part[k]
            acc["dist_hist"] += np.asarray(part["dist_hist"])
            acc["per_slab"].append(part["sl_stats"])
            print(f"slab {part['sl_stats']['iz']} done "
                  f"t={time.time()-t0:.0f}s "
                  f"truth={part['sl_stats']['truth_frac']:.3f} "
                  f"pred={part['sl_stats']['pred_frac']:.3f}", flush=True)

    n = max(acc["n_center"], 1)
    stats = dict(
        threshold=TH_SHEET,
        n_centerline=int(acc["n_center"]),
        recall_19um=acc["hits"][1] / n,
        recall_37um=acc["hits"][2] / n,
        recall_56um=acc["hits"][3] / n,
        null_recall_37um=acc["null_hits"] / n,
        n_pred_pos=int(acc["n_pred"]),
        pred_beyond_37um=acc["pred_far2"] / max(acc["n_pred"], 1),
        pred_beyond_75um=acc["pred_far4"] / max(acc["n_pred"], 1),
        hit_intensity_mean=acc["hit_int_sum"] / max(acc["hit_int_n"], 1),
        miss_intensity_mean=acc["miss_int_sum"] / max(acc["miss_int_n"], 1),
        per_slab=acc["per_slab"])
    print(json.dumps({k: v for k, v in stats.items() if k != "per_slab"},
                     indent=1), flush=True)
    json.dump(stats, open(OUT / "pass5_stats.json", "w"), indent=1)
    np.save(OUT / "pass5_dist_hist.npy", acc["dist_hist"])
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
