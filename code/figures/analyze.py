#!/usr/bin/env python3
"""Re-run the shipped evaluator's arc logic on single axial slices.

PHerc0139: check each flagged stretch against the per-arc list published in
audit_submission/results/pass6_gone_pts.npy. Point for point.

PHerc1203: no such list ships, so the only check available is the rate. The
per-slice fully-missed fraction has to sit around the 0.1267 in the committed
eval_selfcheck_1203.json. That is a weaker check and is reported as one.

Usage: analyze.py <scroll> <z> [z ...]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_util import SCROLLS, read_box, read_labels    # noqa: E402

OUT = Path(os.environ.get("SLICES",
                          Path(__file__).resolve().parent / "slices"))


def _dist(band):
    if not band.any():
        return np.full(band.shape, np.inf, np.float32)
    return ndi.distance_transform_edt(~band).astype(np.float32)


def slice_arcs(scroll, gz, verbose=True):
    cfg = SCROLLS[scroll]
    oz, oy, ox = cfg["origin"]
    _, Y, X = cfg["shape"]
    lz = gz - oz
    lab = read_labels(lz, lz + 1, 0, Y, 0, X, scroll)[0]
    # the shipped evaluator skips a slice with under 500 centerline pixels;
    # apply the same guard before paying for the prediction chunks. The fine
    # scans are mosaics, so some z carry no reference at all.
    if ((lab & 4) > 0).sum() < 500:
        return None, dict(lab=lab, pred=None)
    p0 = read_box(cfg["pred"], 0, 192, True, 2 * gz, 2 * gz + 2,
                  2 * oy, 2 * (oy + Y), 2 * ox, 2 * (ox + X),
                  verbose=verbose)
    pred = p0.reshape(1, 2, Y, 2, X, 2).max((1, 3, 5))[0] > 0
    del p0

    valid = (lab & 1) > 0
    ctr = (lab & 4) > 0
    pb = pred & valid

    dpred = _dist(pb)
    ys, xs = np.nonzero(ctr)
    cd = dpred[ys, xs]
    lab2, _ = ndi.label(ctr, structure=np.ones((3, 3), int))
    lv = lab2[ys, xs]
    aid = (lv.astype(np.int64) * 10_000_000
           + (ys // 64).astype(np.int64) * 2000 + xs // 64)
    _, inv = np.unique(aid, return_inverse=True)
    cnt = np.bincount(inv)
    cov = np.bincount(inv, weights=(cd <= 2)) / cnt
    gy = np.bincount(inv, weights=ys) / cnt
    gx = np.bincount(inv, weights=xs) / cnt
    big = cnt >= 20
    arcs = [dict(idx=int(i), n=int(cnt[i]), cov=float(cov[i]),
                 cy=float(gy[i]), cx=float(gx[i]),
                 gone=bool(cov[i] < 0.1), hit=bool(cov[i] >= 0.5))
            for i in np.where(big)[0]]
    return arcs, dict(lab=lab, pred=pred)


def check(scroll, gz):
    cfg = SCROLLS[scroll]
    oz, oy, ox = cfg["origin"]
    arcs, S = slice_arcs(scroll, gz)
    if arcs is None:
        print(f"{scroll} z={gz}: no reference on this slice (mosaic gap), "
              f"skipped, same as the evaluator does", flush=True)
        return 0, 0, 0, 0
    gone = [a for a in arcs if a["gone"]]
    hit = [a for a in arcs if a["hit"]]
    rate = len(gone) / max(len(arcs), 1)

    line = (f"{scroll} z={gz}: arcs>=20px {len(arcs)}  fully-missed "
            f"{len(gone)} ({100*rate:.1f}%)  hit {len(hit)}")
    matched = npub = 0
    pubz = np.zeros((0, 5))
    if cfg["gone_pts"]:
        pub = np.load(cfg["gone_pts"])
        pubz = pub[pub[:, 0].astype(int) == gz]
        mine = (np.array([[a["cy"] + oy, a["cx"] + ox] for a in gone])
                if gone else np.zeros((0, 2)))
        for p in pubz[:, 1:3]:
            if len(mine) and np.hypot(mine[:, 0] - p[0],
                                      mine[:, 1] - p[1]).min() <= 3.0:
                matched += 1
        npub = len(pubz)
        line += (f"  published {npub}  matched {matched}/{npub}"
                 f" ({100.0 * matched / max(npub, 1):.0f}%)")
    print(line, flush=True)
    np.savez_compressed(OUT / f"{scroll}_z{gz}.npz", lab=S["lab"],
                        pred=S["pred"], arcs=json.dumps(arcs), pubz=pubz)
    return len(arcs), len(gone), matched, npub


if __name__ == "__main__":
    scroll = sys.argv[1]
    zs = [int(a) for a in sys.argv[2:]]
    OUT.mkdir(exist_ok=True)
    ta = tg = tm = tp = 0
    for gz in zs:
        na, ng, m, p = check(scroll, gz)
        ta += na
        tg += ng
        tm += m
        tp += p
    print(f"\n{scroll} TOTAL: {tg}/{ta} arcs fully missed "
          f"({100.0 * tg / max(ta, 1):.2f}%)", flush=True)
    if tp:
        print(f"point-for-point against published list: {tm}/{tp} "
              f"({100.0 * tm / max(tp, 1):.1f}%)", flush=True)
