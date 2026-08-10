#!/usr/bin/env python3
"""Pass 4b: account for every tissue window - can the spacing instrument fail?

Every window with enough tissue gets an outcome:
  ok          spacing measured
  drop_coher  structure tensor coherence < 0.2  (isotropic texture / fog)
  drop_flat   detrended profile std < 1.0       (no stripe contrast)
  drop_peaks  fewer than 3 peaks                (merged / short profile)
Compressed fog would hide in the drop channels; map them.
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks

import sys
sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
from pass1 import read_zrange, zmeta

BASE = Path("/root/fl_probe/reg0139")
OUT = BASE / "reg2"
HI4, VOX = BASE / "hi_L4", 18.064

WIN, STRIDE, MASK_MIN = 96, 48, 0.6
ZSLICES_MM = [2.0, 5.5, 9.0, 12.5, 16.0, 19.5]

OK, D_COH, D_FLAT, D_PK = 0, 1, 2, 3


def window_outcome(w):
    g = ndi.gaussian_filter(w.astype(np.float32), 1.5)
    gy, gx = ndi.sobel(g, 0), ndi.sobel(g, 1)
    J = np.array([[(gy * gy).mean(), (gy * gx).mean()],
                  [(gy * gx).mean(), (gx * gx).mean()]])
    ev, evec = np.linalg.eigh(J)
    coher = (ev[1] - ev[0]) / max(ev[1] + ev[0], 1e-9)
    if coher < 0.2:
        return D_COH, np.nan
    n = evec[:, 1]
    yy, xx = np.mgrid[0:w.shape[0], 0:w.shape[1]]
    proj = (yy - w.shape[0] / 2) * n[0] + (xx - w.shape[1] / 2) * n[1]
    t = np.round(proj).astype(int)
    t -= t.min()
    prof = np.bincount(t.ravel(), weights=g.ravel())
    cnt = np.bincount(t.ravel())
    okm = cnt > 3
    prof = prof[okm] / cnt[okm]
    if len(prof) < 12:
        return D_PK, np.nan
    prof = prof - ndi.gaussian_filter1d(prof, 12)
    sd = prof.std()
    if sd < 1.0:
        return D_FLAT, np.nan
    pk, _ = find_peaks(prof, prominence=0.6 * sd)
    if len(pk) < 3:
        return D_PK, np.nan
    return OK, float(np.median(np.diff(pk)) * VOX)


def main():
    m = zmeta(HI4)
    Z, Y, X = m["shape"]
    rows = []
    for zmm in ZSLICES_MM:
        zc = int(zmm * 1000 / VOX)
        sl = read_zrange(HI4, zc - 1, zc + 2).mean(0)
        msk = sl > 0
        for y0 in range(0, Y - WIN, STRIDE):
            for x0 in range(0, X - WIN, STRIDE):
                wm = msk[y0:y0 + WIN, x0:x0 + WIN]
                if wm.mean() < MASK_MIN:
                    continue
                o, sp = window_outcome(sl[y0:y0 + WIN, x0:x0 + WIN])
                rows.append((zmm, y0 + WIN / 2, x0 + WIN / 2, o, sp))
        print(f"z={zmm} cum={len(rows)}", flush=True)
    rows = np.array(rows)
    o = rows[:, 3]
    stats = dict(
        n_tissue=int(len(rows)),
        frac_ok=float((o == OK).mean()),
        frac_drop_coher=float((o == D_COH).mean()),
        frac_drop_flat=float((o == D_FLAT).mean()),
        frac_drop_peaks=float((o == D_PK).mean()))
    sp = rows[o == OK, 4]
    stats.update(spacing_median_um=float(np.median(sp)),
                 frac_below_100um_of_ok=float((sp < 100).mean()),
                 frac_below_130um_of_ok=float((sp < 130).mean()))
    print(json.dumps(stats, indent=1), flush=True)
    np.savez_compressed(OUT / "pass4b_outcomes.npz", rows=rows)
    json.dump(stats, open(OUT / "pass4b_stats.json", "w"), indent=1)

    # per-slice outcome+spacing map
    from PIL import Image
    for zmm in (9.0, 16.0):
        zc = int(zmm * 1000 / VOX)
        sl = read_zrange(HI4, zc, zc + 1)[0]
        H4, W4 = sl.shape[0] // 4, sl.shape[1] // 4
        bg = (np.clip(sl[:H4 * 4:4, :W4 * 4:4].astype(np.float32) / 160,
                      0, 1) * 130).astype(np.uint8)
        img = np.stack([bg, bg, bg], -1)
        sel = rows[np.abs(rows[:, 0] - zmm) < 0.1]
        r = 5
        for _, yc, xc, oo, s in sel:
            y, x = int(yc / 4), int(xc / 4)
            if oo == OK:
                if s < 130:
                    col = (255, 60, 60)
                elif s < 180:
                    col = (255, 200, 50)
                else:
                    continue
            elif oo == D_COH:
                col = (60, 120, 255)
            elif oo == D_FLAT:
                col = (170, 60, 255)
            else:
                col = (60, 255, 140)
            y0m, x0m = max(0, y - r), max(0, x - r)
            img[y0m:min(H4, y + r), x0m:min(W4, x + r)] = col
        Image.fromarray(img).save(OUT / f"pass4b_map_z{zmm:.0f}.png")
    print("maps saved", flush=True)


if __name__ == "__main__":
    main()
