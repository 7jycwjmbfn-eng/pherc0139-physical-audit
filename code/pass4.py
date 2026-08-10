#!/usr/bin/env python3
"""Pass 4: does the hi ROI cover compressed (difficult) regions?

Local sheet spacing map from hi_L4 (18.064 um): 2D windows on axial slices,
structure-tensor dominant orientation, intensity projected onto the sheet
normal, peak-to-peak spacing.  Report the fraction of tissue windows whose
center-to-center spacing is below thresholds, and a spatial map.

Thresholds (center-to-center):
  < 100 um  "compressed"  (sheet ~55 um thick -> air gap < ~45 um ~ 4-5 native
             lo voxels; the regime issue #191 flags as failing)
  <  65 um  "near-fused"  (gap ~ 0-10 um; unresolvable at 9.362 um)
Reference: median winding pitch across GP scrolls is 207-259 um.
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks

import sys
sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
from pass1 import read_zrange

BASE = Path("/root/fl_probe/reg0139")
OUT = BASE / "reg2"
HI4, VOX = BASE / "hi_L4", 18.064          # um/vox

WIN = 96            # window (1.73 mm)
STRIDE = 48
ZSLICES_MM = [2.0, 5.5, 9.0, 12.5, 16.0, 19.5]
MASK_MIN = 0.6


def window_spacing(w):
    """Median peak spacing (um) in one 2D window, or None."""
    g = ndi.gaussian_filter(w.astype(np.float32), 1.5)
    gy = ndi.sobel(g, 0)
    gx = ndi.sobel(g, 1)
    J = np.array([[(gy * gy).mean(), (gy * gx).mean()],
                  [(gy * gx).mean(), (gx * gx).mean()]])
    ev, evec = np.linalg.eigh(J)
    coher = (ev[1] - ev[0]) / max(ev[1] + ev[0], 1e-9)
    n = evec[:, 1]                          # dominant gradient dir = normal
    if coher < 0.2:
        return None, coher
    yy, xx = np.mgrid[0:w.shape[0], 0:w.shape[1]]
    proj = (yy - w.shape[0] / 2) * n[0] + (xx - w.shape[1] / 2) * n[1]
    t = np.round(proj).astype(int)
    t -= t.min()
    prof = np.bincount(t.ravel(), weights=g.ravel())
    cnt = np.bincount(t.ravel())
    ok = cnt > 3
    prof = prof[ok] / cnt[ok]
    if len(prof) < 12:
        return None, coher
    prof = prof - ndi.gaussian_filter1d(prof, 12)      # detrend
    sd = prof.std()
    if sd < 1.0:
        return None, coher
    pk, _ = find_peaks(prof, prominence=0.6 * sd)
    if len(pk) < 3:
        return None, coher
    d = np.diff(pk) * VOX
    return float(np.median(d)), coher


def main():
    import pass1
    m = pass1.zmeta(HI4)
    Z, Y, X = m["shape"]
    rows = []
    for zmm in ZSLICES_MM:
        zc = int(zmm * 1000 / VOX)
        if zc + 2 >= Z:
            continue
        sl = read_zrange(HI4, zc - 1, zc + 2).mean(0)
        msk = sl > 0
        for y0 in range(0, Y - WIN, STRIDE):
            for x0 in range(0, X - WIN, STRIDE):
                w = sl[y0:y0 + WIN, x0:x0 + WIN]
                wm = msk[y0:y0 + WIN, x0:x0 + WIN]
                if wm.mean() < MASK_MIN:
                    continue
                sp, coh = window_spacing(w)
                if sp is not None:
                    rows.append((zmm, y0 + WIN / 2, x0 + WIN / 2, sp, coh))
        print(f"z={zmm}mm done, cum windows={len(rows)}", flush=True)
    rows = np.array(rows)
    sp = rows[:, 3]
    stats = dict(
        n_windows=int(len(rows)),
        spacing_median_um=float(np.median(sp)),
        spacing_p10_um=float(np.percentile(sp, 10)),
        frac_below_100um=float((sp < 100).mean()),
        frac_below_65um=float((sp < 65).mean()),
        frac_below_130um=float((sp < 130).mean()),
        note="center-to-center sheet spacing from hi_L4 structure-tensor "
             "projected profiles; winding pitch reference 207-259 um")
    print(json.dumps(stats, indent=1), flush=True)
    np.savez_compressed(OUT / "pass4_spacing.npz", rows=rows)
    json.dump(stats, open(OUT / "pass4_stats.json", "w"), indent=1)

    # spatial map PNG for the middle slice
    try:
        from PIL import Image
        zmid = 9.0
        sel = rows[np.abs(rows[:, 0] - zmid) < 0.1]
        img = np.zeros((Y // 4, X // 4, 3), np.uint8)
        sl = read_zrange(HI4, int(zmid * 1000 / VOX), int(zmid * 1000 / VOX) + 1)[0]
        bg = (np.clip(sl[::4, ::4].astype(np.float32) / 160, 0, 1) * 130
              ).astype(np.uint8)
        img[..., 0] = bg
        img[..., 1] = bg
        img[..., 2] = bg
        for _, yc, xc, s, _ in sel:
            y, x = int(yc / 4), int(xc / 4)
            r = 5
            if s < 65:
                col = (255, 40, 40)
            elif s < 100:
                col = (255, 160, 30)
            elif s < 130:
                col = (250, 240, 60)
            else:
                continue
            img[max(0, y - r):y + r, max(0, x - r):x + r] = col
        Image.fromarray(img).save(OUT / "pass4_map_z9.png")
        print("map saved", flush=True)
    except Exception as e:
        print("map skipped:", e, flush=True)


if __name__ == "__main__":
    main()
