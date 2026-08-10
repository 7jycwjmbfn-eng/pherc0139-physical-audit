#!/usr/bin/env python3
"""Visual + raw-intensity verification of the pass1 lock, and lock record.

Lock (pass1, L3/74.896um grid): flip=0, theta=183 deg, zoff=33.6 mm,
shift (dy,dx)=(157.2,153.2).  Mapping for a hi pixel p (y,x) on the 74.896um
resampled grid:  p_lo = R(theta) @ (p - c_hi) + c_hi + (dy,dx),
z_lo_mm = z_hi_mm + zoff,  where c_hi = ((H-1)/2,(W-1)/2) of the resampled
hi slab array (scipy rotate convention), hi array origin at lo array origin.
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

import sys
sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
from pass1 import (LO_ROOT, LO_VOX, HI_ROOT, HI_VOX, slab_image, feature2d,
                   fwd_products, masked_ncc_max)

OUT = Path("/root/fl_probe/reg0139/reg2")
THETA, ZOFF, DY, DX = 183.0, 33.6, 157.2, 153.2


def hi_slab_raw_on_lo_grid(zc_mm):
    img, mask = slab_image(HI_ROOT, HI_VOX, zc_mm)
    zf = HI_VOX / LO_VOX
    img = ndi.zoom(ndi.gaussian_filter(img, 1.0), zf, order=1)
    mask = ndi.zoom(mask.astype(np.float32), zf, order=1) > 0.5
    return img, mask


def place(img, mask, shape):
    """Rotate by THETA about own center, then paste at origin + (DY,DX)."""
    r = ndi.rotate(img, THETA, reshape=False, order=1)
    rm = ndi.rotate(mask.astype(np.float32), THETA, reshape=False, order=1) > 0.5
    out = np.zeros(shape, np.float32)
    om = np.zeros(shape, bool)
    oy, ox = int(round(DY)), int(round(DX))
    H, W = r.shape
    y0, x0 = max(0, oy), max(0, ox)
    y1, x1 = min(shape[0], oy + H), min(shape[1], ox + W)
    out[y0:y1, x0:x1] = r[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    om[y0:y1, x0:x1] = rm[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return out, om


def norm01(a, m):
    v = a[m]
    lo_, hi_ = np.percentile(v, 2), np.percentile(v, 98)
    return np.clip((a - lo_) / max(hi_ - lo_, 1e-6), 0, 1)


def main():
    rows = []
    for zh in (4.0, 10.4, 16.8):
        zl = zh + ZOFF
        lo_img, lo_mask = slab_image(LO_ROOT, LO_VOX, zl)
        hi_img, hi_mask = hi_slab_raw_on_lo_grid(zh)
        hi_p, hi_pm = place(hi_img, hi_mask, lo_img.shape)

        # raw-intensity masked NCC at the locked pose (features not reused)
        both = lo_mask & hi_pm
        a = lo_img[both].astype(np.float64)
        b = hi_p[both].astype(np.float64)
        r = float(np.corrcoef(a, b)[0, 1])
        rows.append(dict(z_hi_mm=zh, z_lo_mm=zl, n_overlap=int(both.sum()),
                         raw_pearson=round(r, 4)))

        # checkerboard overlay png
        A = norm01(lo_img, lo_mask)
        B = norm01(hi_p, hi_pm)
        cell = 52
        yy, xx = np.mgrid[0:A.shape[0], 0:A.shape[1]]
        cb = ((yy // cell + xx // cell) % 2).astype(bool)
        mix = np.where(cb, A, B)
        rgb = np.stack([np.where(cb, A, B * 0.55),
                        np.where(cb, A * 0.75, B),
                        mix], -1)
        try:
            from PIL import Image
            Image.fromarray((rgb * 255).astype(np.uint8)).save(
                OUT / f"verify_checker_z{zh:.1f}.png")
        except ImportError:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.imsave(OUT / f"verify_checker_z{zh:.1f}.png", rgb)
        print("slab", rows[-1], flush=True)

    lock = dict(
        frame="lo_L3 74.896um array grid, axes (z,y,x), scipy rotate about "
              "array center ((H-1)/2,(W-1)/2) of the resampled hi slab",
        flip=0, theta_deg=THETA, zoff_mm=ZOFF, dy_vox=DY, dx_vox=DX,
        lo_vox_um=74.896, hi_resampled_from_um=36.128,
        score_pass1=0.7798, zscore_pass1=27.22, scatter_vox=0.346,
        gantry_pred_zoff_mm=35.457, raw_intensity_check=rows)
    json.dump(lock, open(OUT / "pass1_lock.json", "w"), indent=1)
    print("lock written", flush=True)


if __name__ == "__main__":
    main()
