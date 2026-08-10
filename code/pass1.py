#!/usr/bin/env python3
"""PHerc0139 coarse registration, pass 1: exhaustive global search.

hi = 20260413113053 (1.129 um mosaic ROI), slab features from L5 (36.128 um)
lo = 20250728140407 (9.362 um whole scroll), slab features from L3 (74.896 um)

Search space: z offset over the ENTIRE lo scroll (0.8 mm grid) x in-plane
rotation 0..357 deg (3 deg grid) x z-flip x translation (masked NCC via FFT,
exact for |shift| <= ~25 mm at pad 1024).  No reliance on gantry priors.

Feature: band-pass energy envelope tuned to the ~225 um sheet pitch, z-scored
inside the valid mask.  Identical feature definition for both volumes (hi slab
is resampled to the 74.896 um grid first).

Outputs reg2/pass1_scores.npz with score/dy/dx per (theta, slab, center) and
reg2/pass1_top.json with aggregated candidates.

Modes:
  python3 pass1.py selftest   # synthetic rotation self-recovery on lo data
  python3 pass1.py run        # the real search
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.fft import irfft2, rfft2

BASE = Path("/root/fl_probe/reg0139")
OUT = BASE / "reg2"
OUT.mkdir(exist_ok=True)

LO_ROOT, LO_LEVEL, LO_VOX = BASE / "lo_L3", 3, 74.896   # um/vox
HI_ROOT, HI_LEVEL, HI_VOX = BASE / "hi_L5", 5, 36.128

PAD = 1024                 # FFT size; linear-exact for |shift|<~340 vox
SHIFT_MAX = 270            # +-20.2 mm translation window
SLAB_MM = 0.72             # slab thickness
HI_SLABS_MM = [2.4, 6.4, 10.4, 14.4, 18.4]   # symmetric: flip maps set->itself
ZGRID_MM = 0.8
THETAS = np.arange(0, 360, 3.0)
MIN_MASK = 3000            # min valid pixels for a usable lo slab
MIN_OVL_FRAC = 0.30        # min overlap as fraction of hi mask area
EPS = 1e-6


# ---------------------------------------------------------------- zarr reader
def zmeta(root):
    return json.loads((root / ".zarray").read_text())


def read_zrange(root, z0, z1):
    """Read slices [z0,z1) of a raw uint8 chunked store. Missing chunk = 0."""
    m = zmeta(root)
    Z, Y, X = m["shape"]
    cz, cy, cx = m["chunks"]
    z0, z1 = max(0, z0), min(Z, z1)
    out = np.zeros((z1 - z0, Y, X), np.uint8)
    for iz in range(z0 // cz, -(-z1 // cz)):
        for iy in range(-(-Y // cy)):
            for ix in range(-(-X // cx)):
                f = root / str(iz) / str(iy) / str(ix)
                if not f.is_file():
                    continue
                blk = np.fromfile(f, np.uint8)
                if blk.size != cz * cy * cx:
                    continue
                blk = blk.reshape(cz, cy, cx)
                az0 = iz * cz
                s0, s1 = max(az0, z0), min(az0 + cz, z1)
                if s1 <= s0:
                    continue
                y1, x1 = min(Y, iy * cy + cy), min(X, ix * cx + cx)
                out[s0 - z0:s1 - z0, iy * cy:y1, ix * cx:x1] = \
                    blk[s0 - az0:s1 - az0, :y1 - iy * cy, :x1 - ix * cx]
    return out


# ------------------------------------------------------------------- features
def slab_image(root, vox_um, zc_mm, thick_mm=SLAB_MM):
    half = 0.5 * thick_mm * 1000.0 / vox_um
    zc = zc_mm * 1000.0 / vox_um
    z0, z1 = int(round(zc - half)), int(round(zc + half))
    vol = read_zrange(root, z0, z1)
    if vol.shape[0] == 0:
        return None, None
    img = vol.mean(axis=0, dtype=np.float32)
    valid = (vol > 0).mean(axis=0)
    mask = valid > 0.3
    return img, mask


def feature2d(img, mask):
    """Sheet-pitch band-pass energy envelope, z-scored in mask (74.9 um grid)."""
    f = img.astype(np.float32)
    dog = ndi.gaussian_filter(f, 0.8) - ndi.gaussian_filter(f, 1.6)
    env = ndi.gaussian_filter(np.abs(dog), 2.0)
    m = mask & (env > 0)
    if m.sum() < 500:
        return None, None
    mu, sd = env[m].mean(), env[m].std() + EPS
    feat = np.where(m, (env - mu) / sd, 0.0).astype(np.float32)
    return feat, m


def hi_slab_feature(zc_mm):
    """hi slab -> resample to 74.896 um grid -> same feature def as lo."""
    img, mask = slab_image(HI_ROOT, HI_VOX, zc_mm)
    if img is None:
        return None, None
    zf = HI_VOX / LO_VOX
    img = ndi.zoom(ndi.gaussian_filter(img, 1.0), zf, order=1)
    mask = ndi.zoom(mask.astype(np.float32), zf, order=1) > 0.5
    return feature2d(img, mask)


# ------------------------------------------------------- masked NCC via FFT
def fwd_products(feat, mask):
    m = mask.astype(np.float32)
    fm = feat * m
    return (rfft2(m, s=(PAD, PAD)), rfft2(fm, s=(PAD, PAD)),
            rfft2(fm * feat, s=(PAD, PAD)), float(m.sum()))


_WIN = None


def shift_window():
    global _WIN
    if _WIN is None:
        w = np.full((PAD, PAD), False)
        w[:SHIFT_MAX + 1, :SHIFT_MAX + 1] = True
        w[:SHIFT_MAX + 1, -SHIFT_MAX:] = True
        w[-SHIFT_MAX:, :SHIFT_MAX + 1] = True
        w[-SHIFT_MAX:, -SHIFT_MAX:] = True
        _WIN = w
    return _WIN


def masked_ncc_max(hiF, loF, hi_area):
    """Max masked NCC over shifts within the window; returns (score, dy, dx).

    Correlation convention: value at (u,v) corresponds to shifting the hi
    image by (u,v) (with wraparound; window keeps it linear-exact).
    """
    Hm, Hf, Hff, _ = hiF
    Lm, Lf, Lff, _ = loF
    n = irfft2(np.conj(Hm) * Lm, s=(PAD, PAD))
    sh = irfft2(np.conj(Hf) * Lm, s=(PAD, PAD))
    sl = irfft2(np.conj(Hm) * Lf, s=(PAD, PAD))
    cr = irfft2(np.conj(Hf) * Lf, s=(PAD, PAD))
    shh = irfft2(np.conj(Hff) * Lm, s=(PAD, PAD))
    sll = irfft2(np.conj(Hm) * Lff, s=(PAD, PAD))
    nmin = max(MIN_OVL_FRAC * hi_area, 2000.0)
    ok = n > nmin
    num = cr - sh * sl / np.maximum(n, 1.0)
    dh = shh - sh * sh / np.maximum(n, 1.0)
    dl = sll - sl * sl / np.maximum(n, 1.0)
    den = np.sqrt(np.maximum(dh, 0.0) * np.maximum(dl, 0.0)) + EPS
    ncc = np.where(ok & shift_window(), num / den, -2.0)
    i = int(np.argmax(ncc))
    dy, dx = divmod(i, PAD)
    if dy > PAD // 2:
        dy -= PAD
    if dx > PAD // 2:
        dx -= PAD
    return float(ncc.flat[i]), dy, dx


# ------------------------------------------------------------------ the search
def lo_centers_mm():
    m = zmeta(LO_ROOT)
    zmax_mm = m["shape"][0] * LO_VOX / 1000.0
    c0, c1 = HI_SLABS_MM[0], zmax_mm - 1.0
    return np.round(np.arange(c0, c1, ZGRID_MM), 4)


_LO_CACHE = {}


def build_lo(centers):
    prods, areas = {}, {}
    t0 = time.time()
    for i, c in enumerate(centers):
        img, mask = slab_image(LO_ROOT, LO_VOX, c)
        if img is None:
            continue
        feat, m = feature2d(img, mask)
        if feat is None or m.sum() < MIN_MASK:
            continue
        prods[c] = fwd_products(feat, m)
        areas[c] = float(m.sum())
        if i % 40 == 0:
            print(f"  lo {i}/{len(centers)} t={time.time()-t0:.0f}s", flush=True)
    return prods, areas


HI_FEATS = {}          # zc_mm -> (feat, mask)
LO_PRODS = {}
LO_AREAS = {}


def work_theta(theta):
    rows = []
    for k, zc in enumerate(HI_SLABS_MM):
        feat, mask = HI_FEATS[zc]
        rf = ndi.rotate(feat, theta, reshape=False, order=1)
        rm = ndi.rotate(mask.astype(np.float32), theta, reshape=False,
                        order=1) > 0.5
        rf = np.where(rm, rf, 0.0).astype(np.float32)
        hiF = fwd_products(rf, rm)
        area = hiF[3]
        for c, loF in LO_PRODS.items():
            s, dy, dx = masked_ncc_max(hiF, loF, area)
            rows.append((theta, k, c, s, dy, dx))
    return rows


def run():
    print("building hi slab features...", flush=True)
    for zc in HI_SLABS_MM:
        feat, mask = hi_slab_feature(zc)
        assert feat is not None, f"hi slab {zc} empty"
        HI_FEATS[zc] = (feat, mask)
        print(f"  hi slab {zc}mm mask={int(mask.sum())}", flush=True)

    print("building lo slab features (whole scroll)...", flush=True)
    centers = lo_centers_mm()
    prods, areas = build_lo(centers)
    LO_PRODS.update(prods)
    LO_AREAS.update(areas)
    print(f"  lo slabs usable: {len(LO_PRODS)}/{len(centers)}", flush=True)

    t0 = time.time()
    all_rows = []
    with ProcessPoolExecutor(max_workers=22) as ex:
        for j, rows in enumerate(ex.map(work_theta, THETAS)):
            all_rows.extend(rows)
            if j % 10 == 0:
                print(f"theta {j}/{len(THETAS)} t={time.time()-t0:.0f}s",
                      flush=True)
    arr = np.array(all_rows, np.float32)
    np.savez_compressed(OUT / "pass1_scores.npz", rows=arr,
                        slabs_mm=np.array(HI_SLABS_MM))
    print(f"saved {arr.shape} rows in {time.time()-t0:.0f}s", flush=True)
    aggregate(arr)


def aggregate(arr):
    """Combine per-slab rows into per-(flip,theta,zoff) candidates."""
    slabs = np.array(HI_SLABS_MM)
    span = slabs[0] + slabs[-1]          # flip: zc -> span - zc
    key = {}
    for t, k, c, s, dy, dx in arr:
        key[(round(float(t), 1), int(k), round(float(c), 4))] = (s, dy, dx)
    cands = []
    centers = sorted({round(float(c), 4) for _, _, c, _, _, _ in arr})
    cset = set(centers)
    zoffs = np.round(np.arange(centers[0] - slabs[-1],
                               centers[-1] - slabs[0] + ZGRID_MM, ZGRID_MM), 4)
    for flip in (0, 1):
        smap = slabs if not flip else span - slabs
        for th in THETAS:
            for zo in zoffs:
                got = []
                for k in range(len(slabs)):
                    c = round(float(zo + smap[k]), 4)
                    v = key.get((round(float(th), 1), k, c))
                    if v is not None and v[0] > -1.5:
                        got.append((k, *v))
                if len(got) < 4:
                    continue
                sc = np.array([g[1] for g in got])
                dys = np.array([g[2] for g in got])
                dxs = np.array([g[3] for g in got])
                # allow a linear drift of shift vs slab index (tilt), penalize
                # residual scatter
                kk = np.array([g[0] for g in got], np.float32)
                ry = dys - np.poly1d(np.polyfit(kk, dys, 1))(kk)
                rx = dxs - np.poly1d(np.polyfit(kk, dxs, 1))(kk)
                scat = float(np.sqrt((ry ** 2 + rx ** 2).mean()))
                cands.append(dict(flip=flip, theta=float(th), zoff=float(zo),
                                  n=len(got), score=float(sc.mean()),
                                  smin=float(sc.min()), scatter=scat,
                                  dy=float(dys.mean()), dx=float(dxs.mean())))
    cands.sort(key=lambda d: -d["score"])
    allsc = np.array([d["score"] for d in cands])
    stats = dict(n_cand=len(cands), mean=float(allsc.mean()),
                 std=float(allsc.std()),
                 zscore_top=float((allsc.max() - allsc.mean()) / allsc.std()))
    json.dump(dict(stats=stats, top=cands[:40]),
              open(OUT / "pass1_top.json", "w"), indent=1)
    print("stats", stats, flush=True)
    for d in cands[:10]:
        print(d, flush=True)


# ------------------------------------------------------------------- selftest
def selftest():
    """Cut a rotated copy of a lo slab and confirm the pipeline recovers it.

    Uses the SAME rotation code path as work_theta (rotate hi by +theta), so
    a synthetic hi made by rotating lo data by th_true must be recovered at
    theta = (360 - th_true) % 360.
    """
    zc_true, th_true = 60.0, 37.0
    img, mask = slab_image(LO_ROOT, LO_VOX, zc_true)
    feat, m = feature2d(img, mask)
    # crop a hi-sized window, off-center by a known amount
    cy, cx = 414 + 30, 414 - 22
    hw, hh = 244, 272                      # ~ hi extent at 74.9 um
    sub = feat[cy - hh:cy + hh, cx - hw:cx + hw].copy()
    subm = m[cy - hh:cy + hh, cx - hw:cx + hw].copy()
    rf = ndi.rotate(sub, th_true, reshape=False, order=1)
    rm = ndi.rotate(subm.astype(np.float32), th_true, reshape=False,
                    order=1) > 0.5
    rf = np.where(rm, rf, 0.0).astype(np.float32)
    los = {}
    for zc in (52.0, 56.0, 60.0, 64.0):
        im2, mk2 = slab_image(LO_ROOT, LO_VOX, zc)
        f2, m2 = feature2d(im2, mk2)
        los[zc] = fwd_products(f2, m2)
    print("selftest: scanning theta x a few z centers...", flush=True)
    best = None
    for th in np.arange(0, 360, 3.0):
        rrf = ndi.rotate(rf, th, reshape=False, order=1)
        rrm = ndi.rotate(rm.astype(np.float32), th, reshape=False,
                         order=1) > 0.5
        rrf = np.where(rrm, rrf, 0.0).astype(np.float32)
        hF = fwd_products(rrf, rrm)
        for zc, loF in los.items():
            s, dy, dx = masked_ncc_max(hF, loF, hF[3])
            if best is None or s > best[0]:
                best = (s, th, zc, dy, dx)
    print("selftest best:", best, flush=True)
    print(f"expected: theta~{(360 - th_true) % 360}, zc~{zc_true}, "
          f"dy~{cy - hh}, dx~{cx - hw}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "selftest":
        selftest()
    else:
        run()
