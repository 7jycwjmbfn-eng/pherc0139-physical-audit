#!/usr/bin/env python3
"""Pass 3: re-match under the corrected pose, then validate.

1. Compose the pass2 rigid+scale correction into the forward map.
2. Re-run block matching (displacements should now be centered near zero).
3. Lattice-average the displacement field; split-half (even/odd z lattice
   cells) cross-validation: fit the deformation lattice on half the blocks,
   predict the other half.  The held-out error is the honest label-casting
   accuracy; the |D| median without the lattice is the rigid-only baseline.
4. Save the composed transform, lattice field, and diagnostic PNGs.
"""
import json
from pathlib import Path

import numpy as np
from concurrent.futures import ProcessPoolExecutor

import sys
sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
import pass2
from pass2 import BLK, STRIDE, LO2_VOX, LO2_Z0, LO2_Z1, read_zrange, LO2_ROOT

OUT = Path("/root/fl_probe/reg0139/reg2")
LAT = 96                       # lattice spacing, L2 vox (~3.6 mm)


def skew(sol):
    tz, ty, tx, rz, ry, rx, s = sol
    return (np.array([[s, -rx, ry], [rx, s, -rz], [-ry, rz, s]]),
            np.array([tz, ty, tx]))


def main():
    f = np.load(OUT / "pass2_field.npz")
    sol, ctr, M, t = f["sol"], f["ctr"], f["M"], f["t"]
    Wm, tsol = skew(sol)
    W = np.eye(3) + Wm
    M2 = W @ M
    t2 = W @ t + tsol - Wm @ ctr

    print("resampling under corrected pose...", flush=True)
    pass2.LO = read_zrange(LO2_ROOT, LO2_Z0, LO2_Z1)
    hiT, msk, box0 = pass2.resample_hi_to_L2(M2, t2)
    pass2.HIT, pass2.MSK, pass2.BOX0 = hiT, msk, box0

    jobs = []
    for bz in range(0, hiT.shape[0] - BLK, STRIDE):
        for by in range(0, hiT.shape[1] - BLK, STRIDE):
            for bx in range(0, hiT.shape[2] - BLK, STRIDE):
                jobs.append((bz, by, bx))
    res = []
    with ProcessPoolExecutor(max_workers=20) as ex:
        for r in ex.map(pass2.match_block, jobs, chunksize=16):
            if r is not None:
                res.append(r)
    res = np.array(res)
    P, D, NC = res[:, :3], res[:, 3:6], res[:, 6]
    dn = np.linalg.norm(D, axis=1)
    print(f"rematch: {len(res)} blocks ncc_med={np.median(NC):.3f} "
          f"|d| med={np.median(dn):.3f} p95={np.percentile(dn,95):.3f} vox",
          flush=True)

    # residual global fit must now be ~zero
    sol2, keep = pass2.robust_fit(P - P.mean(0), D)
    print("residual rigid fit (should be ~0):",
          np.round(sol2, 5).tolist(), flush=True)

    # ---- lattice field + split-half CV
    cell = np.floor(P / LAT).astype(int)
    cmin = cell.min(0)
    cidx = cell - cmin
    csh = cidx.max(0) + 1

    def build_lattice(mask_rows):
        sums = np.zeros((*csh, 3))
        wts = np.zeros(csh)
        for i in np.where(mask_rows)[0]:
            c = tuple(cidx[i])
            sums[c] += D[i] * NC[i]
            wts[c] += NC[i]
        lat = np.full((*csh, 3), np.nan)
        ok = wts > 1.0
        lat[ok] = sums[ok] / wts[ok, None]
        return lat, ok

    def interp(lat, ok, pts):
        # fill NaN cells by nearest valid, then trilinear
        from scipy import ndimage as ndi
        filled = lat.copy()
        if (~ok).any():
            ind = ndi.distance_transform_edt(
                ~ok, return_distances=False, return_indices=True)
            filled = lat[tuple(ind)]
        g = pts / LAT - cmin - 0.5
        out = np.empty((len(pts), 3))
        from scipy.ndimage import map_coordinates
        for a in range(3):
            out[:, a] = map_coordinates(filled[..., a], g.T, order=1,
                                        mode="nearest")
        return out

    zc = cidx[:, 0]
    A, B = (zc % 2 == 0), (zc % 2 == 1)
    latA, okA = build_lattice(A)
    pred = interp(latA, okA, P[B])
    err = np.linalg.norm(D[B] - pred, axis=1)
    base = np.linalg.norm(D[B], axis=1)
    cv = dict(
        n_test=int(B.sum()),
        rigid_only_med_um=float(np.median(base) * LO2_VOX),
        rigid_only_p95_um=float(np.percentile(base, 95) * LO2_VOX),
        heldout_med_um=float(np.median(err) * LO2_VOX),
        heldout_p95_um=float(np.percentile(err, 95) * LO2_VOX))
    print("split-half CV:", json.dumps(cv, indent=1), flush=True)

    # full lattice from all blocks (the deliverable deformation field)
    latF, okF = build_lattice(np.ones(len(P), bool))

    # smoothness diagnostic: variogram
    rng = np.random.default_rng(0)
    ii = rng.integers(0, len(P), 20000)
    jj = rng.integers(0, len(P), 20000)
    dist = np.linalg.norm(P[ii] - P[jj], axis=1) * LO2_VOX / 1000.0
    dv = np.linalg.norm(D[ii] - D[jj], axis=1) * LO2_VOX
    bins = np.array([0.5, 1, 2, 4, 8, 16, 32])
    vario = []
    for k in range(len(bins) - 1):
        m = (dist >= bins[k]) & (dist < bins[k + 1])
        if m.sum() > 50:
            vario.append((float(bins[k]), float(np.median(dv[m])),
                          int(m.sum())))
    print("variogram (mm, median |dD| um):", vario, flush=True)

    np.savez_compressed(
        OUT / "pass3_final.npz", P=P, D=D, NCC=NC, M2=M2, t2=t2,
        lattice=latF, lattice_ok=okF, lat_spacing=LAT, lat_cmin=cmin,
        box0=box0)
    json.dump(dict(
        transform=dict(
            frame="lo_L2 37.448um voxel grid (z,y,x); p_L2 = M2 @ p_hiL5 + t2",
            M2=M2.tolist(), t2=t2.tolist(),
            note="hi frame = hi_L5 voxel index (1.129um vol level5, 36.128um)"),
        rematch=dict(n=int(len(res)), ncc_med=float(np.median(NC)),
                     d_med_um=float(np.median(dn) * LO2_VOX),
                     d_p95_um=float(np.percentile(dn, 95) * LO2_VOX)),
        residual_rigid=np.round(sol2, 6).tolist(),
        cv=cv, variogram=vario),
        open(OUT / "pass3_report.json", "w"), indent=1)

    # quiver + z-profile PNGs
    try:
        from PIL import Image, ImageDraw
        zmid = (P[:, 0] > np.median(P[:, 0]) - 50) & \
               (P[:, 0] < np.median(P[:, 0]) + 50)
        sc = 0.5
        img = Image.new("RGB", (830, 830), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        for p, d in zip(P[zmid], D[zmid]):
            y, x = p[1] / 2, p[2] / 2
            dy, dx = d[1] * 40 * sc, d[2] * 40 * sc
            dr.line([(x, y), (x + dx, y + dy)], fill=(200, 30, 30), width=1)
            dr.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(30, 30, 200))
        img.save(OUT / "pass3_quiver_midz.png")
    except Exception as e:
        print("quiver png skipped:", e, flush=True)


if __name__ == "__main__":
    main()
