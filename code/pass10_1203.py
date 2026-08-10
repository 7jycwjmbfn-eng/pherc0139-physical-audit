#!/usr/bin/env python3
"""Package PHerc1203 physical truth labels (lo L1 grid, streamed).

uint8 bits: 1 valid | 2 material | 4 centerline | 8 recto_band |
            16 boundary_poor (tile gap-visibility < 0.4 at 19 um: the
            physical 'no resolvable boundary nearby' map)
"""
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

import sys
sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
from pass1 import read_zrange
from pass7b import normal_field
sys.path.insert(0, "/root/fl_probe/reg1203")
import pass5b_1203 as S

OUT = Path("/root/fl_probe/reg1203/reg")
DST = Path("/root/fl_probe/reg1203/reg/labels1203_L1.zarr")
BLOSC = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=1)
CH = 128
SHAPE = (len(S.ZCH) * 96, 3456, 3456)
Z_ORIGIN = S.ZCH[0] * 96


def slab_work(iz):
    z1_0 = iz * 96
    pts = np.array([[z1_0, 0, 0], [z1_0 + 96, 3400, 3400]], float)
    hz = (S.Minv1 @ pts.T).T[:, 0] + S.off1[0]
    hz0 = max(int(min(hz)) - 4, 0)
    hi = read_zrange(S.HI3, hz0, int(max(hz)) + 5)
    aoff = (S.off1 + np.array([z1_0, 0, 0]) @ S.Minv1.T
            - np.array([hz0, 0, 0]))
    hiT = ndi.affine_transform(hi, S.Minv1, offset=aoff,
                               output_shape=(96, SHAPE[1], SHAPE[2]),
                               order=1)
    del hi
    valid = ndi.binary_erosion(hiT > 0, iterations=S.ERODE)
    material = (hiT > S.TH) & valid
    flags = valid.astype(np.uint8)
    flags |= material.astype(np.uint8) << 1
    for k in range(96):
        tb = material[k]
        if tb.sum() < 500:
            continue
        dt = ndi.distance_transform_edt(tb)
        ctr = tb & (dt >= ndi.maximum_filter(dt, 3)) & (dt >= 1)
        flags[k] |= ctr.astype(np.uint8) << 2
        g = ndi.gaussian_filter(hiT[k].astype(np.float32), 1.5)
        nyf, nxf, coh = normal_field(g)
        YY, XX = np.mgrid[0:tb.shape[0], 0:tb.shape[1]].astype(np.float32)
        m = np.nonzero(tb)
        cy, cx = m[0].mean(), m[1].mean()
        sgn = np.sign(nyf * (YY - cy) + nxf * (XX - cx))
        sgn[sgn == 0] = 1.0
        in_air = map_coordinates(tb.astype(np.float32),
                                 [YY - 1.5 * nyf * sgn,
                                  XX - 1.5 * nxf * sgn], order=1) < 0.5
        recto = ndi.binary_dilation(tb & in_air, iterations=1)
        flags[k] |= (recto & valid[k]).astype(np.uint8) << 3
        num = ndi.uniform_filter((tb & (dt < 3)).astype(np.float32), 64)
        den = ndi.uniform_filter(tb.astype(np.float32), 64)
        gv = num / np.maximum(den, 1e-6)
        flags[k] |= ((gv < 0.40) & tb).astype(np.uint8) << 4
    z_loc = z1_0 - Z_ORIGIN
    for cz in range(z_loc // CH, -(-(z_loc + 96) // CH)):
        bz0, bz1 = cz * CH, min((cz + 1) * CH, SHAPE[0])
        s0, s1 = max(bz0, z_loc), min(bz1, z_loc + 96)
        if s1 <= s0:
            continue
        for cy_ in range(-(-SHAPE[1] // CH)):
            for cx_ in range(-(-SHAPE[2] // CH)):
                by1 = min((cy_ + 1) * CH, SHAPE[1])
                bx1 = min((cx_ + 1) * CH, SHAPE[2])
                blk = np.zeros((CH, CH, CH), np.uint8)
                part = flags[s0 - z_loc:s1 - z_loc,
                             cy_ * CH:by1, cx_ * CH:bx1]
                if not part.any():
                    continue
                blk[s0 - bz0:s1 - bz0, :part.shape[1],
                    :part.shape[2]] = part
                f = DST / str(cz) / str(cy_) / str(cx_)
                f.parent.mkdir(parents=True, exist_ok=True)
                np.save(f.with_suffix(f".part{iz}"), blk)
    return iz


def main():
    t0 = time.time()
    f = np.load(OUT / "pass3_final.npz")
    S.M2 = f["M2"]
    S.t2 = 2.0 * f["t2"] + 0.5
    S.Minv1 = np.linalg.inv(S.M2)
    S.off1 = -(S.Minv1 @ S.t2)
    mid = read_zrange(S.HI3, 832, 833)[0]
    v = mid[mid > 0]
    h = ndi.gaussian_filter1d(np.bincount(v, minlength=256).astype(float), 3)
    lo_pk = int(np.argmax(h[:100]))
    S.TH = int(lo_pk + np.argmin(
        h[lo_pk:100 + int(np.argmax(h[100:]))]))
    print(f"TH={S.TH}", flush=True)
    DST.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as ex:
        for iz in ex.map(slab_work, list(S.ZCH)):
            print(f"slab {iz} t={time.time()-t0:.0f}s", flush=True)
    n = 0
    for d in sorted(DST.rglob("*.part*.npy")):
        base = d.name.split(".part")[0]
        final = d.parent / base
        blk = np.load(d)
        if final.is_file():
            blk = blk | np.frombuffer(
                BLOSC.decode(final.read_bytes()), np.uint8).reshape(blk.shape)
        final.write_bytes(BLOSC.encode(np.ascontiguousarray(blk)))
        d.unlink()
        n += 1
    (DST / ".zarray").write_text(json.dumps(dict(
        zarr_format=2, shape=list(SHAPE), chunks=[CH, CH, CH],
        dtype="|u1", fill_value=0, order="C", dimension_separator="/",
        compressor=dict(id="blosc", cname="zstd", clevel=5, shuffle=1,
                        blocksize=0), filters=None), indent=1))
    (DST / ".zattrs").write_text(json.dumps(dict(
        description="Physical truth labels for PHerc1203 (Grand Prize "
                    "scroll), cast from the 2.403 um scan into the "
                    "9.362 um frame (level-1 grid)",
        grid="lo volume 20250820131727, level 1 (18.724 um voxels)",
        origin_l1=[Z_ORIGIN, 0, 0],
        bits=dict(valid=1, material=2, centerline=4, recto_band=8,
                  boundary_poor=16),
        material_threshold=int(S.TH),
        registration_heldout_um=2.38,
        source_hi_volume="20260319130212 (2.403 um)"), indent=1))
    sz = subprocess.run(["du", "-sh", str(DST)], capture_output=True,
                        text=True).stdout.split()[0]
    print(f"merged {n} chunks, {sz}, t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
