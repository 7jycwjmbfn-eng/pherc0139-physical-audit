#!/usr/bin/env python3
"""Pass 10: package the physical truth as a ready-to-use label volume.

Output: a zarr (zstd, chunks 128^3) on the lo-frame L1 grid (18.724 um),
uint8 bit flags per voxel:
  bit0 (1)  valid       inside the mapped 1.129 um ROI (eroded 3 vox)
  bit1 (2)  material    papyrus sheet material (threshold 65, see report)
  bit2 (4)  centerline  per-slice sheet centerline (ridge of the 2D EDT)
  bit3 (8)  recto_band  inward-facing material boundary, 1-vox dilated

Grid: lo volume 20250728140407 level-1 voxel indices; the array is stored
with an `origin_l1` offset so voxel (z,y,x) here = lo L1 voxel origin+(z,y,x).
Images are NOT shipped: the lo CT volume is public; README points at it.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

import sys
sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
from pass5b import Y1_0, X1_0, ERODE, HI4, OUT, read_zrange, zmeta, ZCH
from pass7b import normal_field

TH = 65
DST = Path("/root/fl_probe/reg0139/reg2/labels0139_L1.zarr")
BLOSC = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=1)
CH = 128
HI = MINV = OFFV = None
SHAPE = None            # (z,y,x) of the packaged window, L1 vox


def slab_work(iz):
    origin = np.array([iz * 96, Y1_0, X1_0], float)
    aoff = MINV @ (origin - OFFV)
    out_shape = (96, SHAPE[1], SHAPE[2])
    hiT = ndi.affine_transform(HI, MINV, offset=aoff,
                               output_shape=out_shape, order=1)
    valid = ndi.binary_erosion(hiT > 0, iterations=ERODE)
    material = (hiT > TH) & valid
    flags = valid.astype(np.uint8)
    flags |= material.astype(np.uint8) << 1
    for k in range(out_shape[0]):
        tb = material[k]
        if tb.sum() < 500:
            continue
        dt = ndi.distance_transform_edt(tb)
        ctr = tb & (dt >= ndi.maximum_filter(dt, 3)) & (dt >= 1)
        flags[k] |= ctr.astype(np.uint8) << 2
        # recto band
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
    # write chunks for this slab: slab z = [iz*96-Z0 .. ) in local coords
    z_loc = iz * 96 - ZCH[0] * 96
    for cz in range(z_loc // CH, -(-(z_loc + 96) // CH)):
        for cyc in range(-(-SHAPE[1] // CH)):
            for cxc in range(-(-SHAPE[2] // CH)):
                # chunk z range in local coords
                bz0, bz1 = cz * CH, min((cz + 1) * CH, SHAPE[0])
                s0, s1 = max(bz0, z_loc), min(bz1, z_loc + 96)
                if s1 <= s0:
                    continue
                by1 = min((cyc + 1) * CH, SHAPE[1])
                bx1 = min((cxc + 1) * CH, SHAPE[2])
                blk = np.zeros((bz1 - bz0, CH, CH), np.uint8)
                part = flags[s0 - z_loc:s1 - z_loc,
                             cyc * CH:by1, cxc * CH:bx1]
                blk[s0 - bz0:s1 - bz0, :part.shape[1], :part.shape[2]] = part
                f = DST / str(cz) / str(cyc) / str(cxc)
                f.parent.mkdir(parents=True, exist_ok=True)
                tag = f.with_suffix(f".part{iz}")
                np.save(tag, blk)          # merge pass combines partial z
    return iz


def merge_chunks():
    """Combine per-slab partials (slabs share z chunks at boundaries)."""
    n = 0
    for d in sorted(DST.rglob("*.part*.npy")):
        base = d.name.split(".part")[0]
        final = d.parent / base
        blk = np.load(d)
        if final.is_file():
            old = np.frombuffer(BLOSC.decode(final.read_bytes()),
                                np.uint8).reshape(blk.shape)
            blk = blk | old
        final.write_bytes(BLOSC.encode(np.ascontiguousarray(blk)))
        d.unlink()
        n += 1
    return n


def main():
    global HI, MINV, OFFV, SHAPE
    t0 = time.time()
    f3 = np.load(OUT / "pass3_final.npz")
    M2, t2 = f3["M2"], f3["t2"]
    MINV = np.linalg.inv(M2)
    OFFV = 2.0 * t2 + 0.5
    HI = read_zrange(HI4, 0, zmeta(HI4)["shape"][0])
    # window: z slabs 18..30 of 96, y/x same box as the audits
    nz = (ZCH[-1] - ZCH[0] + 1) * 96
    SHAPE = (nz, 2304, 2208)     # same box as the audit slabs (pass5b)
    print(f"window {SHAPE}, origin L1 = ({ZCH[0]*96}, {Y1_0}, {X1_0})",
          flush=True)
    DST.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        for iz in ex.map(slab_work, list(ZCH)):
            print(f"slab {iz} done t={time.time()-t0:.0f}s", flush=True)
    n = merge_chunks()
    meta = dict(
        zarr_format=2, shape=list(SHAPE), chunks=[CH, CH, CH],
        dtype="|u1", fill_value=0, order="C", dimension_separator="/",
        compressor=dict(id="blosc", cname="zstd", clevel=5, shuffle=1,
                        blocksize=0),
        filters=None)
    (DST / ".zarray").write_text(json.dumps(meta, indent=1))
    attrs = dict(
        description="Physical truth labels for PHerc0139, cast from the "
                    "1.129 um scan into the 9.362 um frame (level-1 grid)",
        grid="lo volume 20250728140407, level 1 (18.724 um voxels)",
        origin_l1=[ZCH[0] * 96, Y1_0, X1_0],
        bits=dict(valid=1, material=2, centerline=4, recto_band=8),
        material_threshold=65,
        registration_heldout_um=4.09,
        source_hi_volume="20260413113053 (1.129 um)")
    (DST / ".zattrs").write_text(json.dumps(attrs, indent=1))
    import subprocess
    sz = subprocess.run(["du", "-sh", str(DST)], capture_output=True,
                        text=True).stdout.split()[0]
    print(f"merged {n} chunks, total {sz}, t={time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
