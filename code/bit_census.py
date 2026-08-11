#!/usr/bin/env python3
"""Full census of every bit plane in both shipped label volumes.

This is the description that should have shipped with the datasets: how
much of each volume each bit actually covers, plus the containment
relations, so nobody has to scan the tarball to find out.

Reads labels only, no prediction volumes.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
from eval_surface_pred import read_box

VOLS = [
    ("PHerc0139", "/root/fl_probe/reg0139/reg2/labels0139_L1.zarr",
     ["valid", "material", "centerline", "recto_band"]),
    ("PHerc1203", "/root/fl_probe/reg1203/reg/labels1203_L1.zarr",
     ["valid", "material", "centerline", "recto_band", "boundary_poor"]),
]
BITS = {"valid": 1, "material": 2, "centerline": 4,
        "recto_band": 8, "boundary_poor": 16}
STEP = 96

out = {}
for name, path, names in VOLS:
    meta = json.loads((Path(path) / ".zarray").read_text())
    Z, Y, X = meta["shape"]
    counts = {n: 0 for n in names}
    # containment probes
    extra = dict(material_not_valid=0, centerline_not_material=0,
                 recto_not_material=0, bp_not_material=0, bp_not_valid=0)
    total = 0
    for z0 in range(0, Z, STEP):
        z1 = min(z0 + STEP, Z)
        lab = read_box(path, z0, z1, 0, Y, 0, X)
        total += lab.size
        planes = {n: (lab & BITS[n]) > 0 for n in names}
        for n in names:
            counts[n] += int(planes[n].sum())
        extra["material_not_valid"] += int(
            (planes["material"] & ~planes["valid"]).sum())
        extra["centerline_not_material"] += int(
            (planes["centerline"] & ~planes["material"]).sum())
        extra["recto_not_material"] += int(
            (planes["recto_band"] & ~planes["material"]).sum())
        if "boundary_poor" in planes:
            extra["bp_not_material"] += int(
                (planes["boundary_poor"] & ~planes["material"]).sum())
            extra["bp_not_valid"] += int(
                (planes["boundary_poor"] & ~planes["valid"]).sum())
        print(f"  {name} z {z0}-{z1}", file=sys.stderr, flush=True)

    v = max(counts["valid"], 1)
    m = max(counts["material"], 1)
    rec = dict(window_voxels=total,
               valid=counts["valid"],
               valid_pct_of_window=round(100 * counts["valid"] / total, 2),
               material=counts["material"],
               material_pct_of_valid=round(100 * counts["material"] / v, 2))
    for n in names:
        if n in ("valid", "material"):
            continue
        rec[n] = counts[n]
        rec[f"{n}_pct_of_material"] = round(100 * counts[n] / m, 2)
    rec["containment"] = extra
    out[name] = rec
    print(f"{name} done", file=sys.stderr, flush=True)

print(json.dumps(out, indent=1))
