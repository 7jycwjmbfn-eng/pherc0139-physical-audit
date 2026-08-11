#!/usr/bin/env python3
"""Did the sparse-prediction guard ever fire on the published m7 volumes?

Two guards share the same second clause across this project:

  evaluator      if ctr.sum() < 500 or pb.sum() < 100: continue
  pass5/6/7/8/9  if  tb.sum() < 500 or pb.sum() < 100: continue

The pb clause takes a slice's truth out of the denominator when the
prediction is sparse there, which flatters sparse predictions. TAUIL-Abd-Elilah
raised it on the villa PR. The question this answers is not whether the guard
is wrong, it is: on the m7 volumes we actually published numbers for, did it
ever fire?

Counts, per scroll, slices where the truth clause passes but pb.sum() < 100,
under both readings of the truth clause, plus how many centerline points sat
in them. Zero means every published number stands unchanged.

Reads the shipped labels and the prediction volume. No distance transforms,
so it is much cheaper than a full evaluation.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
from eval_surface_pred import read_box

VOLS = [
    ("PHerc0139", "/root/fl_probe/reg0139/reg2/labels0139_L1.zarr",
     "/root/fl_probe/reg0139/pred_L0"),
    ("PHerc1203", "/root/fl_probe/reg1203/reg/labels1203_L1.zarr",
     "/root/fl_probe/reg1203/pred_L0"),
]
STEP = 96

out = {}
for name, labp, predp in VOLS:
    attrs = json.loads((Path(labp) / ".zattrs").read_text())
    oz, oy, ox = attrs["origin_l1"]
    meta = json.loads((Path(labp) / ".zarray").read_text())
    Z, Y, X = meta["shape"]

    rec = dict(slices_seen=0, ctr_ge500=0, tb_ge500=0,
               ctr_ge500_and_pb_lt100=0, tb_ge500_and_pb_lt100=0,
               ctr_points_at_risk=0, min_pb_where_ctr_ge500=None,
               min_pb_where_tb_ge500=None)
    for z0 in range(0, Z, STEP):
        z1 = min(z0 + STEP, Z)
        lab = read_box(labp, z0, z1, 0, Y, 0, X)
        gz0, gz1 = oz + z0, oz + z1
        p0 = read_box(predp, 2 * gz0, 2 * gz1, 2 * oy, 2 * (oy + Y),
                      2 * ox, 2 * (ox + X))
        pred = p0.reshape(z1 - z0, 2, Y, 2, X, 2).max((1, 3, 5)) > 0
        del p0
        valid = (lab & 1) > 0
        material = (lab & 2) > 0
        for k in range(0, lab.shape[0], 4):
            pb = int((pred[k] & valid[k]).sum())
            nctr = int(((lab[k] & 4) > 0).sum())
            ntb = int(material[k].sum())
            rec["slices_seen"] += 1
            if nctr >= 500:
                rec["ctr_ge500"] += 1
                m = rec["min_pb_where_ctr_ge500"]
                rec["min_pb_where_ctr_ge500"] = pb if m is None else min(m, pb)
                if pb < 100:
                    rec["ctr_ge500_and_pb_lt100"] += 1
                    rec["ctr_points_at_risk"] += nctr
            if ntb >= 500:
                rec["tb_ge500"] += 1
                m = rec["min_pb_where_tb_ge500"]
                rec["min_pb_where_tb_ge500"] = pb if m is None else min(m, pb)
                if pb < 100:
                    rec["tb_ge500_and_pb_lt100"] += 1
        print(f"  {name} z {z0}-{z1}", file=sys.stderr, flush=True)
    out[name] = rec
    print(f"{name} done", file=sys.stderr, flush=True)

print(json.dumps(out, indent=1))
print()
clean = True
for name, r in out.items():
    a, b = r["ctr_ge500_and_pb_lt100"], r["tb_ge500_and_pb_lt100"]
    print(f"{name}: evaluator path {a} slices, audit-script path {b} slices, "
          f"centerline points at risk {r['ctr_points_at_risk']:,}")
    print(f"    smallest prediction on any scored slice: "
          f"{r['min_pb_where_ctr_ge500']:,} (evaluator), "
          f"{r['min_pb_where_tb_ge500']:,} (audit scripts), against a "
          f"threshold of 100")
    clean &= (a == 0 and b == 0)
print()
print("GUARD NEVER FIRED, every published number stands" if clean
      else "GUARD FIRED, published numbers need recomputing")
