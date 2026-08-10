#!/usr/bin/env python3
"""PHerc1203 compressed-region survey (the fork decider).

Same instrument as 0139's pass4b (structure-tensor oriented profile peak
spacing with full outcome accounting), run on hi_L3 (19.226 um) slices.
"""
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
import pass4b
from pass1 import read_zrange

BASE = Path("/root/fl_probe/reg1203")
HI3, VOX = BASE / "hi_L3", 19.2264
pass4b.VOX = VOX                      # spacing conversions inside the tool

ZC = [192, 448, 832, 1088, 1344, 1728]     # centers of the fetched planes
WIN, STRIDE, MASK_MIN = 96, 48, 0.6
OK, D_COH, D_FLAT, D_PK = 0, 1, 2, 3

rows = []
for zc in ZC:
    sl = read_zrange(HI3, zc - 1, zc + 2).mean(0)
    msk = sl > 0
    Y, X = sl.shape
    for y0 in range(0, Y - WIN, STRIDE):
        for x0 in range(0, X - WIN, STRIDE):
            wm = msk[y0:y0 + WIN, x0:x0 + WIN]
            if wm.mean() < MASK_MIN:
                continue
            o, sp = pass4b.window_outcome(sl[y0:y0 + WIN, x0:x0 + WIN])
            rows.append((zc, y0 + WIN / 2, x0 + WIN / 2, o, sp))
    print(f"zc={zc} cum={len(rows)}", flush=True)

rows = np.array(rows)
o = rows[:, 3]
sp = rows[o == OK, 4]
stats = dict(
    vox_um=VOX,
    n_tissue=int(len(rows)),
    frac_ok=float((o == OK).mean()),
    frac_drop_coher=float((o == D_COH).mean()),
    frac_drop_flat=float((o == D_FLAT).mean()),
    frac_drop_peaks=float((o == D_PK).mean()),
    spacing_median_um=float(np.median(sp)),
    spacing_p10_um=float(np.percentile(sp, 10)),
    spacing_p5_um=float(np.percentile(sp, 5)),
    frac_below_100um=float((sp < 100).mean()),
    frac_below_130um=float((sp < 130).mean()),
    frac_below_65um=float((sp < 65).mean()))
print(json.dumps(stats, indent=1), flush=True)
json.dump(stats, open(BASE / "reg" / "survey_stats.json", "w"), indent=1)
np.savez_compressed(BASE / "reg" / "survey_rows.npz", rows=rows)
