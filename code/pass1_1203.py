#!/usr/bin/env python3
"""PHerc1203 coarse registration: drive the proven pass1 machinery with
1203's geometry.

hi = 20260319130212 (2.403 um, full cross-section x 36.4 mm), slabs from L5
     (76.896 um/vox, shape ~473 x 828 x 828)
lo = 20250820131727 (9.362 um whole scroll), slabs from L3 (74.896 um)

Differences from 0139: hi_L5 voxel is now ~equal to lo_L3 voxel (zoom 1.027
instead of 0.482), the hi footprint is nearly the full cross-section, and
PAD goes to 1280 so the +-270 vox shift window stays linear-exact for the
larger feature images (850+856-1 > 1024).
"""
import sys

sys.path.insert(0, "/root/fl_probe/reg0139/reg2")
import pass1
from pathlib import Path

BASE = Path("/root/fl_probe/reg1203")
pass1.BASE = BASE
pass1.OUT = BASE / "reg"
pass1.OUT.mkdir(exist_ok=True)
pass1.LO_ROOT, pass1.LO_VOX = BASE / "lo_L3", 74.896
pass1.HI_ROOT, pass1.HI_VOX = BASE / "hi_L5", 76.896
pass1.PAD = 1280
pass1.SHIFT_MAX = 270
pass1.HI_SLABS_MM = [4.0, 11.2, 18.4, 25.6, 32.8]   # symmetric, 0.8 grid
pass1._WIN = None            # rebuild shift window for the new PAD

if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "selftest":
        pass1.selftest()
    else:
        pass1.run()
