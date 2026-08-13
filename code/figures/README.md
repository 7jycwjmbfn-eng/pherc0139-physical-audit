# Figures

`figures/fig_reference_<scroll>.png` is what the reference is. Left: the
9.362 um scan the surface models run on. Middle: the same field with
surface-m7's output in green and the scored stretches in red. Right: the
second scan of the same scroll (1.129 um on PHerc0139, 2.403 um on
PHerc1203), put into the working frame by the transform in
`results/pass3_final.npz`. Individual sheets can be followed from the left
panel into the right, so the registration is checkable by eye before any
number is read.

`figures/fig_missed_<scroll>.png` is the scoring. Red marks a 1.2 mm sheet
stretch scored fully missed, blue one scored covered, both in the same crop
and slice so the two sit under the same conditions.

## Rebuilding

From the repository root, with the label stores extracted somewhere:

    export LABELS_0139=/path/to/labels0139_L1.zarr
    export LABELS_1203=/path/to/labels1203_L1.zarr

    python code/figures/analyze.py 0139 2056
    python code/figures/render4.py 0139 2056 fig_missed_0139
    python code/figures/render5.py 0139 2056 1075 1860 70 fig_reference_0139

    python code/figures/analyze.py 1203 5736
    python code/figures/render4.py 1203 5736 fig_missed_1203 130
    python code/figures/render5.py 1203 5736 2120 1758 130 fig_reference_1203

`analyze.py` writes one `.npz` per slice and the renderers read it; both land
in `$SLICES`, which defaults to this directory. Volume chunks are pulled from
the open-data bucket over HTTP and cached under `$CHUNK_CACHE`, default
`code/figures/cache`. The first slice of PHerc1203 pulls about 600 MB.

## What the rebuild checks

`analyze.py` does not read the audit's stored per-arc results. It re-derives
the scored stretches from the published labels and the published prediction
volume, using the same rule as `code/eval_surface_pred.py`: centerline cut
into 1.2 mm tiles, stretches of at least 20 points, covered when at least half
the points have a prediction within 2 voxels, fully missed when under a tenth
do.

On PHerc0139 it then checks every stretch it calls fully missed against the
per-arc list in `results/pass6_gone_pts.npy`. Six slices (1892, 2056, 2188,
2284, 2440, 2636), 172 stretches, 172 matched.

PHerc1203 ships no per-arc list, so the only check available there is the
rate. Seven slices (4140, 4444, 4740, 5004, 5236, 5460, 5736) give 29 fully
missed of 257 scored stretches, 11.3 percent, against the 12.67 percent in
`results/1203/eval_selfcheck_1203.json`. That is a weaker check than the
PHerc0139 one and is not offered as anything more.

Some z carry no reference at all: the fine scans are mosaics and have thin
gaps between tiles. `analyze.py` skips those slices for the same reason
`eval_surface_pred.py` does, under 500 centerline points, and says so rather
than reporting an empty slice as a clean one.

## One thing worth knowing if you reuse the fetcher

A chunk the store genuinely does not have and a chunk whose request failed are
kept apart. Both would land in the array as zeros, and zeros in the prediction
volume read as the model having marked nothing there, which is the finding
this code exists to measure. A dropped request would manufacture missed
sheets, and only in the direction that flatters the result. `fetch_chunks`
raises on the second case instead.

## Selection rule

The rows in `fig_missed_*` are the two longest fully-missed stretches in the
slice that (a) have a covered stretch in the same crop, so red and blue are
always shown under the same conditions, and (b) whose crop is at least 60
percent inside the scanned volume, so no panel is half empty. Nothing is
chosen by eye. The crop half-width is 70 working voxels on PHerc0139 and 130
on PHerc1203, because PHerc1203 carries about a tenth the scored stretches per
slice and a smaller crop often holds no covered stretch to compare against.
