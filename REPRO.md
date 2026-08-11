# Reproduction

Everything below runs on one machine (tested: Windows 11 + WSL2 Ubuntu 22.04, 32 GB RAM of which ~27 GB visible in WSL, no GPU needed). Total compute is a few hours; total download is ~12 GB from the public open-data bucket (anonymous, no credentials).

## Scoring a prediction without rerunning any of it

Most uses need only this. Download a label volume from the release, untar it, point the evaluator at any binary surface prediction on the same scroll's grid:

```bash
pip3 install numpy scipy numcodecs
tar xf labels1203_L1.tar
python3 code/eval_surface_pred.py labels1203_L1.zarr /path/to/prediction.zarr 0
```

The last argument is the pyramid level of the prediction (0 or 1); level 0 is max-pooled onto the label grid. Runtime is about 15 minutes for PHerc0139 and an hour for PHerc1203 on one core, dominated by reading the prediction volume. Verify the tarball first if you plan to freeze anything against it; hashes are in the release notes and in the README.

The output for the published m7 volumes is committed at `results/eval_selfcheck_0139.json` and `results/1203/eval_selfcheck_1203.json`, so a rerun on those volumes can be diffed rather than eyeballed.

Two smaller scripts, neither needing a prediction volume:

```bash
python3 code/bit_census.py                     # what every bit plane covers, both volumes
python3 code/test_side_arms_equivalence.py     # self-check on the side estimator, synthetic
```

`bit_census.py` reads both label volumes and reports coverage and containment per bit; run it before stratifying on any bit plane, since `boundary_poor` covers 74.6 percent of material on PHerc1203 rather than flagging a minority. `test_side_arms_equivalence.py` needs no data at all and exits nonzero if the side arms' point selection ever drifts.

The rest of this file is for rebuilding the labels from the raw scans.

## Environment

```bash
pip3 install numpy scipy numcodecs pillow matplotlib
```

## Data

All from `https://vesuvius-challenge-open-data.s3.amazonaws.com` (raw uint8 zarr chunks; the prediction volume is blosc-zstd compressed). `code/fetch_chunks.sh` fetches a rectangular chunk range. The exact prefixes:

| Alias | S3 prefix | What to fetch |
|---|---|---|
| hi_L4 / hi_L5 | `PHerc0139/volumes/20260413113053-1.129um-0.2m-59keV-masked.zarr/4` (and `/5`) | whole level |
| lo_L3 | `PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/3` | whole level |
| lo_L2 | same volume, level `2` | z chunks 6-11, all y/x |
| pred_L0 | `PHerc0139/representations/predictions/surfaces/20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr/0` | z chunks 18-30, all y/x |

Run each fetch twice and confirm the second pass adds zero files: the fetch script deletes failed downloads, and for masked volumes an absent chunk is indistinguishable from an all-zero one.

## Run order

Each stage writes its outputs next to the scripts and prints its numbers; each carries its own control (self-test, null, or ceiling) and refuses silently wrong readings.

| Stage | Command | Control built in |
|---|---|---|
| 1. Global lock | `python3 pass1.py selftest && python3 pass1.py run` | synthetic rotation recovery, full-search z-score |
| 2. Block refine | `python3 pass2.py` | numeric rotation-convention check |
| 3. Validate + field | `python3 pass3.py` | split-half held-out CV |
| 4. Spacing survey | `python3 pass4b.py` | per-window outcome accounting |
| 5. Point audit | `python3 pass5b.py` | shifted-null at all radii |
| 6. Arc audit | `python3 pass6.py` (+ `SENS_TH=50/80 python3 pass6s.py`) | shifted null + threshold sensitivity |
| 7. Side audit | `python3 pass7b.py` | stripe self-test, shifted null, ideal-recto ceiling |

Figures: `gen_figs.py`. Transform export with landmark self-check: `gen_transform.py`.

Runtimes on the test machine: stage 1 ~30 min (22 cores), stages 2-7 5-10 min each (4 workers; bounded by RAM, not cores).

## PHerc1203

The `code/*_1203.py` drivers re-run the same stages on PHerc1203 (volumes `20250820131727` at 9.362 um and `20260319130212` at 2.403 um; predictions `20250820131727-surface-20260413222639-surface-m7-L0-th0.2`). Data windows and chunk ranges are in each driver's header. The run order mirrors the 0139 stages: `pass1_1203` (global lock, with the same selftest), `pass2_1203`, `pass3_1203` (held-out CV), `survey_1203` + `calib_1203` (zone calibration), `pass5b_1203` (stratified audit), `pass7_1203` (side audit), `pass10_1203` (labels). The 1203 volumes are ~3x larger; stages run with 3 workers and stay under 27 GB RAM.
