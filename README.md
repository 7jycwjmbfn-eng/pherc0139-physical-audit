# A physical audit of surface predictions on PHerc0139

Cross-resolution registration of PHerc0139's two public scans, used as model-free ground truth to audit the official surface prediction volumes — plus the truth labels and the evaluator, packaged for reuse.

## What this is

PHerc0139 is the only Grand-Prize-list scroll with two public scans of the same object at different resolutions: a full-scroll volume at 9.362 um (2025-07, 113 keV) and a 19-tile mosaic ROI at 1.129 um (2025-12, 59 keV). No official registration between them existed. We registered them, validated the registration on held-out data, and then used the 1.129 um scan as model-free ground truth to audit the official `surface-m7` and `surface-recto-090` prediction volumes published for the 9.362 um frame.

The point of using a second physical scan as the reference: every existing validation of surface predictions relies either on another model or on labels derived from meshes that were themselves traced on model output. villa issue #193 closed (2026-08-08) with both model-free validation attempts withdrawn (a CT statistic with window bias, and a second model at chance level). A higher-resolution scan of the same object sidesteps both failure modes.

## Part 1: the registration

Method: band-pass sheet-pitch features, exhaustive global search over the full scroll length x full 360 deg in-plane rotation x translation (masked NCC via FFT), then 3D block matching (40^3 blocks, 10k blocks, median NCC 0.906) with a robust 7-parameter fit (rigid + isotropic scale), then a lattice deformation field.

- Unique global lock: z-score 27.2 over 49,920 candidate poses. Rotation 183 deg, z offset 33.6 mm (gantry metadata predicts 35.5 mm independently).
- Split-half cross-validation of the final mapping: median held-out error 4.1 um (rigid-only 5.9 um), p95 12.4 um. Papyrus is ~40 um thick; median winding pitch here is 271 um.
- Two byproducts:
  1. The nominal voxel sizes of the two volumes disagree by 0.22 percent (fitted isotropic scale 1.00217). Over the 40 mm ROI that is a ~130 um systematic error if uncorrected.
  2. A smooth residual deformation field between the two scans (5.3 um at 2 mm separation rising to 11.0 um at 16 mm, concentrated at mosaic tile seams). This is a dense measurement of cross-scan geometric distortion for a BM18 scroll pair; the transform files published for other scrolls carry no uncertainty information.

The transform ships in the project's transform.json schema (`20260413113053-to-20250728140407.json`) with 24 block-match landmarks embedded, so it can be re-fit or cross-checked from the file alone (landmark self-check: median 0.79 voxel at 9.362 um).

## Part 2: the audit

Prediction volumes: `...surface-m7-L0-th0.2.zarr` (the production nnU-Net, recto-surface semantics, threshold 0.2) and `...surface-recto-090.zarr` (the older recto predictor), both on the native 9.362 um grid. Audited inside the mapped 1.129 um ROI (22 mm of scroll length). Truth: sheet material at intensity threshold 65 (valley between air and papyrus modes), per-slice sheet centerlines — 137.7M centerline points, 82,653 arc segments (1.2 mm stretches of individual sheets). Registration error budget (4.1 um median) is small against every tolerance used (19-56 um).

### Headline numbers (m7)

| Metric | Real | Shifted-null control |
|---|---|---|
| Centerline points with a prediction within 19 um | 63.7% | 39.5% |
| within 37 um | 81.2% | 54.8% |
| within 56 um | 91.0% | 67.7% |
| Arc-level recall (>=50 percent of a 1.2 mm sheet stretch covered at 37 um) | 89.9% | 57.7% |
| Arcs completely absent from the prediction (<10 percent coverage) | 5.7% | 27.0% |
| Predicted positives farther than 37 um from any real papyrus | 2.5% | - |
| farther than 75 um | 0.25% | - |

The null control (predictions shifted 1.2 mm in-plane) shows how much of any radius-based recall is produced by sheet density alone; per-point radius recall overstates model skill and the arc-level numbers are the ones to quote. This applies to radius-based metrics on these volumes generally, not just here.

Point-level tolerance note: m7 predicts the recto surface, not the sheet interior, so an ideal model sits 1-2 voxels off the material centerline by construction. The 56 um row is the semantically fair point-level figure; the 19 um row is reported for completeness, not as a deficiency claim.

### Findings beyond the headline

1. **Missed sheets are darker.** Fully-missed arcs average intensity 105.0 against 118.1 for covered arcs (11.1 percent darker, in the 1.129 um frame). TAUIL-Abd-Elilah reported a 10.3 percent intensity deficit for missed voxels measured in the model's own frame (issue #191, 2026-07-28); this reproduces that result against physical truth, independent of the model's frame and labels. Spatially the misses are part scattered, part clustered: 17.4 percent have another fully-missed stretch within 100 um, median nearest-neighbor distance 389 um, with several z-persistent clusters visible in `figures/fig_gone_map.png`. Two failure modes coexist: sheet-level darkness and localized zones.

2. **Side-of-sheet placement.** m7 is documented as a recto (inward-facing surface) predictor. We test this directly: at each centerline point with a nearby prediction, sample the distance-to-prediction field 2 voxels to each side along the local sheet normal (structure tensor, oriented inward by the radial sign; the estimator passes a synthetic-stripe self-test at 4 angles). Under the identical instrument, 31.9M points:

   | Prediction | Inward fraction of decided cases |
   |---|---|
   | Shifted null | 50.1% |
   | surface-m7 | **69.1%** |
   | surface-recto-090 | 75.8% |
   | Ideal recto band built from truth | 92.7% |

   On a scale where chance is 50 and a perfect recto band scores 92.7, m7 carries 45 percent of the ideal side signal and recto-090 carries 60 percent. By this measure the deployed prediction behaves closer to a side-agnostic sheet-surface detector than to a recto detector.

3. **Mechanism of the side errors.** Signed-offset sampling along the normal: for m7, 72.8 percent of band mass sits inward, 16.0 centered, 11.3 outward (ideal: 94.8 inward). At the 1.2 mm tile level, 85.1 percent of tiles are coherently inward and 0 percent coherently outward. There are no sector-scale orientation flips — the winding-orientation input is not the problem; the recto signal is real but noisy point to point. The actionable target is per-point side stability, not the orientation field.

4. **The older model compares favorably.** Same audit, same truth, same box:

   | Metric | surface-m7 | surface-recto-090 |
   |---|---|---|
   | Arc recall | 89.9% (null 57.7) | 94.7% (null 63.4) |
   | Arcs completely absent | 5.7% (null 27.0) | 3.6% (null 24.0) |
   | Point recall at 37 um | 81.2% (null 54.8) | 88.7% (null 61.1) |
   | Inward (recto) side | 69.1% | 75.8% |
   | Positives beyond 37 um from real papyrus | 2.5% | 2.1% |

   recto-090 predicts a denser band (its higher nulls reflect that); against each model's own null the two are near parity on coverage skill, and on side placement recto-090's advantage survives calibration (60 vs 45 percent of ideal). Within this uncompressed ROI, the newer production model did not improve coverage skill and gives up part of the recto semantics. m7 was tuned for compressed-region benchmarks, which this ROI cannot probe; the comparison is scoped accordingly.

### Scope limits (stated up front)

- The 1.129 um ROI of PHerc0139 contains no compressed regions: 6,755 tissue windows measured, median sheet spacing 271 um, 0.06 percent below 100 um, and the spacing instrument's failure channels account for every window (0.74 percent dropped). This audit says nothing about model behavior in compressed zones, which is where issue #191 locates the hard failures. Its value is the reference-quality error measurement in ordinary tissue, which no model-dependent method provides.
- Truth threshold sensitivity: arc recall / fully-missed at thresholds 50 / 65 / 80 are 90.7 / 89.9 / 90.0 percent and 5.8 / 5.7 / 5.3 percent. The conclusions do not depend on the threshold choice.
- Distances are 2D per-slice (sheets near-vertical; small underestimate of 3D distance).

## The dataset and evaluator

The truth is packaged as `labels0139_L1.zarr` (376 MB, zstd, 128^3 chunks; attached to the GitHub release): a uint8 bit-flag volume on the lo scan's level-1 grid (18.724 um), window 1248 x 2304 x 2208 at origin (1728, 576, 480). Bits: valid (1) / material (2) / centerline (4) / recto_band (8). Images are not shipped; the lo CT volume is public and `.zattrs` names the exact grid.

`code/eval_surface_pred.py` (standalone; numpy + scipy + numcodecs) evaluates any binary surface prediction on the same volume's grid against these labels, with the shifted-null control built in:

```bash
python3 eval_surface_pred.py labels0139_L1.zarr /path/to/prediction.zarr 0
```

Validation: run against the published m7 volume, it reproduces every audit number in this report to four decimals (recall 0.6372 / 0.8116 / 0.9104, arc recall 0.8994, fully-missed 0.0574, nulls matching). Issue #193 closed asking for objective labels that do not depend on a model; this is a first installment, scoped to ordinary tissue.

## Related work

- [herculaneum-scroll-tools](https://github.com/axiosdevs/herculaneum-scroll-tools) audits m7 phantom fractions against the masked CT (including on PHerc0139) and aligns coordinate systems across scans. Complementary: it measures false positives against the CT mask; this work measures recall, side semantics, and miss bias against an independent physical scan, and ships the labels.
- [vesuvius-surface-geometry-diagnostic](https://github.com/Jinhojeong/vesuvius-surface-geometry-diagnostic) stratifies surface-model AUC by curvature/compression on labeled patches (different models and scrolls; labels from the dataset itself). Its intensity-deficit finding for missed voxels is independently reproduced here against physical truth.
- [VesuviusScrollAlignment](https://github.com/Paul-G2/VesuviusScrollAlignment) does affine cross-scan alignment (PHerc1667) via mutual information; no held-out validation or downstream labels.
- Synthetic phantoms (Diego-dcv) provide label-free per-voxel truth with fixed geometry; a second physical scan of the real object is the non-synthetic counterpart.

## Repository layout

- `code/` — the full pipeline, pass1 (global registration) through pass10 (dataset packaging), each stage with a built-in self-test, null control, or ceiling calibration; `eval_surface_pred.py`; the chunk fetcher.
- `results/` — one JSON per stage with the numbers quoted above, plus the transform npz and missed-arc coordinates.
- `figures/` — registration checkerboards, prediction/truth overlays, audit bar charts, missed-arc map.
- `REPRO.md` — environment, exact S3 prefixes and chunk ranges, run order, runtimes (consumer laptop, no GPU).
- `20260413113053-to-20250728140407.json` — the transform in the project schema.

## License

Code: MIT. The label volume and all derived numbers inherit CC BY-NC 4.0 from the underlying Vesuvius Challenge scan data.
