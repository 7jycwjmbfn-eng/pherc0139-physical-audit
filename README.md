# Physical ground truth for surface models, from a second scan of the same scroll (two scrolls)

Every existing way of checking a surface model depends on another model or on mesh-derived annotations. This repository builds ground truth that depends on neither: two scrolls have a second public scan of the same object at higher resolution, and at 1-2 um the sheets are directly visible, so the truth is read off the physical object.

It registers both pairs, releases the resulting label volumes and a standalone evaluator so the check can be run on any surface model, and demonstrates the instrument on the challenge's published predictions (`surface-m7`, the deployed Kaggle-winning nnU-Net, and the older `surface-recto-090`).

- **PHerc0139** (not on the Grand Prize list; loosely wound): 9.362 um full scroll + 1.129 um mosaic ROI. Registration held-out error **4.1 um** median. Byproducts: the two scans' nominal voxel sizes disagree by 0.22 percent, and a smooth 5-11 um cross-scan deformation field was measured. Both were previously unreported.
- **PHerc1203** (on the 13-scroll Grand Prize list; heavily compressed, with 7-14 percent of its tissue in clustered boundary-poor material and fused blocks confirmed visually at 4.8 um): 9.362 um full scroll + 2.403 um scan covering the full cross-section over 36 mm. Registration held-out error **2.4 um** median.

**What is being audited.** The subject of this audit is the officially published prediction volumes (checkpoint 20260413222639; we listed all 13 Grand Prize scrolls individually and each carries this same checkpoint). These artifacts are what downstream tracing consumes, so their quality is a fact about the pipeline's output regardless of its cause. One known upstream confounder must be named: villa [#1364](https://github.com/ScrollPrize/villa/issues/1364) (2026-08-07) reports that the `vesuvius.predict` path ignores the CTNormalization declared in m7's plans.json and normalizes per volume instead, which shifts recall substantially on intensity-shifted volumes. If the published volumes were generated through that path, part of the absolute error measured here may be attributable to the normalization defect rather than to the checkpoint itself. The cross-scroll comparison is less exposed to this (both 9.362 um volumes come from the same 2025 scan campaign with similar intensity statistics), and none of the label products depend on it, but re-auditing regenerated predictions once #1364 is fixed is the obvious follow-up, and the evaluator here can do it unchanged.

## What the instrument shows: the same model on both scrolls

The first thing this truth was used to measure. Arc-level metrics (1.2 mm sheet stretches, shifted-null controlled):

| | PHerc0139 (loose) | PHerc1203 (compressed, GP list) |
|---|---|---|
| Arc recall / shifted null | 89.9% / 57.7% | 76.1% / 64.6% |
| **Margin over null** | **+32.2 pp** | **+11.5 pp** |
| Stretches completely missed | 5.7% | 12.7% |
| Recto-side placement, `side_inward` (null 50.1) | 69.1% = 45% of ideal | 54.8% = **12% of ideal** |

Three conclusions that hold under every analysis choice we tested:

1. **On the compressed Grand-Prize scroll, the production model's margin over the null drops to a third and its recto-side semantics nearly vanish**, while raw recall only falls from 90 to 76: sheet density masks the degradation.
2. **Radius-based recall without a shifted-null control is untrustworthy in dense tissue** (the null alone reaches 64.6 percent on PHerc1203). This applies to any evaluation on these volumes, not just this one.
3. Zone-stratified numbers are sensitive to the zone definition (two physically reasonable zonings give different within-scroll curves; both are in `results/1203/`). The cross-scroll comparison above is zone-free and robust.

On the resolution question a reader should ask: the two truths come from scans at different native resolutions (1.129 vs 2.403 um), but both audits run on matched working grids (18.7 and 19.2 um), so the instruments are equally blunt on both scrolls; the ideal-recto ceilings (92.7 vs 88.3) additionally normalize for scroll geometry. The residual provenance difference is far smaller than the 45-to-12 percent drop it would need to explain.

Every metric reported here carries a shifted-null control; the side metric also carries an ideal-recto ceiling built from the truth; a synthetic self-test guards each instrument; registration error (2.4-4.1 um) is small against every tolerance used (19-56 um). Both label volumes ship with a standalone evaluator, and its exact output on both scrolls is committed at `results/eval_selfcheck_0139.json` and `results/1203/eval_selfcheck_1203.json`, so a rerun can be diffed against a file rather than against prose.

## Part 1: the registration

Method: band-pass sheet-pitch features, exhaustive global search over the full scroll length x full 360 deg in-plane rotation x translation (masked NCC via FFT), then 3D block matching (40^3 blocks, 10k blocks, median NCC 0.906) with a robust 7-parameter fit (rigid + isotropic scale), then a lattice deformation field.

- Unique global lock: z-score 27.2 over 49,920 candidate poses. Rotation 183 deg, z offset 33.6 mm (gantry metadata predicts 35.5 mm independently).
- Split-half cross-validation of the final mapping: median held-out error 4.1 um (rigid-only 5.9 um), p95 12.4 um. Papyrus is ~40 um thick; median winding pitch here is 271 um.
- Two byproducts:
  1. The nominal voxel sizes of the two volumes disagree by 0.22 percent (fitted isotropic scale 1.00217). Over the 40 mm ROI that is a ~130 um systematic error if uncorrected.
  2. A smooth residual deformation field between the two scans (5.3 um at 2 mm separation rising to 11.0 um at 16 mm, concentrated at mosaic tile seams). This is a dense measurement of cross-scan geometric distortion for a BM18 scroll pair; the transform files published for other scrolls carry no uncertainty information.

The transform ships in the project's transform.json schema (`20260413113053-to-20250728140407.json`) with 24 block-match landmarks embedded, so it can be re-fit or cross-checked from the file alone (landmark self-check: median 0.79 voxel at 9.362 um).

## Part 2: the PHerc0139 audit

Prediction volumes: `...surface-m7-L0-th0.2.zarr` and `...surface-recto-090.zarr`, both on the native 9.362 um grid. Audited inside the mapped 1.129 um ROI (22 mm of scroll length). Truth: sheet material at intensity threshold 65 (valley between air and papyrus modes), per-slice sheet centerlines; 137.7M centerline points, 82,653 arc segments (1.2 mm stretches of individual sheets). The standalone evaluator scores every fourth slice, so it reports 35.2M of those points and the same 82,653 arcs.

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

The null control (predictions shifted 1.2 mm in-plane) shows how much of any radius-based recall is produced by sheet density alone; per-point radius recall overstates model skill and the arc-level numbers are the ones to quote.

Point-level tolerance note: m7 predicts the recto surface, not the sheet interior, so an ideal model sits 1-2 voxels off the material centerline by construction. The 56 um row is the semantically fair point-level figure; the 19 um row is reported for completeness, not as a deficiency claim.

### Findings beyond the headline

1. **Missed sheets are darker.** Fully-missed arcs average intensity 105.0 against 118.1 for covered arcs (11.1 percent darker, in the 1.129 um frame). TAUIL-Abd-Elilah reported a 10.3 percent intensity deficit for missed voxels measured in the model's own frame (issue #191, 2026-07-28); this reproduces that result against physical truth, independent of the model's frame and labels. Spatially the misses are part scattered, part clustered: 17.4 percent have another fully-missed stretch within 100 um, median nearest-neighbor distance 389 um, with several z-persistent clusters visible in `figures/fig_gone_map.png`. Two failure modes coexist: sheet-level darkness and localized zones.

2. **Side-of-sheet placement** (`side_inward` in the evaluator, not to be confused with `recto_side_ratio`, which is a separate mass-overlap quantity). m7 is documented as a recto (inward-facing surface) predictor. We test this directly: at each centerline point that has the band in question within 3 voxels, sample that band's distance field 2 voxels to each side along the local sheet normal (structure tensor, oriented inward by the radial sign; the estimator passes a synthetic-stripe self-test at 4 angles). Each arm selects its own points under the same rule, so the rows below rest on 31.9M, 21.6M and 30.2M decided cases rather than on one shared set:

   | Prediction | Inward fraction of decided cases |
   |---|---|
   | Shifted null | 50.1% |
   | surface-m7 | **69.1%** |
   | surface-recto-090 | 75.8% |
   | Ideal recto band built from truth | 92.7% |

   On a scale where the shifted null sits at 50.1 and a perfect recto band scores 92.7, m7 carries 45 percent of the ideal side signal and recto-090 carries 60 percent. By this measure the deployed prediction behaves closer to a side-agnostic sheet-surface detector than to a recto detector.

   The standalone evaluator reimplements this instrument from the packaged labels alone and returns 0.6911 / 0.5011 / 0.9270 with a skill of 0.4459, against 0.691154 / 0.501147 / 0.927155 here. The one difference: the pass above takes sheet normals from the registered 1.129 um grayscale, which the label package does not carry, so the evaluator takes them from the smoothed material mask instead. That changes which points clear the coherence floor (28.4M decided cases against 31.9M) without moving the fractions.

3. **Mechanism of the side errors.** Signed-offset sampling along the normal: for m7, 72.8 percent of band mass sits inward, 16.0 centered, 11.3 outward (ideal: 94.8 inward). At the 1.2 mm tile level, 85.1 percent of tiles are coherently inward and 0 percent coherently outward. There are no sector-scale orientation flips, so the winding-orientation input is not the problem; the recto signal is real but noisy point to point. The actionable target is per-point side stability, not the orientation field.

4. **The older model compares favorably here.** Same audit, same truth, same box:

   | Metric | surface-m7 | surface-recto-090 |
   |---|---|---|
   | Arc recall | 89.9% (null 57.7) | 94.7% (null 63.4) |
   | Arcs completely absent | 5.7% (null 27.0) | 3.6% (null 24.0) |
   | Point recall at 37 um | 81.2% (null 54.8) | 88.7% (null 61.1) |
   | Inward (recto) side | 69.1% | 75.8% |
   | Positives beyond 37 um from real papyrus | 2.5% | 2.1% |

   recto-090 predicts a denser band (its higher nulls reflect that); against each model's own null the two are near parity on coverage skill, and on side placement recto-090's advantage survives calibration (60 vs 45 percent of ideal). Within this uncompressed ROI, the newer production model did not improve coverage skill and gives up part of the recto semantics. m7 was tuned for compressed-region benchmarks, which this ROI cannot probe. The natural test would be the same comparison on PHerc1203, but no recto-090 prediction volume has been published for that scroll, so the model-vs-model comparison is limited to PHerc0139; on PHerc1203 only m7 can be audited as a published artifact.

### Scope limits (stated up front)

- The 1.129 um ROI of PHerc0139 contains no compressed regions: 6,755 tissue windows measured, median sheet spacing 271 um, 0.06 percent below 100 um, and the spacing instrument's failure channels account for every window (0.74 percent dropped). This PHerc0139 audit therefore only covers ordinary tissue; the PHerc1203 campaign below is where compressed zones are measured.
- Truth threshold sensitivity: arc recall / fully-missed at thresholds 50 / 65 / 80 are 90.7 / 89.9 / 90.0 percent and 5.8 / 5.7 / 5.3 percent. The conclusions do not depend on the threshold choice.
- Distances are 2D per-slice (sheets near-vertical; small underestimate of 3D distance).

## The PHerc1203 campaign (Grand Prize scroll)

- **Registration**: exhaustive global search locks at z-score 44.0 (rotation 0 deg, z offset 74.4 mm); 19,116 matched blocks, median NCC 0.899; split-half held-out error 2.4 um median, 6.1 um p95. Fitted scale correction is only -0.025 percent for this pair, against 0.22 percent for the 0139 pair: the calibration quality of published volumes varies by scan campaign.
- **Difficult tissue confirmed and mapped**: 7-14 percent of tissue windows are unmeasurable by structure-tensor spacing at 19 um (73-86 percent of them clustered, growing along z); a 4.8 um look inside the largest cluster shows crosshatched fiber bundles pressed into near-solid blocks (1 mm windows at 99.75 percent material). A threshold-robust physical zone variable, gap visibility (the fraction of local material within 57 um of a resolvable air boundary), separates loose (~87 percent) from boundary-poor (~28-35 percent) tissue and ships as a bit plane in the label volume.
- **Registration holds in dense tissue**: per-zone block-match residuals are 3.5 / 5.1 / 4.2 um (loose / dense / boundary-poor), so labels cast in difficult regions inherit only ~5 um mapping error.
- **Side-of-sheet replication** (`side_inward`): 9.1M points, m7 inward fraction 54.8 percent against a 50.1 null and an 88.3 ideal ceiling. That is 12 percent of the ideal signal, versus 45 percent on PHerc0139. The standalone evaluator returns 0.5502 / 0.5011 / 0.8835 with a skill of 0.1286; it scores every fourth slice where this pass scored every eighth, and takes normals from the material mask rather than the 2.403 um grayscale, which is where the 0.2 point difference in the inward fraction comes from.
- **Fully-missed stretches double** (12.7 percent vs 5.7). The evaluator's full output on this scroll is committed at `results/1203/eval_selfcheck_1203.json`.

## The datasets and evaluator

Two label volumes ship as release assets, each a uint8 bit-flag zarr on its scroll's level-1 grid (18.724 um), images not included (the CT volumes are public; `.zattrs` names the exact grid):

- `labels0139_L1.zarr` (376 MB): window 1248 x 2304 x 2208 at origin (1728, 576, 480). Bits: valid (1) / material (2) / centerline (4) / recto_band (8).
- `labels1203_L1.zarr` (497 MB): window 2016 x 3456 x 3456 at origin (3936, 0, 0). Same bits plus **boundary_poor (16)**: the physical map of material with no resolvable boundary within 57 um at the truth resolution. That map also bears on the official 2027 question of telling "no ink" from "ink not yet recovered"; in boundary-poor tissue the limiting factor is physics, not the pipeline.

Release tarball hashes, for pinning a version:

```
labels0139_L1.tar  392693760 bytes  sha256 42fe53b760c2c9347d9f215bafa68beec8e96121d03549dab56a52a9a0a9e8dd
labels1203_L1.tar  515379200 bytes  sha256 32a09f6081342b0f015b258ec577d0296ff23a55892af9785689d8a55bff344c
```

`code/eval_surface_pred.py` (standalone; numpy + scipy + numcodecs) evaluates any binary surface prediction on the same volume's grid against these labels. Every metric it prints carries a shifted-null control except `recto_side_ratio`, which is a raw mass-overlap quantity kept for continuity; the side metric to read is `side_inward`, which ships with its null and its ideal-recto ceiling. The normal estimator self-tests on synthetic stripes at four angles before any volume is read.

```bash
python3 eval_surface_pred.py labels0139_L1.zarr /path/to/prediction.zarr 0
```

Validation: its full output against the published m7 volume is committed at `results/eval_selfcheck_0139.json` and `results/1203/eval_selfcheck_1203.json`, so a rerun can be diffed. The point and arc numbers match this report to four decimals; the side arm agrees to three (0.6911 / 0.5011 / 0.9270 against 0.691154 / 0.501147 / 0.927155), the residual coming from normals taken off the material mask rather than the high-resolution grayscale, which the label package does not carry. villa issue #193 asked for objective labels that do not depend on a model; these two volumes are built that way, and the PHerc1203 one covers compressed and boundary-poor tissue.

What these labels do not yet show: that training on them makes a model better. That is the second half of the acceptance criterion discussed in #193 ("prove it is better"), it requires a training run, and it is the natural next step for these datasets rather than part of this release.

## Outside checks

- Jinhojeong reran the standalone evaluator on 2026-08-10 against prediction chunks fetched independently from the open-data bucket, and reported the point and arc numbers matching this report ([villa #191](https://github.com/ScrollPrize/villa/issues/191)).
- TAUIL-Abd-Elilah is freezing a PHerc1203 mesh A/B against the +64-L1 shifted null published here, with `boundary_poor` reported as a separate stratum (same thread).
- The gap that thread found, the standalone evaluator printing only `recto_side_ratio` while this report quotes `side_inward`, is fixed: the pass7b instrument now ships in the evaluator with both its controls.

## Related work

- The ongoing #193 thread (TAUIL-Abd-Elilah and Jinhojeong, 2026-08-05 onward) reached, from label-side statistics alone, the conclusion that the existing label set has no purchase on the failure modes the open-problems page names, and proposed building a label set that does contain the hard regions. The PHerc1203 volume here is an existing instance of that object, built from a second physical scan rather than from annotation, and their thread is also where the #1364 normalization defect was isolated.
- [herculaneum-scroll-tools](https://github.com/axiosdevs/herculaneum-scroll-tools) audits m7 phantom fractions against the masked CT (including on PHerc0139) and aligns coordinate systems across scans. Complementary: it measures false positives against the CT mask; this work measures recall, side semantics, and miss bias against an independent physical scan, and ships the labels.
- [vesuvius-surface-geometry-diagnostic](https://github.com/Jinhojeong/vesuvius-surface-geometry-diagnostic) stratifies surface-model AUC by curvature/compression on labeled patches (different models and scrolls; labels from the dataset itself). Its intensity-deficit finding for missed voxels is independently reproduced here against physical truth.
- [VesuviusScrollAlignment](https://github.com/Paul-G2/VesuviusScrollAlignment) does affine cross-scan alignment (PHerc1667) via mutual information; no held-out validation or downstream labels.
- Synthetic phantoms (Diego-dcv) provide label-free per-voxel truth with fixed geometry; a second physical scan of the real object is the non-synthetic counterpart.

## Repository layout

- `code/` contains the full pipeline for both scrolls: pass1 (global registration) through pass10 (dataset packaging) for PHerc0139, and the `*_1203.py` drivers that re-run the same machinery on PHerc1203; each stage has a built-in self-test, null control, or ceiling calibration; `eval_surface_pred.py`; the chunk fetcher.
- `results/` holds one JSON per stage (PHerc0139 at the root, PHerc1203 under `results/1203/`), plus the transform npz and missed-arc coordinates.
- `figures/` holds registration checkerboards, prediction/truth overlays, audit bar charts, the missed-arc map, and for PHerc1203 the unmeasurable-zone map (`dropmap_z1344.png`), the 4.8 um fused-block interior (`blue_zoom0.png`), and the zone map (`zonemap_z1344.png`).
- `REPRO.md` gives the environment, exact S3 prefixes and chunk ranges, run order, and runtimes (consumer laptop, no GPU).
- `20260413113053-to-20250728140407.json` is the PHerc0139 transform in the project schema. The PHerc1203 transform lives in `results/1203/pass3_final.npz` (affine `M2, t2`, hi_L4 voxel to lo_L2 voxel, plus the block-match field and deformation lattice).

## License

Code: MIT. The label volumes and all derived numbers inherit CC BY-NC 4.0 from the underlying Vesuvius Challenge scan data.
