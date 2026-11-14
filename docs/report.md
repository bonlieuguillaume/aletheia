# Aletheia: design report

Living document: every design choice of the project, **why** it was made, and
what remains open. It is updated as the work progresses; the task list itself
lives in the [GitHub Project](https://github.com/users/bonlieuguillaume/projects/5).

Status: v0.1 code written (tiling, copy-move levels 1-3, U-Net, evaluation);
scenes preprocessed, tiling (2000 patches), one model trained per level 1-3,
evaluation across levels in progress (sections 5.1, 5.2). Last update:
2026-11-14.
Architecture and metric procedure of v0.1, with every trade-off listed in one
table (section 7): [architecture_v0.1.md](architecture_v0.1.md).

## 1. Goal and scope

Detect forged SAR images: give an **image-level score** (how likely the
image is tampered) and a **pixel-level anomaly map** (where).

The project grows by steps, each one answering one question:

| Version | Forgery | Question |
|---|---|---|
| v0.1 | copy-move inside the patch | Can a first model detect it? (mockup) |
| v0.2 | same | Does knowledge of SAR physics add value? |
| v0.3 | distant-source copy-move, then splicing | Does it hold when the pasted region comes from elsewhere? |
| v1 | inpainting, generated images, other sensors... | Does it generalize, including against an informed attacker? |

Target use case for v0.1: a **military base in a forest hidden by
copy-pasted forest**, the SAR counterpart of the optical case documented by
[Vertical 52](https://vertical52.org/en/lab/fake-forests).

## 2. Data

### 2.1 Source images

- **Sentinel-1 IW GRD, VV only, amplitude**, 10 m pixel spacing.
  - *Why VV:* we have no Sentinel-1 HH data for now (S1 acquires HH mainly
    over polar areas). A single-band detector also matches how images
    circulate: many SAR products are single-pol, and an image published
    online is a single grayscale picture. Other polarisations: #34.
- **10 full scenes, forest only for now**, matching the use case (forests in
  Ukraine hiding military bases). Diversity is sought within forest:
  regions, ascending/descending orbits, seasons. v0.1 uses 10 South American
  scenes (Amazon rain forest and Gran Chaco dry forest, 2017-2025, 8
  descending + 2 ascending), listed with their Sentinel-1 product in
  [v0.1_south_american_forest.csv](v0.1_south_american_forest.csv).
  - *Known limit:* a model trained on forest only is not expected to
    generalize to other land covers.
- **Orbits balanced: ideally 5 ascending + 5 descending**, the orbit written
  in each scene folder name. Sentinel-1 looks to the right of its track: on a
  north-up image an ascending pass looks towards ~78 deg (shadows to the
  east), a descending one towards ~282 deg (shadows to the west). If the
  scenes are not 5-5, tiling **mirrors east-west** some scenes of the
  majority orbit: the mirror maps a look direction d to 360 - d, i.e. exactly
  78 deg <-> 282 deg. A mirrored scene replaces its original (never both, or
  one landscape could end up in two splits). The manifest keeps the orbit of
  the acquisition, whether it was mirrored, and the resulting geometry.
- No speckle filter, no extra multilook. IW GRD is nevertheless **already
  multilooked by ESA** (about 5x1 looks, ENL ~4.4): the speckle is not
  single-look, which matters for any speckle statistic (v0.2).
- Preprocessed by the user with SNAP 13 on their PC, **one graph shared by
  all scenes** ([snap/GRD_pre_process.xml](snap/GRD_pre_process.xml), local
  paths replaced by `${input}` / `${output}`; only the input product and the
  UTM zone change). The code reads none of it: from each scene it only uses the
  pixel values of the image and the orbit written in the folder name. The
  graph:
  - Read: whole product, no subset;
  - Apply-Orbit-File: precise orbits (`continueOnFail`: falls back silently
    to less accurate orbits if they are missing);
  - ThermalNoiseRemoval (VV), negative values clipped to 0;
  - Border-Noise-Removal, **for every scene**. ESA's IPF removes the GRD
    border noise itself since version 2.90 (March 2018), but two v0.1
    scenes are older (ds_forest_4, 2017-12-20; ds_forest_10, 2017-08-04).
    Running it on all scenes keeps a single graph; on a recent product it
    is expected to change little (at most a few low-value pixels at the
    swath edges set to 0, i.e. no-data) for a small extra cost;
  - Calibration: **Sigma0 VV, linear intensity** (not dB, not amplitude), so
    `input_scale = "linear"`, and the ENL of v0.2 is computed on intensity;
  - Terrain-Correction: Copernicus 30 m DEM, **bilinear image resampling**,
    10 m pixels, UTM zone of the scene, no radiometric terrain normalization
    (slopes keep their natural brightness). "Mask out areas without
    elevation" (`nodataValueAtSea`) is on: pixels where the DEM has no
    value (open sea) become 0, so the patches that contain them are
    rejected by the 5 % no-data rule. It does not skip flat terrain and
    saves no meaningful computation; it only matters for the coastal
    scenes (ds_forest_7: 568 valid windows, ds_forest_10: 735, against
    ~1250-1750 for the inland scenes). **To reuse the chain on maritime
    scenes (ships, oil slicks, coasts), switch it off**, or the sea is
    blanked out and tiling rejects every sea patch;
  - Write: BEAM-DIMAP (the graph did not run with GeoTIFF-BigTIFF as
    output). **The code reads both formats**: a GeoTIFF, or the Sigma0_VV
    band of a BEAM-DIMAP product, read directly from its ENVI `.hdr` +
    `.img` with numpy (raw big-endian float32, ~2.3 GB per scene), with no
    conversion and no new dependency. A GeoTIFF export (File > Export in
    SNAP, classic GeoTIFF, under the 4 GB limit of the format) stays useful
    to look at the scenes in other software; converting in the code would
    only add a 2.3 GB write per scene and, for a georeferenced file, a GDAL
    dependency. SNAP's own GeoTIFF reader fails to display a file past
    ~2.1 GB (32-bit size overflow, error `capacity < 0`) but the file is
    valid, and a different reader opens it; tifffile has no such limit.
- Terrain correction **resamples** the image: neighbouring speckle values
  become slightly correlated, which changes local statistics (relevant for
  v0.2 physical cues).

### 2.2 Patches

- **512x512** pixels, i.e. 5.12 x 5.12 km at 10 m.
  - *Why not 256:* room for several sizeable pastes in one patch; a 2-3 km
    base fits entirely.
- **Crop, never resize**: resizing would destroy the fine-scale traces a
  detector relies on.
- Tiling runs on **Colab** (#28), not in the preprocessing: the user's chain
  stops at scene GeoTIFFs, everything dataset-related lives in this repo.
- ~200 patches drawn per scene (a full scene gives ~1,000-1,200 usable ones):
  diversity matters more than count.
- Plain single-band `.tif`, **without georeferencing**: the network sees
  pixels only.
- **No-data rule:** terrain correction leaves large empty areas around the
  rotated footprint; isolated zero-valued pixels are also a normal part of
  SAR images. A patch is rejected only if its no-data fraction exceeds a small
  threshold, **5 % of the patch pixels** (10 % was judged too permissive),
  contiguous blocks being the target. Code must handle zeros safely (log / dB).
- A **manifest** (one CSV, one row per patch) is the single index: scene,
  position in the scene, split, and the forgery parameters.

### 2.3 Splits and reproducibility

- **Split by scene, 6 train / 2 val / 2 test**: the test only contains
  landscapes never seen in training. Val and test each take one
  ascending-like and one descending-like scene, so both geometries are
  measured.
  - *Possible improvement, not planned (too time-consuming):* scene-level
    cross-validation, e.g. 5 rounds where 2 different scenes serve as test
    and the 8 others for training, averaging the scores. Training would see
    more scenes and the test score would rest on all 10 scenes, at the cost of
    5 trainings.
  - *Why:* a network with millions of parameters can memorize thousands of
    images (Zhang et al., 2017, fit random labels on CIFAR-10); with
    patch-level splits, it would recognize scenes instead of forgeries.
  - *Draw of v0.1 (seed 0):* train = ds_forest_2, 3, 4, 5 (mirrored), 8,
    10; val = ds_forest_6 (mirrored), 7; test = ds_forest_1, 9 (mirrored).
    *Known limit:* the only two truly ascending scenes (4, 8) and the three
    Gran Chaco dry-forest scenes (4, 5, 8) all fell in train, so the
    ascending-like scenes of val and test are mirrored descending ones, and
    val and test are Amazon rain forest only. Accepted for v0.1, which
    measures copy-move detection, not transfer to another forest type;
    `FORCED_SPLITS` can put a real ascending Chaco scene in test.
- An authentic patch and its forged version form a **pair** and always land
  on the same side of the split (guaranteed by the split by scene). Pairs
  force the model to learn the forgery, the only difference between the two.
- Every random choice is **seeded**; val and test sets are fixed and
  precomputed.

## 3. Forgeries

### 3.1 v0.1: copy-move inside the patch

- **2,000 real + 2,000 forged** patches, each forged patch derived from a
  real one.
- Source and target both inside the patch: the duplicate is visible to the
  model. A naive copy-move duplicates the exact speckle realization, which is
  physically impossible in a real acquisition: a cue in itself (v0.2).
- **Difficulty levels**, used for both training and evaluation, from simple
  to hard: one small rectangle of 2-5 % of the patch (#24); varied simple
  shapes (#25); several pastes (#26); widely varying sizes, 1 % to 25 %+
  (#27).
  - At level 1, source and target stay **at least 32 px from the patch
    border**, so a failure cannot come from border effects. From level 2 on,
    pastes may touch the border, and their effect is measured (#32).
  - Level 2 (#25): rectangle, ellipse or slightly irregular polygon, drawn
    with equal odds; the shape itself (not its box) covers 2-5 % of the
    patch; only the pixels inside the shape are copied.
  - Level 3 (#26): 2 to 4 pastes per patch, each as at level 2, none
    overlapping another paste nor any source (4-20 % of the patch in all).
    One manifest row per sample is kept: the per-paste details go into a
    JSON column (`pastes`).
  - Every level is built from the **same authentic patches and split**, so
    a model can be scored on every level, and the scores compared.
- **Plan across levels (decided 2026-11-14).** Level 1 is saturated (test
  MCC 1.000): it leaves no room to measure the added value of SAR cues
  (v0.2). So, level by level:
  1. generate the level-k dataset and evaluate the **current model** on it,
     without retraining and with the settings of its own level kept frozen
     (`REUSE_SETTINGS = True`, used throughout the project): the model is
     scored exactly as it came out of its own level, weights and thresholds.
     Re-choosing the thresholds on the level-k val (`False`) stays possible
     in the code but is not used: it would mix a model fixed on one level
     with settings tuned on another, which blurs what is measured;
  2. if the scores hold, no new training is needed; if not, train on level k
     (or on a mix of levels), and check that the new model still does well
     on the **lower** levels too;
  3. stop at the first model that generalizes well across the levels. The
     result is a table *model x level*, the baseline against which the SAR
     cues of v0.2 are measured, level by level.
- **Training set fixed and precomputed first**, for full control over the
  data. On-the-fly generation (#29) comes later, seeded from (epoch, index) so
  any training sample can be regenerated.
- **Masks:** single-band `uint8` lossless `.tif`, `0` background, `1` pasted
  region, `2` source region; authentic patches have none. Forgery parameters
  (level, shape, bounding box, source offset, seed) go into the manifest.
  - *Why keep the source:* duplicate-finding methods (block correlation,
    Vertical 52) flag both regions without knowing which one is the copy; the
    source must not count as a false alarm. Training can still use the binary
    target `mask == 1`, changed later without regenerating the data.
  - When the source lies outside the patch (#30, #17), the mask holds only
    `0` and `1`; the source location goes into the manifest (scene, position,
    range/azimuth offset). A `2` always means "source visible in this patch".

### 3.2 Later forgeries

- **Distant-source copy-move** (#30): source in the same scene but outside
  the patch. Seen from the patch it looks like splicing, but physically it is
  the same acquisition (same date, calibration, ENL, noise), so physical cues
  should see little. Except for one SAR nuance: in IW mode the incidence angle
  varies from about 30 deg to 46 deg across the swath, so a source far away in
  **range** has a slightly different radiometry, while one displaced in
  **azimuth** keeps the same incidence.
- **Splicing** (#17): region from another acquisition (other incidence,
  calibration, speckle); where SAR statistics should help most.
- v1: inpainting, fully generated images (GAN, diffusion, autoregressive),
  attacker-side speckle / multilook handling, robustness to publication
  formats.

## 4. Models

- **Decision: supervised U-Net** (#9), trained on forged patches + masks
  with a pixel-wise loss on `mask == 1`; it outputs the map directly, and the
  image score is derived from it (e.g. max or high percentile).
  - *Why not an auto-encoder* (trained on real only, anomaly = reconstruction
    error): it would generalize to unseen forgeries, but a copy-moved region
    is genuine SAR from the same image, likely reconstructed well, hence
    invisible. Our masks are free, so supervision costs nothing.
  - Kept as comparisons: the auto-encoder, and *block correlation*
    (non-deep, finds duplicated speckle) as a sanity baseline.
- **Loss: BCE + Dice computed over the whole batch** (#10), on `mask == 1`.
  - *Why Dice:* plain BCE averages over all pixels, 97 % of which are
    background, so flagging nothing already gives a low loss; Dice (= pixel
    F1) only looks at the predicted and true regions, so "nothing" scores 0.
  - *Why over the batch rather than per patch:* no 0/0 on authentic patches
    (empty mask), no smoothing constant, stable gradients. Large pastes weigh
    more, which is accepted: large forgeries (a hidden base) are the
    priority, and level-1 pastes are all 2-5 %. Per-patch Dice with smoothing
    is the fallback if small pastes are poorly detected at level 4.
  - *Why keep BCE:* it penalizes every falsely flagged pixel, including on
    authentic patches, where batch Dice dilutes false alarms.
- **Input scale**: linear values converted to dB, then **one global
  normalization** (mean / std of the authentic training patches).
  - *Why not per patch:* per-patch standardization would erase the
    radiometric differences between regions, which splicing detection (v0.3)
    relies on.
- **Augmentation: east-west mirror only, compared with none (#37).** The
  mirror turns a patch into a plausible image of the other orbit (see 2.1).
  Not used: north-south mirror and 90 deg rotations (look directions
  Sentinel-1 never has), 180 deg rotation (~24 deg off the descending look
  direction, because of the ~12 deg inclination of the track). Off for the
  first run, then one run with it, all else equal.
- **Optimization: AdamW, learning rate 3e-4 with a cosine decay, batches
  of 8 patches, mixed precision.**
  - *Learning rate* = size of the step the optimizer takes on the weights
    after each batch: too large, the loss oscillates or explodes; too small,
    learning crawls. 3e-4 is a safe default for AdamW on a pretrained
    encoder. The **cosine schedule** lowers it smoothly from 3e-4 (epoch 1)
    to ~1.5e-4 (epoch 20) to ~0 (epoch 40): large steps while far from a
    good solution, finer and finer ones to settle in it. This is also why
    the val loss can still improve late, and why a patience of 8 epochs is
    not too short.
  - *Weight decay 1e-4* (the "W" of AdamW): pulls the weights towards 0,
    a mild protection against over-fitting.
  - *Batch of 8* = patches processed together before one weight update
    (300 updates per epoch for 2400 training patches). Larger batches give a
    steadier update direction but need more GPU memory; 8 fits a T4 (16 GB)
    with 512x512 patches. The batch Dice is computed over these 8 patches.
    Not to be confused with the patience, which is also 8 by coincidence.
  - *Mixed precision*: float16 on the GPU, about 2x faster on a T4.
- **Checkpoint selection on the validation loss** (threshold-free); early
  stopping after 8 epochs without improvement.
  - *Val loss and val MCC can diverge (confidence).* BCE punishes an error
    by its confidence: for a forged pixel, -log(p) is 0.92 at p = 0.4 but
    9.2 at p = 0.0001, while the MCC counts both as one error. A model that
    grows over-confident (right answers 0.9 -> 0.999, its few errors
    0.4 -> 0.0001) sees its val loss rise with an unchanged MCC. Little
    impact on the final scores (thresholds chosen on val), but the
    probabilities stop meaning what they say, which matters for a map shown
    to an analyst: **calibration** is a possible improvement (open question).
- Output size: modern U-Nets pad their convolutions, so a 512x512 input
  gives a 512x512 map (input divisible by 2^depth).
- **Borders:** near the patch edge the network lacks context and the padding
  invents pixels; zero padding even creates an artificial edge it could flag.
  Hence **reflection padding** (#31) and metrics measured with and without a
  border margin (#32).

## 5. Evaluation

- **Metrics: MCC (main) and F1**, for the image score (#7) and the pixel
  map (#8); ROC AUC as a threshold-free complement.
  - *Why MCC:* it uses the whole confusion matrix and stays meaningful under
    class imbalance, extreme at pixel level (a level-1 paste covers 2-5 % of
    the pixels: a model flagging nothing would reach ~97 % accuracy). F1 is
    the usual metric of forgery-localization papers, kept for comparison.
  - Thresholds are chosen on the validation set, then frozen for the test:
    a threshold tuned on the test set would fit those particular patches and
    overstate the score on new images.
  - Image score from the map: max, mean of the top-k % pixels, or a high
    percentile, compared on the validation set (#7). Top-k is more robust to
    isolated pixels, but k must stay below the smallest paste size.
  - Pixel metrics are computed on forged patches only (pixel F1 is undefined
    without positives); false alarms on authentic patches are measured
    separately. **Source pixels (`2`) are ignored** (neither hit nor false
    alarm), in the loss and the metrics, for every method: duplicate-finding
    methods cannot tell the original from the copy. Trade-off: the map may
    also light up the source (a false positive on genuine pixels, larger for
    larger pastes), in exchange for a simpler learning of the duplication cue
    and a fair comparison between methods
    ([architecture_v0.1.md](architecture_v0.1.md), section 7.4).
  - *Prevalence (later, #36):* our sets are 50 % forged, forgeries are rare
    in practice. The model's probabilities embed that 50 % prior, and a small
    false-positive rate turns into many false alarms at low prevalence; scores
    will also be reported re-weighted for realistic prevalences.
  - Realistic targets: still to be set, per difficulty level.
- Results table `patch -> score, IoU` joined with the manifest, so results
  can be sliced by level, paste size, scene.
- Predicted maps saved for the test set only (`uint8` `.tif`).
- `predict(model, config, path)` runs on a single patch, for inspection and
  for the presentation example. The network computes one probability map;
  `predict` returns the image score, the **decision** (score >= image
  threshold), the map, and the **zone** (pixels >= pixel threshold, empty if
  the patch is called authentic), with the settings chosen on val.
- Comparison with the Vertical 52 keypoint-matching method (#12).

### 5.1 First training run (v0.1, level 1, 2026-11-13)

Run `v0.1_level1_resnet34` (default config, T4, ~100 s per epoch): early
stopping at epoch 23, **best checkpoint at epoch 15** (val loss 0.0232),
~38 min. Monitoring metrics on val, fixed threshold 0.5, "max" image score:
image MCC 0.944, mean pixel MCC 0.979 at epoch 15.

- **Loss and MCC pick different "best" epochs:** epoch 6 has the highest
  image MCC (0.988) but not the lowest val loss, so it is not the kept
  checkpoint (see "Val loss and val MCC can diverge" in section 4). Selection
  stays on the loss: threshold-free, and the epoch MCCs are at a fixed 0.5
  threshold that the evaluation replaces anyway.
- **Image MCC below pixel MCC at 0.5:** with the "max" score, one pixel above
  0.5 is enough to call an authentic patch forged, so a few stray pixels turn
  into image-level false alarms. The evaluation chooses the score method and
  the image threshold on val, which is expected to raise the image MCC.
- Mild over-fitting at the end (train loss 0.006, val loss stuck at
  0.025-0.03), handled by early stopping.
- Level 1 (exact copy, speckle included) is the easy case: these scores
  validate the pipeline, not the difficulty of the problem (levels 2-4,
  #25-#27).

**Settings chosen on val** (`eval_level1_trained1/settings.json`, rule of
[architecture_v0.1.md](architecture_v0.1.md), section 5, step 3, and section 7.5):

| Score method | Val image MCC | Threshold (middle) | Margin |
|---|---|---|---|
| max | 0.988 | 1.000 | 0.000 |
| top0.5% | 0.993 | 0.539 | 0.916 |
| top1% | 0.993 | 0.476 | 0.871 |
| top2% | 0.993 | 0.240 | 0.440 |
| **p99.5** | **0.993** | **0.497** | **0.992** |

Kept: **p99.5, image threshold 0.497**; pixel threshold **0.30** (val mean
pixel MCC 0.979).

- **Why the margin rule.** With `max`, even authentic patches hold a pixel at
  ~1 (the over-confidence of section 4): no room at all between the classes
  (margin 0). With the other methods it is the opposite: the classes are
  far apart, e.g. with top0.5% no val patch scores between 0.081 and 0.997.
  A threshold taken at the edge of that gap (0.997) would touch the least
  clear forged patch, and a slightly weaker forgery on new images would be
  missed; the middle (0.539) keeps 0.46 on each side. Four methods tie at
  the best MCC (0.993, ~3 val errors out of 800, on the wrong side whatever
  the threshold): the widest margin decides, not the order of the list.

**Test results (v0.1, level 1)** (`eval_level1_trained1/summary_test.json`):

| Metric | Test |
|---|---|
| Image MCC / F1 / ROC AUC | **1.000** / 1.000 / 1.000 (400 TP, 0 FP, 0 FN, 400 TN) |
| Pixel MCC, mean per forged patch | **0.993** (0.993 without the 16-px border) |
| Pixel F1, mean per forged patch | 0.993 |
| Pixel MCC / F1, global | 0.993 / 0.993 |
| False alarms on authentic patches (fraction of pixels) | 3.5e-7, i.e. ~0.09 pixel per patch |

Per test scene: ds_forest_1 pixel MCC 0.992, ds_forest_9 (mirrored) 0.994,
every forged patch detected in both. Mean and global pixel MCC are nearly
equal: the localization does not depend on the paste size.

### 5.2 Across levels (2026-11-14)

Plan of section 3.1: each model is scored on every level, with the settings
of its own level frozen. Test results, mean pixel MCC per forged patch:

| Model (trained on) | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| Level 1 | image 1.000 / pixel 0.993 | image 0.577 / pixel 0.463 | image 0.869 / pixel 0.579 |
| Level 2 | image 0.998 / pixel 0.989 | image **0.998** / pixel **0.981** | image 1.000 / pixel 0.978 |
| Level 3 | image 0.998 / pixel **0.996** | image 0.998 / pixel **0.992** | image 0.998 / pixel **0.991** |

- **The level-1 model does not generalize to varied shapes.** On level 2:
  200 of the 400 forged test patches missed, no false alarm on the 400
  authentic ones (the same patches as at level 1), AUC 0.855, pixel MCC 0.463
  (global 0.642). It learned the rectangle, not the duplication: a model
  had to be trained on level 2.
- **The level-1 model looks better on level 3 than on level 2, for the same
  reason** (`eval_level3_trained1/`, level-1 settings frozen): image MCC
  0.869 (344 TP, 56 FN, 0 FP), pixel MCC 0.579. Each of the 2-4 pastes is a
  rectangle one time in three, so most level-3 patches hold at least one,
  and one rectangle found is enough to call the patch forged. Split by the
  number of rectangles in the patch (from `metrics_by_shape.csv`):

  | Rectangles in the patch | 0 | 1 | 2 | 3 |
  |---|---|---|---|---|
  | Forged patches (test) | 102 | 173 | 96 | 28 |
  | Detected | 54 % | 95 % | 100 % | 100 % |
  | Pixel MCC | 0.30 | 0.62 | 0.74 | 0.81 |

  The image decision is rescued by the rectangles, not the localization:
  the ellipses and polygons stay mostly missed in the map, and the pixel
  MCC grows with the share of rectangles. Detection also grows with the
  number of pastes (78 % with 2, 89 % with 3, 91 % with 4).
- **Level-2 run** (`v0.1_level2_resnet34`): all 40 epochs (~102 s each),
  best checkpoint at epoch 36 (val loss 0.0208; at 0.5 and "max": val
  image MCC 0.998, pixel MCC 0.981).
- **Level-2 settings** (`eval_level2_trained2/settings.json`): top0.5%,
  image threshold 0.043, pixel threshold 0.45, val image MCC 1.0, but a
  **margin of only 0.066** (level 1: 0.992). The interval of equally good
  thresholds is about (0.010, 0.076]: the least visible forged val patch
  scores ~0.08, barely above the authentic ones. The threshold is therefore
  low and tight, which is the likely cause of the one missed test patch, and
  these frozen settings will meet the other levels with little room. The
  selection rule (section 5, architecture 7.5) is **kept unchanged**: it was
  fixed before these results, and changing it in view of a test result would
  tune it on the test set.
- On its own level, the level-2 model does almost as well as the level-1
  model on level 1, on a harder task. Pixel MCC 0.981 vs 0.993: curved and
  irregular shapes have more edge per area, and pixel errors sit on the
  edges (to be checked in `metrics_by_shape.csv`).
- **The level-2 model generalizes down to level 1** (`eval_level1_trained2/`,
  level-2 settings frozen): image MCC 0.998 (399 TP, 1 FN, 0 FP, 400 TN),
  AUC 1.000, pixel MCC 0.989 (global 0.990), false alarms ~0.01 pixel per
  authentic patch. Almost the level-1 model's own score (1.000 / 0.993),
  despite the narrow margin of its threshold.
- **The level-2 model also generalizes up to level 3** (`eval_level3_trained2/`),
  although it only saw one paste per patch: image MCC 1.000 (400 TP, 0 FP),
  pixel MCC 0.978 (global 0.978). It finds every paste, not only one: a
  missed paste out of three would pull the pixel MCC of the patch far lower.
  Pixel MCC by number of pastes: 0.983 (2), 0.980 (3), 0.971 (4); 0.980 when
  no paste touches the border, 0.958 when one does (42 patches).
- **Consequence for v0.2.** With one paste of any simple shape learned, the
  image decision is saturated on levels 1-3 (MCC 0.998-1.000) and the
  pixel MCC stays above 0.97: these levels leave little room to measure the
  added value of the SAR cues. That room has to come from harder forgeries:
  sizes from 1 % to 25 %+ (level 4, #27), a distant source (#30), splicing
  (#17).
- **Level-3 run** (`v0.1_level3_resnet34`): early stopping at epoch 36,
  best checkpoint at epoch 28 (val loss 0.0126; at 0.5 and "max": val image
  MCC 0.993, pixel MCC 0.991), ~1 h at ~103 s per epoch. Val figures of
  different levels are not comparable (different val sets): the table
  above is.
- **Level-3 settings** (`eval_level3_trained3/settings.json`): p99.5, image
  threshold 0.500, pixel threshold 0.45, val image MCC 1.0, margin 0.9999:
  on val the two classes sit at the two ends of [0, 1], unlike the level-2
  model (margin 0.066).
- **Level-3 model on level 3:** image MCC 0.998 (400 TP, **1 FP**, 0 FN,
  399 TN), AUC 1.000, pixel MCC **0.991** (global 0.991), against 0.978 for
  the level-2 model on the same test: having seen several pastes improves
  the localization, and it no longer degrades with their number (0.992 /
  0.991 / 0.991 for 2 / 3 / 4 pastes; 0.987 when a paste touches the
  border). The one false alarm is an authentic patch with a flagged region
  (~2,000 pixels in all over the 400 authentic patches, against ~0 for the
  other models): the first false-alarm candidate of the gallery, to be
  looked at.
- **Level-3 model on level 1** (`eval_level1_trained3/`, level-3 settings
  frozen): image MCC 0.998 (400 TP, 1 FP), pixel MCC **0.996** (global
  0.996), above the level-1 model on its own level (0.993). The false alarm
  is the same authentic patch as on level 3 (the authentic patches are the
  same at every level; same flagged fraction, 2.05e-5).
- **Level-3 model on level 2** (`eval_level2_trained3/`): image MCC 0.998
  (400 TP, the same 1 FP), pixel MCC **0.992** (global 0.992), against 0.981
  for the level-2 model on its own level. By shape: rectangle 0.996,
  ellipse 0.993, polygon 0.988.

**Conclusion: the level-3 model is the reference model of v0.1.**

- *Image decision: equivalent.* Both the level-2 and the level-3 models make
  one error out of 800 test patches on every level. The level-2 model's
  error is a miss, made with a threshold 0.03 away from the scores of
  either class (margin 0.066); the level-3 model's error is a false alarm on
  one authentic patch, always the same, with a threshold in the middle of
  an empty [0, 1] on val (margin 0.9999); the patch itself is to be looked
  at. The two margins do **not** show that the models generalize very
  differently: a margin is set by the single val patch closest to the
  boundary (one hard forged patch at ~0.08 is enough to shrink it), the two
  were measured on different val sets (level 2 vs level 3) with different
  score methods (top0.5% vs p99.5), and the level-3 model still made a test
  error despite its margin. Both separate the classes almost perfectly on
  test (AUC 1.000).
- *Localization: better on every level.* Pixel MCC 0.996 / 0.992 / 0.991 on
  levels 1 / 2 / 3, against 0.989 / 0.981 / 0.978 for the level-2 model;
  it no longer drops with the number of pastes nor at the border.
- The level-2 model was the first one to generalize (plan of section 3.1),
  but the level-3 model had already been trained and does better on every
  level: it is kept, and no further level is trained for v0.1.
- Limits: the gaps are small (about 0.01 of pixel MCC, one test error out of
  800), measured on two test scenes; they show a trend, not a proven
  ranking.

## 6. Presentation example

Built **by hand** (#11): choose a forest scene and a base, paste forest
over it at scene level, cut the 512x512 window, run `predict`. At 10 m a
2-3 km base spans 200-300 px, enough to be visible.

## 7. Infrastructure

- **All code runs on Google Colab** (GPU); the PC is used for editing, git
  and the upstream preprocessing. Colab installs with pip on its own VM,
  nothing is installed on the PC.
- Notebooks are thin drivers of `.py` modules, versioned in the repo and
  opened in Colab from GitHub. Code layout: one folder per feature in
  `features/` (`tiling`, `copy_move`, `unet`, `evaluation`, plus the
  `sar_io` library), each with its README.
- Each Colab step reads its input from Drive and writes its output back
  (patches zip, dataset zip, run folder), so steps can run in separate
  sessions.
- **Tests on GitHub Actions** at every push: synthetic data, CPU, nothing on
  the user's PC ([github_actions.md](github_actions.md)).
- Data and checkpoints on Google Drive (paid plan: the 10 scenes take ~20 GB);
  inputs copied to the VM's local disk before training; checkpoints written
  back to Drive so a disconnected session can resume.
- Cost: a v0.1 training run is estimated at 1-2 h on a free T4. If needed,
  Colab compute units cost ~$10 per 100; a T4 uses ~1.2-1.8 units/h, an A100
  ~5.4 (third-party figures, March 2026): faster GPUs cost about the same
  per run.

## 8. Later extensions (v1)

Inpainting (#19), generated images (#20), metadata (#21), robustness to
formats (#22), attacker-side speckle / multilook (#23), whole-scene
inference with an overlapping sliding window (#33), other polarisations
(#34), other sensor: ICEYE GRD, X-band, radar geometry (#35), evaluation at
realistic prevalence (#36), telling the pasted region from its source (#38).

## 9. Open questions

- Realistic metric targets, per difficulty level (#6-#8).
- Calibration of the predicted probabilities (does 0.7 mean "70 % likely
  forged"?), if the val loss rises while the val MCC holds (section 4).
