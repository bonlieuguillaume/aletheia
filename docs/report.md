# Aletheia: design report

Living document: every design choice of the project, **why** it was made, and
what remains open. It is updated as the work progresses; the task list itself
lives in the [GitHub Project](https://github.com/users/bonlieuguillaume/projects/5).

Status: v0.1 code written (tiling, level-1 copy-move, U-Net, evaluation),
not yet run on real data; scenes being prepared. Last update: 2026-11-12.
Architecture and metric procedure of v0.1: [architecture_v0.1.md](architecture_v0.1.md).

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
  regions, ascending/descending orbits, seasons.
  - *Known limit:* a model trained on forest only is not expected to
    generalize to other land covers.
- No speckle filter, no extra multilook. IW GRD is nevertheless **already
  multilooked by ESA** (about 5x1 looks, ENL ~4.4): the speckle is not
  single-look, which matters for any speckle statistic (v0.2).
- Preprocessed by the user with rosarium / SNAP on their PC (calibration,
  terrain correction). Each scene comes with the **XML of its SNAP graph**,
  from which we record: calibration (sigma0 / gamma0 / beta0), scale
  (linear / dB), terrain-correction resampling method, pixel spacing.
  *To fill in once the scenes are delivered.*
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
  landscapes never seen in training.
  - *Possible improvement, not planned (too time-consuming):* scene-level
    cross-validation, e.g. 5 rounds where 2 different scenes serve as test
    and the 8 others for training, averaging the scores. Training would see
    more scenes and the test score would rest on all 10 scenes, at the cost of
    5 trainings.
  - *Why:* a network with millions of parameters can memorize thousands of
    images (Zhang et al., 2017, fit random labels on CIFAR-10); with
    patch-level splits, it would recognize scenes instead of forgeries.
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
- **No augmentation by default.** Flips and rotations change the apparent
  radar look direction (shadow and layover on the other side), which never
  happens in real data: to be tested, not assumed.
- **Checkpoint selection on the validation loss** (threshold-free); early
  stopping after 8 epochs without improvement.
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
    alarm), for every method: duplicate-finding methods cannot tell the
    original from the copy.
  - *Prevalence (later, #36):* our sets are 50 % forged, forgeries are rare
    in practice. The model's probabilities embed that 50 % prior, and a small
    false-positive rate turns into many false alarms at low prevalence; scores
    will also be reported re-weighted for realistic prevalences.
  - Realistic targets: still to be set, per difficulty level.
- Results table `patch -> score, IoU` joined with the manifest, so results
  can be sliced by level, paste size, scene.
- Predicted maps saved for the test set only (`uint8` `.tif`).
- `predict(model, path)` runs on a single patch, for inspection and for the
  presentation example.
- Comparison with the Vertical 52 keypoint-matching method (#12).

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
realistic prevalence (#36).

## 9. Open questions

- Realistic metric targets, per difficulty level (#6-#8).
- SNAP graph parameters, once the XMLs are delivered.
