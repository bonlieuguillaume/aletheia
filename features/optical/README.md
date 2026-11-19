# optical

Issue #47, optical control. Does the copy-move model rely on SAR (speckle,
radar texture), or only on duplication? The same forgeries are made on
Sentinel-2 images of the two SAR test scenes, and the SAR model is evaluated
on them as it is: no retraining, settings frozen on its SAR val. Design and
reading: `docs/report.md`, section 5.3.

- **Module** `optical.py`: `s2_gray`, `sar_reference`, `to_sar_db`
  (histogram matching or affine), `add_speckle`, `pseudo_sar_scene`,
  `build_optical_datasets`.
- **Notebook** `optical.ipynb` drives the module on Colab (CPU runtime).
  It is not a mirror of the module. The evaluation itself is done by
  `features/evaluation/evaluation.ipynb` with `DATA_SOURCE = "_optical"` or
  `"_optical_speckle"`.

| | Where | Content |
|---|---|---|
| Reads | `MyDrive/aletheia/data/scenes_S2/` (or one folder below) | the Sentinel-2 L2A `.SAFE` products, by name; only `*_B02/B03/B04_10m.jp2` are copied |
| Reads | `MyDrive/aletheia/data/dataset_v0.1_level3.zip` | authentic training patches of the SAR reference model (read inside the zip) |
| Writes | `MyDrive/aletheia/data/dataset_v0.1_level<L>_optical.zip`, `..._optical_speckle.zip` | same layout as the SAR datasets (`real/`, `forged/`, `masks/`, `manifest.csv`) + `conversion.json` |
| Figures | `MyDrive/aletheia/data/figures/` | `optical_sources_<id>.png`, `forgeries_level3_optical[_speckle]_<id>.png` |

Pipeline per scene:
1. grey = **float mean of B02, B03, B04** (10 m); a pixel is valid if all
   three bands are > 0 (0 = no-data of Sentinel-2, e.g. a tile cut by the
   swath). The L2A offset (+1000) is a constant shift, invisible to step 3;
2. `_optical_speckle` only: grey x Gamma(4.4, 1/4.4), one draw per pixel
   (fully developed speckle, ENL of Sentinel-1 IW GRD), seeded per scene;
3. grey -> dB: **histogram matching** onto the valid pixels of 200 SAR
   training patches (default), or **affine** (mean / std only). Statistics
   over the whole scene, never per patch. Speckle is added before, so the
   histogram is the SAR one in both sources;
4. dB -> linear, no-data back to 0; written as one scene `.tif`.

Then `tiling` (200 patches per scene, the 5 % no-data rule, **all scenes in
test**, **no east-west mirror**: it only balances radar look directions) and
`copy_move` (levels 1-3, same generators and seed as SAR), both unchanged.
Inside the model, dB conversion and normalization apply as for SAR: the
network sees the optical patches on the same scale as SAR ones.

JPEG 2000 is read with **Pillow** (its wheels ship OpenJPEG): no new
dependency.

Known limits:
- a control on the nature of what was learned, not a cross-sensor
  benchmark: S1 and S2 dates differ (1 to 6 days), and optical shows other
  structures (sun shadows, colour turned into grey);
- clouds are kept on purpose (a realistic optical case), so some patches
  show clouds or their shadows;
- the S2 tiles may overlap a little a SAR training scene (e.g. T21NZE near
  ds_forest_10, across the Maroni); accepted for this small experiment,
  given the change of sensor;
- the manifest still has an `orbit` column (set to "desc", unused).
