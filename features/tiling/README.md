# tiling

Issue #28. Cuts the preprocessed Sentinel-1 scenes into 512x512 authentic
patches and assigns every scene to train / val / test.

- **Module** `tiling.py`: `find_scenes`, `assign_splits`, `valid_windows`,
  `tile_scenes`; command line `python features/tiling/tiling.py --scenes-dir ... --out-dir ...`.
- **Notebook** `tiling.ipynb` drives the module on Colab (CPU runtime is
  enough). It is not a mirror of the module.

| | Where | Content |
|---|---|---|
| Reads | `MyDrive/aletheia/data/scenes/<scene>/` | one VV GeoTIFF (+ its SNAP graph XML, not read) |
| Writes | VM `/content/patches/`, then `MyDrive/aletheia/data/patches_v0.1.zip` | `real/<patch_id>.tif`, `patches.csv` |

`patches.csv`: `patch_id, scene, row, col, split` (row / col = top-left
corner in the scene, kept for the incidence-angle study #30).

Rules:
- crop, never resize; non-overlapping grid of windows;
- a window is rejected if more than 5 % of its pixels are no-data (0 or NaN):
  the empty borders of terrain correction are rejected, isolated zeros kept;
- 200 patches drawn at random per scene (seeded by `(seed, scene index)`);
- split by scene, default 2 val + 2 test scenes drawn with the seed, the rest
  train; `forced_splits` chooses scenes by hand.

Known limits: the no-data rule counts all zeros, it does not distinguish a
contiguous block from scattered zeros (the 5 % threshold does it in practice).
A whole scene is read into memory (~2 GB in float32: fine on Colab).
