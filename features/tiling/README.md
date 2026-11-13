# tiling

Issue #28. Cuts the preprocessed Sentinel-1 scenes into 512x512 authentic
patches and assigns every scene to train / val / test.

- **Module** `tiling.py`: `find_scenes`, `assign_splits`, `valid_windows`,
  `tile_scenes`; command line `python features/tiling/tiling.py --scenes-dir ... --out-dir ...`.
- **Notebook** `tiling.ipynb` drives the module on Colab (CPU runtime is
  enough). It is not a mirror of the module.

| | Where | Content |
|---|---|---|
| Reads | `MyDrive/aletheia/data/scenes/<scene>/` | one VV image, at any depth: a GeoTIFF, or a BEAM-DIMAP band (`Sigma0_VV.hdr` + `.img`, e.g. the `.data/` folder of SNAP copied as is; the `.dim` is not needed). Not both: a folder with two images is refused. Its SNAP graph XML is not read. `asc` or `desc` in the folder name |
| Writes | VM `/content/patches/`, then `MyDrive/aletheia/data/patches_v0.1.zip` | `real/<patch_id>.tif`, `patches.csv` |

`patches.csv`: `patch_id, scene, row, col, orbit, mirrored, geometry, split`
(row / col = top-left corner in the original, unmirrored scene, kept for the
incidence-angle study #30; `orbit` = of the acquisition, `geometry` = look
geometry of the patch, the opposite one when `mirrored`).

Rules:
- crop, never resize; non-overlapping grid of windows;
- a window is rejected if more than 5 % of its pixels are no-data (0 or NaN):
  the empty borders of terrain correction are rejected, isolated zeros kept.
  The SNAP graph also sets the open sea to 0 ("Mask out areas without
  elevation"), so sea patches are rejected too: switch that option off
  before tiling maritime scenes;
- 200 patches drawn at random per scene (seeded by `(seed, scene index)`);
- orbits balanced: if the scenes are not 50/50 ascending / descending, scenes
  of the majority orbit are mirrored east-west (`mirror_east_west`: on a
  north-up image it maps the ascending look direction, ~78 deg, onto the
  descending one, ~282 deg); a mirrored scene replaces its original;
- split by scene, default 2 val + 2 test scenes drawn with the seed, one of
  each geometry, the rest train; `forced_splits` chooses scenes by hand.

Known limits: the no-data rule counts all zeros, it does not distinguish a
contiguous block from scattered zeros (the 5 % threshold does it in practice).
A whole scene is read into memory (~2 GB in float32: fine on Colab).
