# copy_move

Issues #5, #24 (level 1), #25 (level 2) and #26 (level 3); level 4 (#27) to come.
Makes one forged twin of every authentic patch, with its ground-truth mask,
and writes the dataset manifest.

- **Module** `copy_move.py`: `copy_move_level1`, `copy_move_level2` (with
  `random_shape`, `ellipse_mask`, `random_polygon`, `polygon_mask`),
  `copy_move_level3` (with `place_boxes`),
  `build_dataset`, `show_forgery`, `LEVELS`; command line
  `python features/copy_move/copy_move.py --dataset-dir ...`.
- **Notebook** `copy_move.ipynb` drives the module on Colab (CPU runtime).

| | Where | Content |
|---|---|---|
| Reads | `MyDrive/aletheia/data/patches_v0.1.zip` | output of tiling |
| Writes | VM `/content/dataset_level<L>/`, then `MyDrive/aletheia/data/dataset_v0.1_level<L>.zip` | `forged/`, `masks/`, `manifest.csv` (+ `real/`, `patches.csv`) |
| Figures | `MyDrive/aletheia/data/figures/` | `forgeries_level<L>_<patch_id>.png` (3 examples per level) |

Level 1: one rectangle, 2-5 % of the patch, aspect ratio 0.5-2, source and
target in the patch, not overlapping, both >= 32 px from the border, exact
copy (hard edges, no transform).

Level 2: the same, but the shape is drawn among a rectangle, an ellipse (or
disc) and a slightly irregular polygon (5-8 vertices around the centre, each
at 0.7-1 of the half-size), and source and target may touch the border. The
**shape itself** covers 2-5 % of the patch (its box is rescaled for that);
only the pixels inside the shape are copied. Every level uses the same
authentic patches and the same split, so the levels are directly comparable.

Level 3: 2 to 4 pastes per patch (equal odds), each drawn as at level 2
(2-5 % each, so 4-20 % of the patch in all). No paste overlaps another paste
nor any source; every source is read from the untouched authentic patch. The
mask is the union of all the pastes (1) and of all their sources (2).

Mask: uint8, `0` background, `1` pasted, `2` source. Authentic patches have
no mask.

`manifest.csv`, one row per sample: `sample_id` (`<patch_id>_real` /
`<patch_id>_forged`), `patch_id`, `label` (0 / 1), `file`, `mask_file`,
`scene`, `row`, `col`, `split`, and for forged samples `level`, `seed`,
`shape`, `bbox_row_min / col_min / row_max / col_max` (box of the pasted
region, max excluded), `src_offset_row / col` (source minus target),
`paste_fraction` (pixels of the paste / pixels of the patch), and from level 2
`touches_border` (1 if the source or the target box touches the border). From
level 3: `n_pastes`; `shape` lists every paste's shape joined by `+`; the
`bbox_*` columns enclose all the pastes; `src_offset_*` are empty (one offset
per paste); `paste_fraction` is the total; `pastes` holds every paste's
details as JSON text (shape, box, source offset, fraction).

Every forgery uses its own generator seeded by `(seed, patch index)`: same
seed, same dataset; any single forgery can be regenerated.
