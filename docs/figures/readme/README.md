# Figures of the root README

Empty placeholders until filled; replace each file, keeping its name.

| File | Content | Source |
|---|---|---|
| `hero_example.png` | Forged test patch with mixed shapes: input, ground truth, predicted map | evaluation gallery / predictions of `v0.1_level3_resnet34`, `eval_level3_trained3`, patch 000200 |
| `level1.png`, `level2.png`, `level3.png` | One forged example per level: patch + mask, as drawn by `show_forgery` (level 2: preferably an ellipse or a polygon; level 3: 3-4 pastes of different shapes; not necessarily the same patch) | `MyDrive/aletheia/data/figures/forgeries_level<L>_<patch_id>.png` (cell 6 of the `copy_move` notebook) |
| `false_alarm.png` | The only false alarm of the level-3 model, authentic patch 000343 | `eval_level3_trained3/gallery/`, the file of 000343 ending in `_FP.png` |
