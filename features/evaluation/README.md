# evaluation

Issues #6, #7, #8, #32. Turns probability maps and masks into metrics. The
module knows nothing about the model. Full procedure:
`docs/architecture_v0.1.md`.

- **Module** `evaluation.py`: `mcc`, `f1`, `image_score`,
  `sample_statistics`, `collect_statistics`, `best_image_threshold` (middle
  of the interval of equally good thresholds, and its width = margin),
  `choose_thresholds` (on val; ties between score methods broken by margin),
  `compute_metrics` (on test, with frozen settings), `metrics_by_group`,
  `plot_sample`.
- **Notebook** `evaluation.ipynb` drives the evaluation of a trained run on
  Colab (**GPU runtime**), the gallery and the single-patch prediction.

| | Where | Content |
|---|---|---|
| Reads | dataset zip, `runs/<run name>/best.pt` | |
| Writes | `runs/<run name>/eval_level<D>_trained<T>/`: the run trained on level T, evaluated on level D. `eval_level1_trained1/` = its own level (settings chosen on its val). On another level, the weights AND the settings of level T are kept as they are (rule of the project). `_retuned` suffix: weights of level T, settings chosen again on the val of level D (option kept, not used; see `docs/architecture_v0.1.md`, section 5) | `settings.json` (chosen on val), `summary_test.json`, `results_test.csv`, `pred_maps/<sample_id>.tif` (test only, uint8), `gallery/*.png` (figures of cell 7), `metrics_by_<group>.csv` (cell 6: per level, scene, shape, border contact, number of pastes), `predictions/*.png` (cell 8) |

Rules:
- main metric MCC, second F1, ROC AUC as a threshold-free complement;
- the image score (max / top-k % / percentile), the image threshold and the
  pixel threshold are chosen on **val** (best MCC), then frozen for **test**;
- pixel metrics on forged patches only, against `mask == 1`; source pixels
  (`mask == 2`) ignored; reported as mean per patch and global, and without a
  16-px border (#32);
- authentic patches: fraction of pixels flagged by mistake.

Known limits: pixel thresholds are searched on a 0.05 grid. Evaluating at a
realistic prevalence of forgeries is issue #36.
