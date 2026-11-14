# external_image

Part of #11 (presentation example, Vertical 52 / Kronstadt scenario). Runs a
trained model on **an image from outside the dataset** (PNG, JPG), typically
a screenshot of a published image suspected of tampering.

**Out-of-domain test, for illustration only.** The models are trained on
Sentinel-1 VV amplitude; a screenshot is usually an optical image,
resampled and often compressed. Its result shows what the model reacts to;
it has no metric value and is never reported as one.

- **Module** `external_image.py`: `read_image_gray`, `to_pseudo_sar`,
  `predict_large_image`, `analyse_image`, `plot_result`.
- **Notebook** `external_image.ipynb` drives the module, in **one cell**:
  set `IMAGE_PATH` (and `RUN_NAME`, the level-3 reference model by default),
  run it. GPU runtime preferred.

| | Where | Content |
|---|---|---|
| Reads | `IMAGE_PATH` on Drive (default `MyDrive/aletheia/external/`) | the image |
| Reads | `MyDrive/aletheia/runs/<run>/` | `best.pt`, `config.json`, `eval_level<T>_trained<T>/settings.json` |
| Writes | `MyDrive/aletheia/runs/<run>/external/` | `<image name>.png` (figure), `<image name>.json` (score, decision, settings, commit) |

Choices:
- **grey = float mean of R, G, B**, no rounding and no binarization, which
  would destroy the texture;
- **domain matching**: the grey levels are given the mean and standard
  deviation, in dB, of the training patches (`normalization_mean` /
  `normalization_std` of `config.json`), over the whole image, then go
  through the model's own pipeline (dB, global normalization). This
  amounts to standardizing the image as a whole; it matches the range, not
  the speckle statistics;
- **512x512 windows overlapping by half** (the size the thresholds were
  chosen for). Each pixel keeps the **highest** probability among its
  windows: copy-move is only visible to a window holding both the copy and
  its source, so one window seeing it is enough. Reflection padding only
  when the image is smaller than a window;
- **image decision**: forged if at least one window reaches the image
  threshold frozen on val; image score = highest window score. The zone is
  the pixels above the frozen pixel threshold, reported only for an image
  called forged (as `unet.predict`); the binary map is drawn anyway, for
  information.

Known limits:
- out of domain: optical texture, no speckle, unknown pixel size (the
  model learned objects at 10 m pixel spacing) and compression artefacts
  the model never saw; a false alarm or a miss says little about SAR
  performance;
- a copy whose source lies more than one window away (> 512 px) cannot be
  seen as a duplication (same limit as #30);
- the maximum over overlapping windows raises the false-alarm rate compared
  with one patch: more windows, more chances for one to fire.
