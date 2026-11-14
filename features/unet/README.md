# unet

Issues #9, #10, #31. U-Net with a ResNet-34 encoder pretrained on ImageNet
(`segmentation_models_pytorch`), 1 input channel (VV), 1 output map.

- **Module** `unet.py`: `train` (with automatic resume), `load_model`,
  `predict` (one patch -> score, decision forged / authentic, probability
  map, zone called forged; with the settings chosen on val once the
  evaluation has run), `iterate_predictions` (a whole split, for the
  evaluation), `build_model`, `loss_function`.
- **Notebook** `unet.ipynb` drives the training on Colab (**GPU runtime**).
  Its last cell saves the training curves as `training_curves.png` in the run
  folder, next to `log.csv` (they can be redrawn from `log.csv` at any time).

| | Where | Content |
|---|---|---|
| Reads | `MyDrive/aletheia/data/dataset_v0.1_level1.zip`, unzipped to the VM | output of copy_move |
| Writes | `MyDrive/aletheia/runs/<run name>/` | `config.json`, `log.csv`, `last.pt`, `best.pt` |

Choices (rationale in `docs/report.md`, section 4):
- input: linear values converted to dB, then **one global normalization**
  (mean / std of the training patches), never per patch, to keep the
  radiometric differences splicing detection will need;
- **reflection padding** in every padded convolution (#31);
- loss **BCE + Dice over the batch**, source pixels ignored;
- AdamW, cosine learning-rate schedule, mixed precision on GPU;
- best checkpoint = lowest **validation loss**; early stopping after 8 epochs
  without improvement; `last.pt` saved every epoch for resuming;
- `config.json` records the configuration, the normalization and the git
  commit of the code.

Augmentation (`augment`, off by default, compared in #37): east-west mirror
only, half of the time, which turns a patch into a plausible image of the
other orbit. No north-south mirror nor rotation (impossible look directions).

Known limits: the validation metrics in
`log.csv` use a fixed 0.5 threshold and are for monitoring only; the real
thresholds are chosen by the evaluation.
