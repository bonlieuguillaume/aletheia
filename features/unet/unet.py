"""U-Net with a pretrained ResNet encoder: training and prediction (#9, #10, #31).

- Model: `segmentation_models_pytorch.Unet`, ResNet-34 encoder pretrained on
  ImageNet, 1 input channel, 1 output channel (logit of "pasted pixel").
  All padded convolutions are switched to reflection padding (#31).
- Loss: BCE + Dice computed over the whole batch, source pixels ignored.
- Data: the dataset folder written by `tiling` + `copy_move` (manifest.csv).
- Checkpoints and logs go to a run folder on Google Drive; training resumes
  from `last.pt` if the Colab session was cut.

The rationale of every choice is in docs/report.md (section 4) and the flow
of training / validation / test in docs/architecture_v0.1.md.
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.evaluation.evaluation import confusion_counts, image_score, mcc  # noqa: E402
from features.sar_io.sar_io import MASK_PASTED, MASK_SOURCE, read_manifest, read_tif  # noqa: E402

# Default configuration. A run saves the configuration it used in
# `<run_dir>/config.json`, so every result can be traced back.
DEFAULT_CONFIG = {
    "encoder": "resnet34",        # ResNet encoder of the U-Net
    "pretrained": True,           # ImageNet weights for the encoder
    "reflect_padding": True,      # issue #31
    "input_scale": "linear",      # "linear" (converted to dB here) or "db"
    "augment": False,             # east-west mirror, see SARPatchDataset
    "batch_size": 8,
    "epochs": 40,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "early_stopping_patience": 8, # epochs without val improvement before stopping
    "num_workers": 2,
    "mixed_precision": True,      # float16 on the GPU: ~2x faster on a T4
    "seed": 0,
    "score_method": "max",        # replaced by the one chosen on val (evaluation)
}

EPSILON = 1e-6  # avoids log(0) on no-data / zero pixels


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def to_model_scale(patch, input_scale):
    """Convert raw SAR values to the scale the network sees.

    SAR amplitudes span several orders of magnitude; in dB (log scale) they
    become roughly Gaussian-looking, which suits a network far better.
    Zeros (no-data or very dark pixels) are clamped to EPSILON before the log.
    """
    values = np.nan_to_num(patch.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if input_scale == "linear":
        values = 10.0 * np.log10(np.maximum(values, EPSILON))
    elif input_scale != "db":
        raise ValueError(f"input_scale must be 'linear' or 'db', got {input_scale}")
    return values


def compute_normalization(dataset_dir, manifest, input_scale, max_patches=200, seed=0):
    """Mean and standard deviation of the training patches, in model scale.

    One global normalization for the whole dataset, NOT per patch: a
    per-patch normalization would erase radiometric differences between
    regions, which are exactly what splicing detection will rely on (v0.3).
    Computed on authentic training patches only (never on val / test).
    """
    train_real = manifest[(manifest["split"] == "train") & (manifest["label"] == 0)]
    rng = np.random.default_rng(seed)
    n = min(max_patches, len(train_real))
    chosen = rng.choice(len(train_real), size=n, replace=False)

    sums, sums_of_squares, count = 0.0, 0.0, 0
    for index in chosen:
        file = train_real.iloc[index]["file"]
        values = to_model_scale(read_tif(Path(dataset_dir) / file), input_scale).astype(np.float64)
        sums += values.sum()
        sums_of_squares += (values ** 2).sum()
        count += values.size

    mean = sums / count
    std = np.sqrt(sums_of_squares / count - mean ** 2)
    return float(mean), float(std)


class SARPatchDataset(Dataset):
    """Samples of one split: image, target, valid-pixel map.

    For each sample, returns three tensors of shape (1, H, W):
    - image: the normalized patch;
    - target: 1 where the pixel was pasted (mask == 1), 0 elsewhere;
    - valid: 0 on source pixels (mask == 2, ignored by loss and metrics),
      1 elsewhere. Authentic patches have target = 0 and valid = 1 everywhere.

    Augmentation (off by default, compared in an experiment): east-west
    mirror only, half of the time. On a north-up image it turns an ascending
    look direction (~78 deg) into exactly the descending one (~282 deg), see
    `tiling.mirror_east_west`. North-south mirrors and 90 deg rotations are
    NOT used: they give look directions Sentinel-1 never has.
    """

    def __init__(self, dataset_dir, manifest, split, mean, std, input_scale, augment=False):
        self.dataset_dir = Path(dataset_dir)
        self.rows = manifest[manifest["split"] == split].reset_index(drop=True)
        self.mean = mean
        self.std = std
        self.input_scale = input_scale
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def load(self, index):
        """Return (row, normalized image, mask or None) as numpy arrays."""
        row = self.rows.iloc[index]
        patch = read_tif(self.dataset_dir / row["file"])
        image = (to_model_scale(patch, self.input_scale) - self.mean) / self.std

        mask = None
        if row["label"] == 1:
            mask = read_tif(self.dataset_dir / row["mask_file"])
        return row, image.astype(np.float32), mask

    def __getitem__(self, index):
        _, image, mask = self.load(index)

        if mask is None:
            target = np.zeros(image.shape, dtype=np.float32)
            valid = np.ones(image.shape, dtype=np.float32)
        else:
            target = (mask == MASK_PASTED).astype(np.float32)
            valid = (mask != MASK_SOURCE).astype(np.float32)

        if self.augment:
            # East-west mirror (columns reversed), the same for the image and
            # its target / valid maps. torch random numbers: correctly seeded
            # in every DataLoader worker.
            if torch.rand(1).item() < 0.5:
                image, target, valid = image[:, ::-1], target[:, ::-1], valid[:, ::-1]

        # (H, W) -> (1, H, W): one channel. ascontiguousarray undoes the flips' views.
        tensors = [torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0) for a in (image, target, valid)]
        return tensors[0], tensors[1], tensors[2]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def use_reflection_padding(model):
    """Switch every padded convolution of the model to reflection padding (#31).

    By default convolutions pad with zeros: near the patch border the network
    sees a band of zeros next to speckle, an artificial edge it could learn
    to flag as a forgery. Reflection mirrors the image instead.
    """
    n_changed = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and any(p > 0 for p in module.padding):
            module.padding_mode = "reflect"
            n_changed += 1
    return n_changed


def build_model(config):
    """Create the U-Net described by the configuration."""
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name=config["encoder"],
        encoder_weights="imagenet" if config["pretrained"] else None,
        in_channels=1,   # one SAR band (VV); smp merges the RGB weights of the first layer
        classes=1,       # one output map: logit of "pasted pixel"
    )
    if config["reflect_padding"]:
        use_reflection_padding(model)
    return model


def loss_function(logits, target, valid):
    """BCE + Dice over the whole batch; source pixels (valid == 0) ignored.

    - BCE: average over valid pixels of the per-pixel binary cross-entropy.
      Penalizes every flagged pixel, including on authentic patches.
    - Dice: computed over the whole batch (docs/report.md, section 4). Every
      batch contains forged patches, so the denominator is never zero; the
      tiny EPSILON only guards against a division by zero in odd cases.
    """
    bce_per_pixel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce_per_pixel * valid).sum() / valid.sum()

    probabilities = torch.sigmoid(logits) * valid
    target = target * valid
    intersection = (probabilities * target).sum()
    dice = 2 * intersection / (probabilities.sum() + target.sum() + EPSILON)

    return bce + (1 - dice)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_seed(seed):
    """Seed every random generator used during training."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit():
    """Commit of the code in use, recorded with every run (reproducibility)."""
    try:
        result = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def run_one_epoch(model, loader, device, optimizer=None, scaler=None, use_amp=False):
    """Train (if an optimizer is given) or evaluate the model for one epoch.

    Returns the mean loss and, for monitoring only, two quick metrics at a
    fixed 0.5 threshold with the "max" score: image MCC and mean pixel MCC on
    forged patches. The real thresholds are chosen later, on val, by
    `evaluation.choose_thresholds`.
    """
    training = optimizer is not None
    model.train(training)

    total_loss, n_batches = 0.0, 0
    scores, labels, pixel_mccs = [], [], []

    for image, target, valid in loader:
        image, target, valid = image.to(device), target.to(device), valid.to(device)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(image)
            loss = loss_function(logits.float(), target, valid)

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        n_batches += 1

        if not training:
            probabilities = torch.sigmoid(logits.float()).cpu().numpy()
            target_np = target.cpu().numpy().astype(bool)
            valid_np = valid.cpu().numpy().astype(bool)
            for k in range(probabilities.shape[0]):
                is_forged = bool(target_np[k].any())
                scores.append(image_score(probabilities[k, 0], "max"))
                labels.append(is_forged)
                if is_forged:
                    c = confusion_counts(probabilities[k, 0] >= 0.5, target_np[k, 0], valid_np[k, 0])
                    pixel_mccs.append(mcc(c["tp"], c["fp"], c["fn"], c["tn"]))

    result = {"loss": total_loss / max(n_batches, 1)}
    if not training:
        c = confusion_counts(np.array(scores) >= 0.5, np.array(labels))
        result["image_mcc_at_0.5"] = mcc(c["tp"], c["fp"], c["fn"], c["tn"])
        result["pixel_mcc_at_0.5"] = float(np.mean(pixel_mccs)) if pixel_mccs else float("nan")
    return result


def train(dataset_dir, run_dir, config=None):
    """Train the U-Net; resume automatically if `run_dir/last.pt` exists.

    Parameters
    ----------
    dataset_dir : str or Path
        Dataset folder (manifest.csv, real/, forged/, masks/), ideally on the
        Colab VM's local disk (faster than Drive).
    run_dir : str or Path
        Run folder, on Google Drive so that it survives a disconnection.
        Receives config.json, log.csv, last.pt and best.pt.
    config : dict, optional
        Overrides of DEFAULT_CONFIG.

    Returns
    -------
    Path
        Path of the best checkpoint (lowest validation loss).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    log_path = run_dir / "log.csv"

    # On resume, the saved configuration wins: same normalization, same model.
    if last_path.exists() and config_path.exists():
        config = json.loads(config_path.read_text())
        print(f"Resuming the run in {run_dir}")
    else:
        full_config = dict(DEFAULT_CONFIG)
        full_config.update(config or {})
        config = full_config

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config["mixed_precision"] and device.type == "cuda"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))

    manifest = read_manifest(Path(dataset_dir) / "manifest.csv")

    if "normalization_mean" not in config:
        mean, std = compute_normalization(dataset_dir, manifest, config["input_scale"], seed=config["seed"])
        config["normalization_mean"] = mean
        config["normalization_std"] = std
        config["git_commit"] = git_commit()
        config["dataset_dir"] = str(dataset_dir)
        # Level of the training forgeries: names the evaluations of the run
        # (eval_level<L>_trained<this level>, see own_eval_dir).
        config["dataset_level"] = training_level(manifest)
        config_path.write_text(json.dumps(config, indent=2))
    print(f"Normalization: mean {config['normalization_mean']:.3f}, std {config['normalization_std']:.3f}")

    datasets = {}
    for split in ["train", "val"]:
        datasets[split] = SARPatchDataset(dataset_dir, manifest, split,
                                          config["normalization_mean"], config["normalization_std"],
                                          config["input_scale"], augment=config["augment"] and split == "train")
        print(f"{split}: {len(datasets[split])} samples")

    loaders = {
        "train": DataLoader(datasets["train"], batch_size=config["batch_size"], shuffle=True,
                            drop_last=True, num_workers=config["num_workers"], pin_memory=device.type == "cuda"),
        "val": DataLoader(datasets["val"], batch_size=config["batch_size"], shuffle=False,
                          num_workers=config["num_workers"], pin_memory=device.type == "cuda"),
    }

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scaler = torch.amp.GradScaler(enabled=use_amp)

    start_epoch, best_val_loss, epochs_without_improvement = 0, float("inf"), 0
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        print(f"Resumed at epoch {start_epoch + 1}")

    for epoch in range(start_epoch, config["epochs"]):
        start = time.time()
        train_result = run_one_epoch(model, loaders["train"], device, optimizer, scaler, use_amp)
        val_result = run_one_epoch(model, loaders["val"], device, use_amp=use_amp)
        scheduler.step()

        log_row = {
            "epoch": epoch + 1,
            "train_loss": round(train_result["loss"], 5),
            "val_loss": round(val_result["loss"], 5),
            # float(): a numpy float would print as "np.float64(...)".
            "val_image_mcc_at_0.5": round(float(val_result["image_mcc_at_0.5"]), 4),
            "val_pixel_mcc_at_0.5": round(float(val_result["pixel_mcc_at_0.5"]), 4),
            "seconds": round(time.time() - start),
        }
        print(log_row)
        write_header = not log_path.exists()
        with log_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_row))
            if write_header:
                writer.writeheader()
            writer.writerow(log_row)

        # Keep the checkpoint with the lowest validation loss.
        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_result}, best_path)
            print(f"  -> new best checkpoint (val loss {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_val_loss": best_val_loss,
                    "epochs_without_improvement": epochs_without_improvement}, last_path)

        # Early stopping: validation has not improved for a while.
        if epochs_without_improvement >= config["early_stopping_patience"]:
            print(f"Early stopping: no improvement for {epochs_without_improvement} epochs")
            break

    print(f"Done. Best val loss {best_val_loss:.4f}, checkpoint {best_path}")
    return best_path


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def training_level(manifest):
    """Level(s) of the forgeries of the training split, e.g. 1, or "1+2" for a mix."""
    forged_train = manifest[(manifest["split"] == "train") & (manifest["label"] == 1)]
    levels = sorted(int(level) for level in forged_train["level"].dropna().unique())
    if len(levels) == 1:
        return levels[0]
    return "+".join(str(level) for level in levels)


def own_eval_dir(run_dir, config):
    """Folder of the evaluation of a run on its own level.

    Evaluations are named `eval_level<dataset level>_trained<training level>`
    inside the run folder; on its own level that is e.g. `eval_level1_trained1`.
    Runs evaluated before this naming used `eval/`, still found here.
    """
    run_dir = Path(run_dir)
    level = config.get("dataset_level", 1)  # runs older than this field: level 1
    new_dir = run_dir / f"eval_level{level}_trained{level}"
    legacy_dir = run_dir / "eval"
    if not new_dir.exists() and legacy_dir.exists():
        return legacy_dir
    return new_dir


def load_model(run_dir, checkpoint="best.pt", device=None):
    """Load a trained model and the configuration of its run.

    If the evaluation of the run on its own level has already chosen its
    settings on the validation set (`own_eval_dir(...)/settings.json`), they
    are merged into the configuration, so that `predict` uses them.
    """
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    settings_path = own_eval_dir(run_dir, config) / "settings.json"
    if settings_path.exists():
        config.update(json.loads(settings_path.read_text()))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model({**config, "pretrained": False})  # weights come from the checkpoint
    state = torch.load(run_dir / checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, config


@torch.no_grad()
def predict_probabilities(model, config, patches):
    """Probability maps for a batch of raw patches (list of 2-D arrays)."""
    device = next(model.parameters()).device
    images = []
    for patch in patches:
        image = (to_model_scale(patch, config["input_scale"]) - config["normalization_mean"]) \
            / config["normalization_std"]
        images.append(image.astype(np.float32))
    batch = torch.from_numpy(np.stack(images)).unsqueeze(1).to(device)  # (B, 1, H, W)
    probabilities = torch.sigmoid(model(batch)).cpu().numpy()
    return probabilities[:, 0]


def predict(model, config, patch_or_path):
    """Run the model on ONE patch (a path to a .tif, or a 2-D array).

    For inspection, and for the hand-made presentation example (#11).
    The network computes one probability map; both answers come from it:
    "is it forged?" (image score vs image threshold), then "where?" (pixels
    above the pixel threshold), only if the patch is called forged.

    The score method and both thresholds are the ones chosen on the
    validation set (`settings.json` of the run's own evaluation, merged by
    `load_model`). Before the
    evaluation has run they do not exist yet: "max" and 0.5 are used, the
    same as the monitoring metrics of training.

    Returns
    -------
    score : float
        Image-level score.
    is_forged : bool
        Decision: score >= image threshold.
    probability_map : numpy.ndarray
        Anomaly map, same size as the patch, values in [0, 1].
    zone : numpy.ndarray of bool
        Pixels called forged (probability >= pixel threshold); all False when
        the patch is called authentic, so that no zone is shown for it.
    """
    if isinstance(patch_or_path, (str, Path)):
        patch = read_tif(patch_or_path)
    else:
        patch = np.asarray(patch_or_path)

    if "image_threshold" not in config:
        print("No settings chosen on val yet (run the evaluation): using 'max' and 0.5")
    score_method = config.get("score_method", "max")
    image_threshold = config.get("image_threshold", 0.5)
    pixel_threshold = config.get("pixel_threshold", 0.5)

    probability_map = predict_probabilities(model, config, [patch])[0]

    # 1. Is it forged?
    score = image_score(probability_map, score_method)
    is_forged = bool(score >= image_threshold)

    # 2. Where? Only for a patch called forged.
    if is_forged:
        zone = probability_map >= pixel_threshold
    else:
        zone = np.zeros(probability_map.shape, dtype=bool)

    return score, is_forged, probability_map, zone


def iterate_predictions(model, config, dataset_dir, split, batch_size=8):
    """Yield (manifest row, probability map, mask or None) for every sample of a split.

    This is what `evaluation.collect_statistics` consumes.
    """
    dataset_dir = Path(dataset_dir)
    manifest = read_manifest(dataset_dir / "manifest.csv")
    rows = manifest[manifest["split"] == split].reset_index(drop=True)

    for start in range(0, len(rows), batch_size):
        batch_rows = rows.iloc[start:start + batch_size]
        patches = [read_tif(dataset_dir / row["file"]) for _, row in batch_rows.iterrows()]
        probability_maps = predict_probabilities(model, config, patches)

        for (_, row), probability_map in zip(batch_rows.iterrows(), probability_maps):
            mask = read_tif(dataset_dir / row["mask_file"]) if row["label"] == 1 else None
            yield row, probability_map, mask
