"""Metrics of aletheia: image score and anomaly map (issues #6, #7, #8, #32).

This module knows nothing about the model: it receives probability maps
(one value in [0, 1] per pixel) and the ground-truth masks, and turns them
into metrics. The full procedure is described in docs/architecture_v0.1.md.

Main metric: MCC; second metric: F1; ROC AUC as a threshold-free complement.

Rules (docs/report.md, section 5):
- thresholds are chosen on the validation set, then frozen for the test;
- pixel metrics are computed on forged patches only, against `mask == 1`;
- source pixels (`mask == 2`) are ignored: neither hit nor false alarm;
- false alarms on authentic patches are measured separately;
- every pixel metric is also computed without a border margin (#32).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.sar_io.sar_io import MASK_PASTED, MASK_SOURCE  # noqa: E402

# Candidate thresholds for the pixel maps: 0.05, 0.10, ..., 0.95.
PIXEL_THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)

# Border margin excluded in the "margin" variant of the pixel metrics (#32).
BORDER_MARGIN = 16

# Ways of summarizing a map into one image score (#7), compared on val.
SCORE_METHODS = ["max", "top0.5%", "top1%", "top2%", "p99.5"]


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

def confusion_counts(predicted, truth, valid=None):
    """Count true / false positives / negatives between two boolean arrays.

    Parameters
    ----------
    predicted, truth : numpy.ndarray of bool
        Same shape. True = "forged" (pixel or image).
    valid : numpy.ndarray of bool, optional
        Elements to take into account; the others are ignored entirely.

    Returns
    -------
    dict
        tp, fp, fn, tn as Python ints.
    """
    if valid is None:
        valid = np.ones(predicted.shape, dtype=bool)

    tp = np.sum(predicted & truth & valid)
    fp = np.sum(predicted & ~truth & valid)
    fn = np.sum(~predicted & truth & valid)
    tn = np.sum(~predicted & ~truth & valid)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def mcc(tp, fp, fn, tn):
    """Matthews correlation coefficient.

        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    1 = perfect, 0 = no better than chance, -1 = always wrong. Uses the four
    cells of the confusion matrix, so it stays meaningful when one class is
    rare (e.g. 3 % of forged pixels). Returns 0 when it is undefined (one of
    the four sums is zero), the usual convention.
    """
    # Python floats: products of pixel counts can exceed the int64 range.
    tp, fp, fn, tn = float(tp), float(fp), float(fn), float(tn)
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator == 0:
        return 0.0
    return (tp * tn - fp * fn) / np.sqrt(denominator)


def f1(tp, fp, fn):
    """F1 score: 2*TP / (2*TP + FP + FN). Equal to the Dice coefficient.

    Returns NaN when there is nothing to find and nothing was found
    (no positive at all), where F1 is undefined.
    """
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return float("nan")
    return 2 * tp / denominator


# ---------------------------------------------------------------------------
# From a map to an image score (#7)
# ---------------------------------------------------------------------------

def image_score(probability_map, method="max"):
    """Summarize a probability map into one image-level score.

    Methods (compared on the validation set):
    - "max": highest pixel; simple, but one noisy pixel is enough;
    - "topX%": mean of the X % highest pixels; more robust, but X must stay
      below the smallest paste, or background pixels dilute the score;
    - "p99.5": 99.5th percentile of the map.
    """
    values = probability_map.ravel()

    if method == "max":
        return float(values.max())

    if method.startswith("top") and method.endswith("%"):
        fraction = float(method[3:-1]) / 100.0
        n_top = max(1, int(round(fraction * values.size)))
        # np.partition puts the n_top largest values at the end, unsorted:
        # cheaper than sorting the 262,144 pixels.
        top_values = np.partition(values, values.size - n_top)[values.size - n_top:]
        return float(top_values.mean())

    if method.startswith("p"):
        return float(np.percentile(values, float(method[1:])))

    raise ValueError(f"Unknown score method: {method}")


# ---------------------------------------------------------------------------
# Per-sample statistics
# ---------------------------------------------------------------------------

def border_region(shape, margin):
    """Boolean array, True everywhere except in a band of `margin` px along the border."""
    inside = np.zeros(shape, dtype=bool)
    inside[margin:shape[0] - margin, margin:shape[1] - margin] = True
    return inside


def sample_statistics(probability_map, mask, label, margin=BORDER_MARGIN):
    """Compute everything the metrics need about one sample.

    Thresholds are not known yet at this stage (they are chosen on the
    validation set), so the pixel counts are stored for every candidate
    threshold in `PIXEL_THRESHOLDS`.

    Parameters
    ----------
    probability_map : numpy.ndarray
        Model output, values in [0, 1].
    mask : numpy.ndarray or None
        Ground-truth mask (0 / 1 / 2), None for an authentic patch.
    label : int
        0 authentic, 1 forged.
    margin : int
        Border band excluded in the "margin" variant (#32).

    Returns
    -------
    dict
        - "score_<method>": image score for every method of SCORE_METHODS;
        - forged patch: "pixel_counts" and "pixel_counts_margin", arrays of
          shape (n_thresholds, 4) holding tp, fp, fn, tn;
        - authentic patch: "false_alarm" and "false_alarm_margin", arrays of
          shape (n_thresholds,) holding the fraction of flagged pixels.
    """
    stats = {}
    for method in SCORE_METHODS:
        stats[f"score_{method}"] = image_score(probability_map, method)

    inside = border_region(probability_map.shape, margin)

    if label == 1:
        truth = mask == MASK_PASTED
        valid = mask != MASK_SOURCE  # source pixels are ignored
        counts = np.zeros((len(PIXEL_THRESHOLDS), 4), dtype=np.int64)
        counts_margin = np.zeros((len(PIXEL_THRESHOLDS), 4), dtype=np.int64)
        for i, threshold in enumerate(PIXEL_THRESHOLDS):
            predicted = probability_map >= threshold
            c = confusion_counts(predicted, truth, valid)
            counts[i] = [c["tp"], c["fp"], c["fn"], c["tn"]]
            c = confusion_counts(predicted, truth, valid & inside)
            counts_margin[i] = [c["tp"], c["fp"], c["fn"], c["tn"]]
        stats["pixel_counts"] = counts
        stats["pixel_counts_margin"] = counts_margin
    else:
        false_alarm = np.zeros(len(PIXEL_THRESHOLDS))
        false_alarm_margin = np.zeros(len(PIXEL_THRESHOLDS))
        for i, threshold in enumerate(PIXEL_THRESHOLDS):
            flagged = probability_map >= threshold
            false_alarm[i] = flagged.mean()
            false_alarm_margin[i] = flagged[inside].mean()
        stats["false_alarm"] = false_alarm
        stats["false_alarm_margin"] = false_alarm_margin

    return stats


def collect_statistics(predictions, margin=BORDER_MARGIN, save_maps_dir=None):
    """Run `sample_statistics` over an iterable of predictions.

    Parameters
    ----------
    predictions : iterable of (row, probability_map, mask)
        `row` is the manifest row (pandas Series) of the sample; `mask` is
        None for authentic samples. `unet.iterate_predictions` yields this.
    margin : int
        See `sample_statistics`.
    save_maps_dir : str or Path, optional
        If given, every probability map is also saved there as
        `<sample_id>.tif` (uint8). Done for the test set only.

    Returns
    -------
    list of dict
        One dict per sample: `sample_statistics` + sample_id and label.
    """
    from features.sar_io.sar_io import write_probability_map

    all_stats = []
    for row, probability_map, mask in predictions:
        stats = sample_statistics(probability_map, mask, int(row["label"]), margin)
        stats["sample_id"] = row["sample_id"]
        stats["label"] = int(row["label"])
        all_stats.append(stats)

        if save_maps_dir is not None:
            write_probability_map(Path(save_maps_dir) / f"{row['sample_id']}.tif", probability_map)

        if len(all_stats) % 100 == 0:
            print(f"  {len(all_stats)} samples done")
    return all_stats


# ---------------------------------------------------------------------------
# Thresholds, chosen on the validation set
# ---------------------------------------------------------------------------

def best_image_threshold(scores, labels):
    """Return the image threshold that maximizes the MCC, and that MCC.

    Every distinct score is tried as a threshold (score >= threshold means
    "forged").
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(bool)

    best_threshold, best_mcc = 0.5, -1.0
    for threshold in np.unique(scores):
        c = confusion_counts(scores >= threshold, labels)
        value = mcc(c["tp"], c["fp"], c["fn"], c["tn"])
        if value > best_mcc:
            best_threshold, best_mcc = float(threshold), value
    return best_threshold, best_mcc


def mean_pixel_metrics(all_stats, threshold_index, key="pixel_counts"):
    """Average the per-patch pixel MCC and F1 over the forged samples."""
    mcc_values, f1_values = [], []
    for stats in all_stats:
        if stats["label"] != 1:
            continue
        tp, fp, fn, tn = stats[key][threshold_index]
        mcc_values.append(mcc(tp, fp, fn, tn))
        f1_values.append(f1(tp, fp, fn))
    return float(np.mean(mcc_values)), float(np.nanmean(f1_values))


def choose_thresholds(val_stats):
    """Choose every setting on the validation set (step 3 of the procedure).

    1. For each score method, the image threshold maximizing the image MCC;
       the method with the best MCC is kept.
    2. The pixel threshold maximizing the mean per-patch pixel MCC.

    Returns
    -------
    dict
        score_method, image_threshold, pixel_threshold, and the validation
        MCCs reached, to be frozen and applied to the test set.
    """
    labels = [s["label"] for s in val_stats]

    # 1. Image level: best (method, threshold) pair.
    best = {"image_mcc_val": -1.0}
    for method in SCORE_METHODS:
        scores = [s[f"score_{method}"] for s in val_stats]
        threshold, value = best_image_threshold(scores, labels)
        print(f"  val image MCC with {method:8s}: {value:.3f} (threshold {threshold:.3f})")
        if value > best["image_mcc_val"]:
            best = {"score_method": method, "image_threshold": threshold, "image_mcc_val": value}

    # 2. Pixel level: best threshold of the grid.
    best_index, best_pixel_mcc = 0, -1.0
    for i in range(len(PIXEL_THRESHOLDS)):
        value, _ = mean_pixel_metrics(val_stats, i)
        if value > best_pixel_mcc:
            best_index, best_pixel_mcc = i, value
    best["pixel_threshold"] = float(PIXEL_THRESHOLDS[best_index])
    best["pixel_mcc_val"] = best_pixel_mcc

    print(f"Chosen on val: score {best['score_method']}, image threshold "
          f"{best['image_threshold']:.3f}, pixel threshold {best['pixel_threshold']:.2f}")
    return best


# ---------------------------------------------------------------------------
# Final metrics, with the frozen thresholds
# ---------------------------------------------------------------------------

def compute_metrics(all_stats, settings, manifest=None):
    """Apply the frozen settings and compute the final metrics (test set).

    Parameters
    ----------
    all_stats : list of dict
        Output of `collect_statistics` on the test set.
    settings : dict
        Output of `choose_thresholds` on the validation set.
    manifest : pandas.DataFrame, optional
        If given, the per-sample table is joined with it, so the results can
        be sliced by level, paste size, scene...

    Returns
    -------
    summary : dict
        Image MCC / F1 / AUC, pixel MCC / F1 (mean per patch and global),
        the same without the border margin, false-alarm rate on authentic
        patches.
    table : pandas.DataFrame
        One row per sample: score, decision, pixel metrics or false alarm.
    """
    from sklearn.metrics import roc_auc_score

    method = settings["score_method"]
    pixel_index = int(np.argmin(np.abs(PIXEL_THRESHOLDS - settings["pixel_threshold"])))

    # Per-sample table.
    rows = []
    for stats in all_stats:
        score = stats[f"score_{method}"]
        row = {
            "sample_id": stats["sample_id"],
            "label": stats["label"],
            "score": score,
            "predicted_label": int(score >= settings["image_threshold"]),
        }
        if stats["label"] == 1:
            tp, fp, fn, tn = stats["pixel_counts"][pixel_index]
            row["pixel_mcc"] = mcc(tp, fp, fn, tn)
            row["pixel_f1"] = f1(tp, fp, fn)
            tp, fp, fn, tn = stats["pixel_counts_margin"][pixel_index]
            row["pixel_mcc_margin"] = mcc(tp, fp, fn, tn)
        else:
            row["false_alarm"] = stats["false_alarm"][pixel_index]
            row["false_alarm_margin"] = stats["false_alarm_margin"][pixel_index]
        rows.append(row)
    table = pd.DataFrame(rows)

    # Image level: one decision per patch.
    labels = table["label"].to_numpy().astype(bool)
    decisions = table["predicted_label"].to_numpy().astype(bool)
    c = confusion_counts(decisions, labels)
    summary = {
        "image_mcc": mcc(c["tp"], c["fp"], c["fn"], c["tn"]),
        "image_f1": f1(c["tp"], c["fp"], c["fn"]),
        "image_auc": float(roc_auc_score(labels, table["score"])) if len(set(labels)) > 1 else float("nan"),
        "image_counts": c,
    }

    # Pixel level, forged patches only: mean over patches...
    summary["pixel_mcc_mean"], summary["pixel_f1_mean"] = mean_pixel_metrics(all_stats, pixel_index)
    summary["pixel_mcc_mean_margin"], _ = mean_pixel_metrics(all_stats, pixel_index, "pixel_counts_margin")

    # ... and global, i.e. all the pixels of all forged patches pooled.
    pooled = np.zeros(4, dtype=np.int64)
    for stats in all_stats:
        if stats["label"] == 1:
            pooled += stats["pixel_counts"][pixel_index]
    tp, fp, fn, tn = pooled
    summary["pixel_mcc_global"] = mcc(tp, fp, fn, tn)
    summary["pixel_f1_global"] = f1(tp, fp, fn)

    # Authentic patches: fraction of pixels flagged by mistake.
    authentic = table[table["label"] == 0]
    summary["false_alarm_mean"] = float(authentic["false_alarm"].mean())
    summary["false_alarm_mean_margin"] = float(authentic["false_alarm_margin"].mean())

    if manifest is not None:
        table = table.merge(manifest, on=["sample_id", "label"], how="left")

    return summary, table


def print_summary(summary):
    """Print the summary in a readable way (ASCII only, see CLAUDE.md)."""
    c = summary["image_counts"]
    print("Image level (is the patch forged?)")
    print(f"  MCC {summary['image_mcc']:.3f} | F1 {summary['image_f1']:.3f} | AUC {summary['image_auc']:.3f}")
    print(f"  TP {c['tp']}  FN {c['fn']}  FP {c['fp']}  TN {c['tn']}")
    print("Pixel level (forged patches, source pixels ignored)")
    print(f"  mean per patch: MCC {summary['pixel_mcc_mean']:.3f} | F1 {summary['pixel_f1_mean']:.3f}")
    print(f"  global:         MCC {summary['pixel_mcc_global']:.3f} | F1 {summary['pixel_f1_global']:.3f}")
    print(f"  without the {BORDER_MARGIN}-px border: MCC {summary['pixel_mcc_mean_margin']:.3f}")
    print("Authentic patches")
    print(f"  flagged pixels: {100 * summary['false_alarm_mean']:.2f} % "
          f"({100 * summary['false_alarm_mean_margin']:.2f} % without the border)")


def metrics_by_group(table, column):
    """Mean results per group, e.g. per `level`, `scene` or paste size bin."""
    forged = table[table["label"] == 1]
    return forged.groupby(column).agg(
        n=("sample_id", "count"),
        detected=("predicted_label", "mean"),
        pixel_mcc=("pixel_mcc", "mean"),
        pixel_f1=("pixel_f1", "mean"),
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def to_display(patch, input_scale="linear"):
    """Turn a SAR patch into an image that is pleasant to look at.

    Linear values are converted to dB, then clipped between the 2nd and 98th
    percentiles: SAR has a huge dynamic range, a few bright targets would
    otherwise make the rest of the patch black.
    """
    values = np.nan_to_num(patch.astype(np.float32))
    if input_scale == "linear":
        values = 10 * np.log10(np.maximum(values, 1e-6))
    low, high = np.percentile(values, [2, 98])
    return np.clip((values - low) / max(high - low, 1e-6), 0, 1)


def plot_sample(patch, mask, probability_map, title="", input_scale="linear", pixel_threshold=0.5):
    """Show one sample: patch | true mask | predicted map over the patch.

    Returns the matplotlib figure, so a notebook can save it for the slides.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(to_display(patch, input_scale), cmap="gray")
    axes[0].set_title("Patch")

    if mask is None:
        mask = np.zeros(patch.shape, dtype=np.uint8)
    # 0 background (black), 1 pasted (white), 2 source (grey).
    shown_mask = np.zeros(mask.shape)
    shown_mask[mask == MASK_PASTED] = 1.0
    shown_mask[mask == MASK_SOURCE] = 0.5
    axes[1].imshow(shown_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth (white pasted, grey source)")

    axes[2].imshow(to_display(patch, input_scale), cmap="gray")
    overlay = axes[2].imshow(np.ma.masked_less(probability_map, 0.05), cmap="inferno", vmin=0, vmax=1, alpha=0.6)
    axes[2].contour(probability_map >= pixel_threshold, levels=[0.5], colors="cyan", linewidths=1)
    axes[2].set_title(f"Prediction (cyan: >= {pixel_threshold:.2f})")
    fig.colorbar(overlay, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
