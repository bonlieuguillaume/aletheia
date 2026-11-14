"""Run a trained model on an image from OUTSIDE the dataset (PNG, JPG, ...).

Part of #11 (presentation example, Vertical 52 / Kronstadt scenario). The
typical input is a screenshot of a published, possibly tampered image: an
optical image, resampled and often compressed. This is an **out-of-domain
test**, shown for illustration only: the model was trained on Sentinel-1 VV
amplitude, so the result has no metric value.

Steps (one function each):
1. `read_image_gray`: colour image -> one float band, mean of the channels;
2. `to_pseudo_sar`: give it the statistics of the training patches, in the
   scale the model expects;
3. `predict_large_image`: overlapping 512x512 windows, one probability map;
4. `analyse_image`: image decision and zone, with the thresholds frozen on
   the validation set of the model's own level;
5. `plot_result`: figure for the notebook and the slides.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.evaluation.evaluation import image_score  # noqa: E402
from features.unet.unet import predict_probabilities  # noqa: E402

TILE_SIZE = 512      # size of the training patches: the thresholds were chosen for it
TILE_STRIDE = 256    # half a window: neighbouring windows overlap by 50 %


def read_image_gray(path):
    """Read an image file and return one float32 band.

    Colour images are turned into grey by the plain MEAN of the R, G, B
    channels, kept as floats (0-255): no rounding and above all no
    binarization, which would destroy the texture the model looks at.
    A transparency channel (alpha) is ignored.
    """
    from PIL import Image  # Pillow: always installed with matplotlib

    with Image.open(path) as image:
        if image.mode in ("L", "I", "I;16", "F"):
            values = np.asarray(image, dtype=np.float32)       # already one band
        else:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            values = rgb.mean(axis=2)                           # (R + G + B) / 3
    return values


def to_pseudo_sar(gray, config):
    """Give a grey image the statistics the model was trained on.

    The model first converts SAR amplitude to dB, then normalizes with the
    mean and standard deviation of the training patches (`config.json`).
    A screenshot has grey levels 0-255 instead: we give its grey levels the
    same mean and standard deviation, in dB, as the training patches, then
    convert back to the scale the model reads (linear if the run reads
    linear values). The model's own pipeline then applies unchanged, and
    the network sees an image with the mean and spread it knows.

    This is only a first-order match (mean and spread, over the whole image,
    never per window): the speckle statistics of SAR are not reproduced.

    Parameters
    ----------
    gray : numpy.ndarray
        2-D grey image, any range.
    config : dict
        Configuration of the run (`normalization_mean`, `normalization_std`,
        `input_scale`).
    """
    gray = gray.astype(np.float64)
    spread = gray.std()
    if spread == 0:
        raise ValueError("The image is uniform: nothing to analyse")

    standardized = (gray - gray.mean()) / spread                       # mean 0, std 1
    values_db = standardized * config["normalization_std"] + config["normalization_mean"]

    if config["input_scale"] == "db":
        return values_db.astype(np.float32)
    # Linear: the model will apply 10 * log10 and get values_db back.
    return (10.0 ** (values_db / 10.0)).astype(np.float32)


def window_starts(length, tile=TILE_SIZE, stride=TILE_STRIDE):
    """First pixel of each window along one axis; the last window ends at the edge.

    Example: length 1000, tile 512, stride 256 -> [0, 256, 488].
    """
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)   # cover the last pixels too
    return starts


def predict_large_image(model, config, raw_image, tile=TILE_SIZE, stride=TILE_STRIDE, batch_size=8):
    """Probability map of an image of any size, window by window.

    Windows of the training size overlap by half. Each pixel keeps the
    HIGHEST probability among the windows that contain it: copy-move is
    only visible to a window that holds both the copy and its source, so a
    pixel is flagged as soon as one window sees the duplication.

    An image smaller than a window is padded by reflection (only then), and
    the map is cropped back to the image size.

    Returns
    -------
    probability_map : numpy.ndarray
        Same size as `raw_image`, values in [0, 1].
    windows : list of dict
        One per window: `row`, `col` (top-left corner) and `score` (image
        score of that window, with the run's score method).
    """
    height, width = raw_image.shape
    pad_rows = max(0, tile - height)
    pad_cols = max(0, tile - width)
    image = raw_image
    if pad_rows > 0 or pad_cols > 0:
        image = np.pad(raw_image, ((0, pad_rows), (0, pad_cols)), mode="reflect")

    corners = []
    for row in window_starts(image.shape[0], tile, stride):
        for col in window_starts(image.shape[1], tile, stride):
            corners.append((row, col))

    score_method = config.get("score_method", "max")
    probability_map = np.zeros(image.shape, dtype=np.float32)
    windows = []
    for first in range(0, len(corners), batch_size):
        batch_corners = corners[first:first + batch_size]
        patches = [image[r:r + tile, c:c + tile] for r, c in batch_corners]
        maps = predict_probabilities(model, config, patches)
        for (row, col), window_map in zip(batch_corners, maps):
            region = probability_map[row:row + tile, col:col + tile]
            np.maximum(region, window_map, out=region)          # keep the highest
            windows.append({"row": row, "col": col, "score": image_score(window_map, score_method)})

    return probability_map[:height, :width], windows


def analyse_image(model, config, path, tile=TILE_SIZE, stride=TILE_STRIDE):
    """Whole pipeline on one image file: decision, map, zone.

    The image decision uses the score method and the image threshold frozen
    on the validation set (merged into `config` by `unet.load_model`). They
    were chosen for 512x512 patches, so the score is computed per window
    and the image is called forged if at least ONE window reaches the
    threshold; the image score is the highest window score.

    Returns
    -------
    dict
        `gray` (input as read), `probability_map`, `zone` (pixels >= pixel
        threshold; all False if the image is called authentic, like
        `unet.predict`), `score`, `is_forged`, `best_window` (corner and
        score of the highest window), `windows`, and the settings used.
    """
    if "image_threshold" not in config:
        raise ValueError("No settings in the configuration: evaluate the run on its own level first")

    gray = read_image_gray(path)
    raw_image = to_pseudo_sar(gray, config)
    probability_map, windows = predict_large_image(model, config, raw_image, tile, stride)

    best_window = max(windows, key=lambda window: window["score"])
    score = best_window["score"]
    is_forged = bool(score >= config["image_threshold"])
    if is_forged:
        zone = probability_map >= config["pixel_threshold"]
    else:
        zone = np.zeros(probability_map.shape, dtype=bool)

    return {
        "gray": gray,
        "probability_map": probability_map,
        "zone": zone,
        "score": score,
        "is_forged": is_forged,
        "best_window": best_window,
        "windows": windows,
        "score_method": config["score_method"],
        "image_threshold": config["image_threshold"],
        "pixel_threshold": config["pixel_threshold"],
        "tile": tile,
    }


def plot_result(result, title=""):
    """Input | probability map over the input | binary zone. Returns the figure."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    gray = result["gray"]
    probability_map = result["probability_map"]
    # Display: grey levels stretched between the 2nd and 98th percentiles.
    low, high = np.percentile(gray, [2, 98])
    shown = np.clip((gray - low) / max(high - low, 1e-6), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(shown, cmap="gray")
    axes[0].set_title("Input (grey = mean of R, G, B)")

    axes[1].imshow(shown, cmap="gray")
    overlay = axes[1].imshow(np.ma.masked_less(probability_map, 0.05), cmap="inferno",
                             vmin=0, vmax=1, alpha=0.6)
    # The window that gave the image score.
    best = result["best_window"]
    tile = result["tile"]
    axes[1].add_patch(Rectangle((best["col"], best["row"]), tile, tile,
                                fill=False, edgecolor="lime", linewidth=1.5, linestyle="--"))
    axes[1].set_title("Probability map (green: window with the highest score)")
    fig.colorbar(overlay, ax=axes[1], fraction=0.046)

    # The binary map is drawn at the pixel threshold even for an image called
    # authentic, for information; the title says when no zone is reported.
    binary = probability_map >= result["pixel_threshold"]
    axes[2].imshow(binary, cmap="gray", vmin=0, vmax=1)
    zone_title = f"Pixels >= {result['pixel_threshold']:.2f} (white)"
    if not result["is_forged"]:
        zone_title += "\nfor information: image called authentic, no zone reported"
    axes[2].set_title(zone_title)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
