"""Optical control (#47): Sentinel-2 scenes turned into pseudo-SAR scenes.

Question: does the copy-move model rely on SAR (speckle, radar texture), or
only on duplication? The same forgeries are made on optical images and the
SAR model is evaluated on them as it is (docs/report.md, section 5.3).

This module turns one Sentinel-2 L2A product (`.SAFE`) into one single-band
"pseudo-SAR" scene that `tiling`, `copy_move` and `evaluation` read with no
code change:

1. grey = float mean of the 10 m bands B02, B03, B04 (`s2_gray`);
2. optionally, synthetic speckle: grey x gamma draw, ENL 4.4 (`add_speckle`),
   the control that tells "needs speckle" from "different-looking image";
3. grey -> SAR values in dB (`to_sar_db`): histogram matching (default) or
   affine (mean / std only) onto the dB values of the SAR training patches
   (`sar_reference`);
4. dB -> linear (10^(dB/10)), as the SAR scenes are stored; no-data stays 0.

Inside the model, the dB conversion and the normalization then apply as for
SAR: the network sees the optical image on the same scale as SAR.
"""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.unet.unet import to_model_scale  # noqa: E402

BANDS = ("B02", "B03", "B04")      # blue, green, red: the 10 m visible bands
ENL = 4.4                          # equivalent number of looks of Sentinel-1 IW GRD
N_QUANTILES = 1001                 # resolution of the SAR reference distribution
METHODS = ("histogram", "affine")


# ---------------------------------------------------------------------------
# Reading Sentinel-2
# ---------------------------------------------------------------------------

def find_s2_bands(safe_dir):
    """Paths of the 10 m B02 / B03 / B04 images of a Sentinel-2 L2A product.

    They are searched at any depth (`GRANULE/<granule>/IMG_DATA/R10m/
    *_B02_10m.jp2` in a `.SAFE`), so the product can be copied as it is.
    """
    safe_dir = Path(safe_dir)
    paths = {}
    for band in BANDS:
        found = sorted(safe_dir.rglob(f"*_{band}_10m.jp2"))
        if len(found) != 1:
            raise ValueError(f"{safe_dir}: expected one *_{band}_10m.jp2, found {len(found)}")
        paths[band] = found[0]
    return paths


def read_jp2(path):
    """Read one JPEG 2000 band (uint16 digital numbers) with Pillow.

    Pillow's wheels ship the OpenJPEG decoder: no extra dependency. A
    Sentinel-2 band is 10980 x 10980 px, above Pillow's guard against
    "decompression bombs", which is lifted for this file only.
    """
    from PIL import Image

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as image:
            values = np.asarray(image)
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    return values


def s2_gray(safe_dir):
    """Grey image of a Sentinel-2 product and its valid-pixel map.

    Grey = float mean of B02, B03, B04, no rounding. A pixel is valid if all
    three bands are > 0: 0 is the no-data value of Sentinel-2 (outside the
    swath, e.g. a tile cut by the orbit). The L2A offset (+1000 on the
    digital numbers since processing baseline 04.00) is not removed: it
    shifts every pixel by the same amount, which neither the histogram
    matching nor the affine map sees.

    Returns
    -------
    gray : numpy.ndarray of float32
        0 where not valid.
    valid : numpy.ndarray of bool
    """
    paths = find_s2_bands(safe_dir)
    total = None
    valid = None
    for band in BANDS:
        values = read_jp2(paths[band]).astype(np.float32)
        band_valid = values > 0
        if total is None:
            total = values
            valid = band_valid
        else:
            total += values
            valid &= band_valid
    gray = total / len(BANDS)
    gray[~valid] = 0.0
    return gray, valid


# ---------------------------------------------------------------------------
# SAR reference
# ---------------------------------------------------------------------------

def sar_reference(dataset_zip, input_scale="linear", max_patches=200, seed=0):
    """Distribution of the SAR training patches, in dB, as the model sees them.

    Read straight from a SAR dataset zip (e.g. `dataset_v0.1_level3.zip`,
    the dataset of the reference model), with no extraction: authentic
    TRAINING patches only, as for the normalization of the model, and only
    their valid pixels (no-data zeros would add a spike at -60 dB).

    Returns
    -------
    dict
        `quantiles` (N_QUANTILES values from the minimum to the maximum),
        `mean` and `std` (in dB), `n_patches`, `source`.
    """
    import tifffile

    with zipfile.ZipFile(dataset_zip) as archive:
        manifest = pd.read_csv(io.BytesIO(archive.read("manifest.csv")))
        train_real = manifest[(manifest["split"] == "train") & (manifest["label"] == 0)]
        rng = np.random.default_rng(seed)
        n = min(max_patches, len(train_real))
        chosen = sorted(rng.choice(len(train_real), size=n, replace=False))

        all_values = []
        for index in chosen:
            file = train_real.iloc[index]["file"]
            patch = tifffile.imread(io.BytesIO(archive.read(file)))
            patch = np.squeeze(patch)
            valid = (patch > 0) & np.isfinite(patch)
            all_values.append(to_model_scale(patch[valid], input_scale))

    values = np.concatenate(all_values).astype(np.float64)
    levels = np.linspace(0.0, 1.0, N_QUANTILES)
    return {
        "quantiles": np.quantile(values, levels).tolist(),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "n_patches": int(n),
        "source": Path(dataset_zip).name,
    }


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def add_speckle(gray, valid, rng, enl=ENL):
    """Multiply the valid pixels by fully developed speckle (control of #47).

    SAR intensity = reflectivity x speckle, speckle ~ Gamma(shape=ENL,
    scale=1/ENL): mean 1, one independent draw per pixel. ENL 4.4 is the
    one of Sentinel-1 IW GRD (multilooked by ESA). The grey level plays the
    part of the reflectivity.
    """
    speckled = gray.copy()
    speckle = rng.gamma(shape=enl, scale=1.0 / enl, size=int(valid.sum())).astype(np.float32)
    speckled[valid] = gray[valid] * speckle
    return speckled


def match_histogram(values, reference_quantiles):
    """Map values onto a reference distribution, quantile by quantile.

    The value at quantile q of `values` takes the value at quantile q of the
    reference. Increasing map: the order of the pixels, hence every
    structure of the image, is kept; only the histogram changes. Equal
    values (frequent: digital numbers are integers) get the same output.
    """
    unique_values, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    # Quantile of each distinct value: middle of the share of pixels it holds.
    cumulative = np.cumsum(counts)
    quantile_of_unique = (cumulative - counts / 2.0) / cumulative[-1]

    levels = np.linspace(0.0, 1.0, len(reference_quantiles))
    matched_unique = np.interp(quantile_of_unique, levels, reference_quantiles)
    return matched_unique[inverse].astype(np.float32)


def to_sar_db(gray, valid, reference, method="histogram"):
    """Give the valid grey pixels the distribution of SAR, in dB.

    - "histogram": the whole distribution of the SAR training patches
      (`match_histogram`);
    - "affine": its mean and standard deviation only,
      (gray - mean) / std * std_SAR + mean_SAR; the histogram keeps its
      optical shape (docs/report.md, section 5.3).

    The statistics of the grey image are taken over the whole scene (valid
    pixels only), never per patch. Returns 0 where not valid.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method}")

    values = gray[valid].astype(np.float64)
    if method == "histogram":
        matched = match_histogram(values, np.asarray(reference["quantiles"]))
    else:
        spread = values.std()
        if spread == 0:
            raise ValueError("The image is uniform: nothing to match")
        standardized = (values - values.mean()) / spread
        matched = standardized * reference["std"] + reference["mean"]

    values_db = np.zeros(gray.shape, dtype=np.float32)
    values_db[valid] = matched
    return values_db


def to_linear(values_db, valid):
    """dB -> linear values, as the SAR scenes are stored; no-data stays 0."""
    linear = np.zeros(values_db.shape, dtype=np.float32)
    linear[valid] = 10.0 ** (values_db[valid] / 10.0)
    return linear


def pseudo_sar_scene(gray, valid, reference, method="histogram", speckle=False, seed=0):
    """Whole conversion of one grey scene into a pseudo-SAR scene (linear).

    Speckle, if any, is added BEFORE the matching, so that the final
    histogram is the SAR one whatever the variant; being increasing, the
    matching keeps the pixel-to-pixel texture of the speckle.
    """
    if speckle:
        rng = np.random.default_rng(seed)
        gray = add_speckle(gray, valid, rng)
    values_db = to_sar_db(gray, valid, reference, method)
    return to_linear(values_db, valid)


# ---------------------------------------------------------------------------
# Whole pipeline: scenes -> patches -> forged datasets, for each variant
# ---------------------------------------------------------------------------

# Data source suffix -> synthetic speckle or not. The suffix ends the names
# of the datasets and of the evaluation folders (eval_level3_trained3_optical).
VARIANTS = {"_optical": False, "_optical_speckle": True}


def build_optical_datasets(safe_dirs, reference, work_dir, levels=(1, 2, 3), variants=None,
                           method="histogram", patch_size=512, patches_per_scene=200, seed=0):
    """Make the forged optical datasets of every variant and level.

    Every scene goes to the TEST split: the model is only evaluated, with
    the settings frozen on its SAR val, so no train / val is needed. The
    scenes are named after the SAR test scenes they cover (ds_forest_1,
    ds_forest_9), so that the per-scene results line up with SAR. No
    east-west mirror: it only balances radar look directions, which optical
    does not have. Forgeries: same generators and seeds as SAR (copy_move).

    Parameters
    ----------
    safe_dirs : dict
        Scene name -> folder of its Sentinel-2 product (`.SAFE`).
    reference : dict
        Output of `sar_reference`.
    work_dir : str or Path
        Local folder (VM disk) for scenes, patches and datasets.

    Returns
    -------
    dict
        (variant, level) -> dataset folder, ready to be zipped.
    """
    from features.copy_move.copy_move import build_dataset
    from features.sar_io.sar_io import write_patch
    from features.tiling.tiling import tile_scenes
    import shutil

    if variants is None:
        variants = VARIANTS
    work_dir = Path(work_dir)
    names = sorted(safe_dirs)

    # 1. Scenes: each Sentinel-2 product is read once, converted for every variant.
    for scene_index, name in enumerate(names):
        print(f"Reading {name}: {Path(safe_dirs[name]).name}")
        gray, valid = s2_gray(safe_dirs[name])
        print(f"  {gray.shape[1]} x {gray.shape[0]} px, {100 * valid.mean():.1f} % valid")
        for variant, speckle in variants.items():
            scene = pseudo_sar_scene(gray, valid, reference, method, speckle, seed=[seed, scene_index])
            write_patch(work_dir / f"scenes{variant}" / name / f"{name}.tif", scene)
        del gray, valid

    # 2. Patches, then 3. the forged twins, level by level.
    datasets = {}
    for variant in variants:
        patches_dir = work_dir / f"patches{variant}"
        tile_scenes(work_dir / f"scenes{variant}", patches_dir, patch_size=patch_size,
                    patches_per_scene=patches_per_scene,
                    split_counts={"val": 0, "test": len(names)},
                    forced_splits={name: "test" for name in names},
                    orbits={name: "desc" for name in names},   # unused: no mirror
                    balance_orbits=False, seed=seed)
        for level in levels:
            dataset_dir = work_dir / f"dataset_level{level}{variant}"
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            shutil.copytree(patches_dir, dataset_dir)
            build_dataset(dataset_dir, level=level, seed=seed)
            datasets[(variant, level)] = dataset_dir
    return datasets
