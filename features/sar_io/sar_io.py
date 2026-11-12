"""Reading and writing the files shared by every feature of aletheia.

Everything that touches the disk goes through this module, so that the file
formats are defined in one place:

- scene GeoTIFFs produced by the user's preprocessing (rosarium / SNAP);
- patch `.tif` files (float32, one band, no georeferencing);
- mask `.tif` files (uint8, one band, values 0 / 1 / 2, lossless);
- the manifest CSV, the single index of the dataset.

`tifffile` is imported inside the functions, so that the parts of the code
that do not read images (metrics, forgery generation) can be imported and
tested without it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Mask values (see issue #5 and docs/report.md, section 3.1).
MASK_BACKGROUND = 0  # untouched pixel
MASK_PASTED = 1      # pixel of the pasted (target) region: the forgery
MASK_SOURCE = 2      # pixel of the source region the paste was copied from


def read_tif(path):
    """Read a single-band TIFF (scene, patch or mask) into a numpy array.

    Parameters
    ----------
    path : str or Path
        Path of the `.tif` file.

    Returns
    -------
    numpy.ndarray
        2-D array, with the dtype stored in the file.
    """
    import tifffile

    image = tifffile.imread(str(path))

    # A GeoTIFF exported by SNAP can carry a leading band axis of size 1.
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"{path}: expected a single-band image, got shape {image.shape}")
    return image


def write_patch(path, patch):
    """Write a SAR patch as a float32 single-band TIFF, without georeferencing.

    No compression: speckle is noise-like and compresses poorly, so
    compression would only slow down reading during training.
    """
    import tifffile

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), patch.astype(np.float32))


def write_mask(path, mask):
    """Write a mask as a uint8 single-band TIFF, losslessly compressed.

    Masks are mostly zeros, so zlib compression makes them a few kB.
    Never use JPEG for masks: lossy compression would move the edges.
    """
    import tifffile

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), mask.astype(np.uint8), compression="zlib")


def write_probability_map(path, probability_map):
    """Write a predicted anomaly map (values in [0, 1]) as a uint8 TIFF.

    The probabilities are quantized to 0-255: 4 times smaller than float32,
    and a precision of 1/255 is far finer than any threshold we will use.
    """
    quantized = np.round(np.clip(probability_map, 0.0, 1.0) * 255).astype(np.uint8)
    write_mask(path, quantized)  # same format as a mask: uint8, lossless


def read_probability_map(path):
    """Read a map written by `write_probability_map`, back to [0, 1] floats."""
    return read_tif(path).astype(np.float32) / 255.0


def zip_folder(folder, zip_path):
    """Pack a folder into one zip file (e.g. the dataset, to put on Drive).

    Google Drive is slow with thousands of small files and fast with one big
    file: datasets travel between Drive and the Colab VM as one zip.
    Patches are stored without compression (speckle barely compresses).
    """
    import zipfile

    folder = Path(folder)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in folder.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for file in files:
            archive.write(file, arcname=file.relative_to(folder).as_posix())
    print(f"Zipped {len(files)} files into {zip_path}")


def unzip_folder(zip_path, folder):
    """Extract a zip made by `zip_folder` into a folder (e.g. the VM's disk)."""
    import zipfile

    Path(folder).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(folder)
    print(f"Extracted {zip_path} into {folder}")


# Integer columns of the manifest that are empty for authentic samples
# (no forgery). "Int64" is pandas' integer type that accepts missing values;
# without it, pandas would turn them into floats (123 -> 123.0).
NULLABLE_INTEGER_COLUMNS = [
    "level",
    "bbox_row_min", "bbox_col_min", "bbox_row_max", "bbox_col_max",
    "src_offset_row", "src_offset_col",
]


def _fix_integer_columns(manifest):
    for column in NULLABLE_INTEGER_COLUMNS:
        if column in manifest.columns:
            manifest[column] = manifest[column].astype("Int64")
    return manifest


def read_manifest(path):
    """Read a manifest CSV into a pandas DataFrame.

    `sample_id`, `patch_id` and `seed` are read as text, so that leading
    zeros (e.g. "000812") are kept; forgery columns stay integers.
    """
    manifest = pd.read_csv(path, dtype={"sample_id": str, "patch_id": str, "seed": str})
    return _fix_integer_columns(manifest)


def write_manifest(path, manifest):
    """Write a manifest DataFrame to CSV (no index column)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _fix_integer_columns(manifest).to_csv(path, index=False)
