"""Reading and writing the files shared by every feature of aletheia.

Everything that touches the disk goes through this module, so that the file
formats are defined in one place:

- scenes produced by the user's preprocessing (rosarium / SNAP), either a
  GeoTIFF or one band of a BEAM-DIMAP product (ENVI `.hdr` + `.img`);
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


# ENVI "data type" codes -> numpy types. Only the types SNAP writes for a
# single-band SAR product are listed; any other code is refused.
ENVI_DATA_TYPES = {
    1: "u1",   # uint8
    2: "i2",   # int16
    3: "i4",   # int32
    4: "f4",   # float32 (Sigma0 written by SNAP)
    5: "f8",   # float64
    12: "u2",  # uint16
}

# Extensions of the scene files the tiling accepts.
SCENE_EXTENSIONS = (".tif", ".tiff", ".hdr")


def envi_header_value(header_text, key, default=None):
    """Return the value of `key = value` in an ENVI header, as a string.

    Parameters
    ----------
    header_text : str
        Whole content of the `.hdr` file.
    key : str
        Field name, e.g. "samples" or "byte order".
    default : str, optional
        Returned when the field is absent; if None, a missing field is an error.

    Returns
    -------
    str
        The value, stripped.
    """
    for line in header_text.splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip().lower() == key:
            return value.strip()
    if default is None:
        raise ValueError(f"ENVI header: field '{key}' not found")
    return default


def read_envi(hdr_path):
    """Read a single-band ENVI image, i.e. one band of a SNAP BEAM-DIMAP product.

    A BEAM-DIMAP product is a `.dim` file (XML metadata, not needed here) plus
    a `.data/` folder holding, for each band, a raw binary `.img` file and a
    small text `.hdr` file that says how to read it (size, data type, byte
    order). SNAP writes the pixels big-endian, while the CPUs of Colab and of
    a PC are little-endian, so the bytes are swapped after reading.

    Parameters
    ----------
    hdr_path : str or Path
        Path of the `.hdr` file; the `.img` file next to it holds the pixels.

    Returns
    -------
    numpy.ndarray
        2-D array (lines, samples), in the native byte order.
    """
    hdr_path = Path(hdr_path)
    img_path = hdr_path.with_suffix(".img")
    header_text = hdr_path.read_text(encoding="utf-8", errors="replace")

    samples = int(envi_header_value(header_text, "samples"))  # image width
    lines = int(envi_header_value(header_text, "lines"))      # image height
    bands = int(envi_header_value(header_text, "bands", "1"))
    data_type = int(envi_header_value(header_text, "data type"))
    byte_order = int(envi_header_value(header_text, "byte order", "0"))  # 1 = big-endian
    header_offset = int(envi_header_value(header_text, "header offset", "0"))

    if bands != 1:
        raise ValueError(f"{hdr_path}: expected a single-band image, got {bands} bands")
    if data_type not in ENVI_DATA_TYPES:
        raise ValueError(f"{hdr_path}: unsupported ENVI data type {data_type}")

    # Data type as stored in the file, with its byte order.
    if byte_order == 1:
        stored_dtype = np.dtype(">" + ENVI_DATA_TYPES[data_type])
    else:
        stored_dtype = np.dtype("<" + ENVI_DATA_TYPES[data_type])

    # A truncated upload would otherwise silently give a shorter image.
    expected_size = header_offset + lines * samples * stored_dtype.itemsize
    actual_size = img_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{img_path}: {actual_size} bytes, expected {expected_size} "
                         f"({lines} x {samples} pixels of {stored_dtype.itemsize} bytes)")

    image = np.fromfile(img_path, dtype=stored_dtype, count=lines * samples,
                        offset=header_offset)

    # Swap the bytes in place (no second copy of a ~2 GB scene), then look at
    # the same memory with the native byte order.
    if not stored_dtype.isnative:
        image.byteswap(inplace=True)
        image = image.view(stored_dtype.newbyteorder("="))

    return image.reshape(lines, samples)


def read_scene(path):
    """Read a preprocessed scene: a GeoTIFF, or the `.hdr` of a BEAM-DIMAP band.

    Parameters
    ----------
    path : str or Path
        `.tif` / `.tiff` file, or `.hdr` file (with its `.img` next to it).

    Returns
    -------
    numpy.ndarray
        2-D array, with the dtype stored in the file.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".hdr":
        return read_envi(path)
    if suffix in (".tif", ".tiff"):
        return read_tif(path)
    raise ValueError(f"{path}: unknown scene format (expected one of {SCENE_EXTENSIONS})")


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
    "touches_border",
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
