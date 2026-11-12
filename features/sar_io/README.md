# sar_io

Library module, no notebook: reading and writing the files shared by all the
features, so that every file format is defined in one place.

| Function | File |
|---|---|
| `read_tif` | any single-band `.tif`: scene GeoTIFF, patch, mask |
| `write_patch` | SAR patch: float32, one band, no georeferencing, uncompressed |
| `write_mask` | mask: uint8, one band, zlib (lossless) |
| `write_probability_map` / `read_probability_map` | predicted anomaly map: probabilities quantized to uint8 (0-255) |
| `read_manifest` / `write_manifest` | manifest CSV (pandas) |
| `zip_folder` / `unzip_folder` | one zip per dataset: Drive is slow with many small files |

Mask values: `MASK_BACKGROUND = 0`, `MASK_PASTED = 1` (the forgery),
`MASK_SOURCE = 2` (where the paste was copied from, when it lies in the
patch). Authentic patches have no mask file.

`tifffile` is imported inside the functions, so the modules that do not read
images can be imported without it.
