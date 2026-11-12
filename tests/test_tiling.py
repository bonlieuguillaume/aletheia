"""Tests of features/tiling (issue #28)."""

import numpy as np

from features.sar_io.sar_io import read_manifest, read_tif
from features.tiling.tiling import assign_splits, valid_windows

PATCH = 64


def test_empty_band_is_rejected_sparse_zeros_are_kept():
    image = np.ones((128, 4 * PATCH), dtype=np.float32)
    image[:, :PATCH + 10] = 0.0                  # windows of columns 0 and 1 touch the band
    image[::7, ::7] = 0.0                        # ~2 % isolated zeros everywhere
    positions = valid_windows(image, PATCH, max_nodata_fraction=0.05)
    columns = {col for _, col in positions}
    assert columns == {2 * PATCH, 3 * PATCH}


def test_split_is_by_scene_and_counts_are_respected():
    names = [f"s{k}" for k in range(10)]
    splits = assign_splits(names, {"val": 2, "test": 2}, seed=0)
    values = list(splits.values())
    assert values.count("val") == 2 and values.count("test") == 2 and values.count("train") == 6


def test_forced_split_is_kept():
    names = [f"s{k}" for k in range(10)]
    splits = assign_splits(names, {"val": 2, "test": 2}, seed=0, forced={"s3": "test"})
    assert splits["s3"] == "test"
    assert list(splits.values()).count("test") == 2


def test_tiling_output(dataset_dir):
    patches = read_manifest(dataset_dir / "patches.csv")
    assert len(patches) == 5 * 6
    # One scene never spans two splits.
    assert patches.groupby("scene")["split"].nunique().max() == 1
    # Patches are written, at the right size, without the empty band.
    for _, row in patches.iterrows():
        patch = read_tif(dataset_dir / "real" / f"{row['patch_id']}.tif")
        assert patch.shape == (256, 256)  # TEST_PATCH_SIZE of conftest.py
        assert (patch == 0).mean() <= 0.05
