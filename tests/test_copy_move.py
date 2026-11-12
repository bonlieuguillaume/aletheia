"""Tests of features/copy_move (issues #5, #24)."""

import numpy as np

from features.copy_move.copy_move import LEVEL1_BORDER_MARGIN, copy_move_level1
from features.sar_io.sar_io import read_manifest, read_tif


def test_level1_forgery_is_an_exact_copy_with_a_correct_mask():
    rng = np.random.default_rng(0)
    patch = rng.gamma(4.4, 1.0, size=(512, 512)).astype(np.float32)
    forged, mask, params = copy_move_level1(patch, np.random.default_rng([0, 7]))

    # Mask values: background, pasted, source.
    assert set(np.unique(mask)) == {0, 1, 2}
    # Nothing changes outside the pasted region.
    assert np.array_equal(patch[mask != 1], forged[mask != 1])
    # The pasted region is an exact copy of the source region.
    r0, c0, r1, c1 = (params[k] for k in ["bbox_row_min", "bbox_col_min", "bbox_row_max", "bbox_col_max"])
    dr, dc = params["src_offset_row"], params["src_offset_col"]
    assert np.array_equal(forged[r0:r1, c0:c1], patch[r0 + dr:r1 + dr, c0 + dc:c1 + dc])
    # Level 1: 2-5 % of the patch (up to rounding), away from the border.
    assert 0.018 < params["paste_fraction"] < 0.052
    rows, cols = np.nonzero(mask)
    assert rows.min() >= LEVEL1_BORDER_MARGIN and cols.min() >= LEVEL1_BORDER_MARGIN
    assert rows.max() < 512 - LEVEL1_BORDER_MARGIN and cols.max() < 512 - LEVEL1_BORDER_MARGIN


def test_same_seed_same_forgery():
    patch = np.random.default_rng(0).random((512, 512))
    first = copy_move_level1(patch, np.random.default_rng([0, 3]))
    second = copy_move_level1(patch, np.random.default_rng([0, 3]))
    assert np.array_equal(first[0], second[0]) and first[2] == second[2]


def test_dataset_manifest(dataset_dir):
    manifest = read_manifest(dataset_dir / "manifest.csv")
    patches = read_manifest(dataset_dir / "patches.csv")
    # One authentic + one forged sample per patch, on the same side of the split.
    assert len(manifest) == 2 * len(patches)
    assert manifest.groupby("patch_id")["split"].nunique().max() == 1
    assert manifest["label"].value_counts().to_dict() == {0: len(patches), 1: len(patches)}
    # Integer columns survive the CSV round trip.
    forged = manifest[manifest["label"] == 1]
    assert str(forged["bbox_row_min"].dtype) == "Int64"
    # Every forged sample has its mask file.
    for _, row in forged.iterrows():
        assert read_tif(dataset_dir / row["mask_file"]).dtype == np.uint8
