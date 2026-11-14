"""Tests of features/copy_move (issues #5, #24-#26)."""

import json

import numpy as np

from features.copy_move.copy_move import (
    LEVEL1_BORDER_MARGIN,
    LEVEL2_SHAPES,
    build_dataset,
    copy_move_level1,
    copy_move_level2,
    copy_move_level3,
    ellipse_mask,
    polygon_mask,
)
from features.sar_io.sar_io import read_manifest, read_tif
from features.tiling.tiling import tile_scenes


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


# ---------------------------------------------------------------------------
# Level 2 (#25): varied simple shapes
# ---------------------------------------------------------------------------

def check_exact_copy(patch, forged, mask, params):
    """The pasted pixels are an exact copy of the source pixels, nothing else changed."""
    assert np.array_equal(patch[mask != 1], forged[mask != 1])
    rows, cols = np.nonzero(mask == 1)
    dr, dc = params["src_offset_row"], params["src_offset_col"]
    assert np.array_equal(forged[rows, cols], patch[rows + dr, cols + dc])
    # The source has exactly the shape of the paste, at the offset.
    assert np.all(mask[rows + dr, cols + dc] == 2)
    assert (mask == 2).sum() == (mask == 1).sum()


def test_level2_shapes_are_exact_copies_of_the_right_size():
    patch = np.random.default_rng(0).gamma(4.4, 1.0, size=(512, 512)).astype(np.float32)
    shapes_seen, border_seen = set(), 0
    for k in range(60):
        forged, mask, params = copy_move_level2(patch, np.random.default_rng([0, k]))
        check_exact_copy(patch, forged, mask, params)
        # The SHAPE (not its box) covers 2-5 % of the patch, up to rounding.
        assert params["paste_fraction"] == (mask == 1).sum() / 512 ** 2
        assert 0.017 < params["paste_fraction"] < 0.055
        shapes_seen.add(params["shape"])
        border_seen += params["touches_border"]
    assert shapes_seen == set(LEVEL2_SHAPES)
    assert border_seen > 0  # pastes may touch the border from level 2 on


def test_ellipse_and_polygon_masks():
    disc = ellipse_mask(101, 101)
    # A disc fills pi / 4 of its square box.
    assert abs(disc.mean() - np.pi / 4) < 0.01
    assert disc[50, 50] and not disc[0, 0]
    # A regular square polygon (corners on the axes) fills half of its box.
    square = polygon_mask(100, 100, np.array([1.0, 0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0, -1.0]))
    assert abs(square.mean() - 0.5) < 0.02


def test_level2_dataset(scenes_dir, tmp_path):
    tile_scenes(scenes_dir, tmp_path, patch_size=256, patches_per_scene=2,
                split_counts={"val": 1, "test": 1}, seed=0)
    manifest = build_dataset(tmp_path, level=2, seed=0)
    forged = manifest[manifest["label"] == 1]
    assert set(forged["level"]) == {2}
    assert "shape" in manifest.columns and "touches_border" in manifest.columns
    reread = read_manifest(tmp_path / "manifest.csv")
    assert str(reread["touches_border"].dtype) == "Int64"


# ---------------------------------------------------------------------------
# Level 3 (#26): several pastes per patch
# ---------------------------------------------------------------------------

def test_level3_several_exact_pastes_without_overlap():
    patch = np.random.default_rng(0).gamma(4.4, 1.0, size=(512, 512)).astype(np.float32)
    n_seen = set()
    for k in range(40):
        forged, mask, params = copy_move_level3(patch, np.random.default_rng([0, k]))
        pastes = json.loads(params["pastes"])
        assert len(pastes) == params["n_pastes"]
        n_seen.add(params["n_pastes"])

        # Nothing changes outside the pasted pixels; as many source as pasted pixels.
        assert np.array_equal(patch[mask != 1], forged[mask != 1])
        assert (mask == 2).sum() == (mask == 1).sum()
        # Each paste is an exact copy of its own source. The boxes do not
        # overlap, so the pasted pixels inside a paste's box are its own.
        for paste in pastes:
            r0, c0, r1, c1 = paste["bbox"]
            dr, dc = paste["src_offset"]
            rows, cols = np.nonzero(mask[r0:r1, c0:c1] == 1)
            rows, cols = rows + r0, cols + c0
            assert np.array_equal(forged[rows, cols], patch[rows + dr, cols + dc])
            assert np.all(mask[rows + dr, cols + dc] == 2)
            assert 0.017 < paste["paste_fraction"] < 0.055
        # The total is the sum of the pastes.
        total = sum(paste["paste_fraction"] for paste in pastes)
        assert abs(total - params["paste_fraction"]) < 1e-3
    assert n_seen == {2, 3, 4}


def test_level3_dataset(scenes_dir, tmp_path):
    tile_scenes(scenes_dir, tmp_path, patch_size=256, patches_per_scene=2,
                split_counts={"val": 1, "test": 1}, seed=0)
    build_dataset(tmp_path, level=3, seed=0)
    manifest = read_manifest(tmp_path / "manifest.csv")
    forged = manifest[manifest["label"] == 1]
    assert set(forged["level"]) == {3}
    assert str(manifest["n_pastes"].dtype) == "Int64"
    # The details of every paste survive the CSV round trip.
    for _, row in forged.iterrows():
        assert len(json.loads(row["pastes"])) == row["n_pastes"]
