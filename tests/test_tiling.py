"""Tests of features/tiling (issue #28)."""

import numpy as np

from features.sar_io.sar_io import read_manifest, read_tif
from features.tiling.tiling import (
    assign_splits,
    choose_mirrored_scenes,
    find_scenes,
    mirror_east_west,
    orbit_from_name,
    valid_windows,
)

PATCH = 64


def test_empty_band_is_rejected_sparse_zeros_are_kept():
    image = np.ones((128, 4 * PATCH), dtype=np.float32)
    image[:, :PATCH + 10] = 0.0                  # windows of columns 0 and 1 touch the band
    image[::7, ::7] = 0.0                        # ~2 % isolated zeros everywhere
    positions = valid_windows(image, PATCH, max_nodata_fraction=0.05)
    columns = {col for _, col in positions}
    assert columns == {2 * PATCH, 3 * PATCH}


def test_orbit_is_read_from_the_name():
    assert orbit_from_name("S1A_2024-05-01_Desc_forest") == "desc"
    assert orbit_from_name("zone3_ASC") == "asc"
    assert orbit_from_name("zone3") is None


def test_mirroring_balances_the_orbits():
    orbits = {f"a{k}": "asc" for k in range(7)}
    orbits.update({f"d{k}": "desc" for k in range(3)})
    mirrored = choose_mirrored_scenes(orbits, seed=0)
    assert len(mirrored) == 2                      # 7-3 -> 5-5
    assert all(orbits[name] == "asc" for name in mirrored)
    assert choose_mirrored_scenes({"a": "asc", "d": "desc"}) == set()


def test_east_west_mirror_reverses_columns_only():
    image = np.arange(12).reshape(3, 4)
    mirrored = mirror_east_west(image)
    assert np.array_equal(mirrored[:, 0], image[:, -1])
    assert np.array_equal(mirrored[0], image[0, ::-1])


def test_split_is_by_scene_and_counts_are_respected():
    names = [f"s{k}" for k in range(10)]
    splits = assign_splits(names, {"val": 2, "test": 2}, seed=0)
    values = list(splits.values())
    assert values.count("val") == 2 and values.count("test") == 2 and values.count("train") == 6


def test_val_and_test_take_both_geometries():
    names = [f"s{k}" for k in range(10)]
    geometries = {name: ("asc" if k < 5 else "desc") for k, name in enumerate(names)}

    splits = assign_splits(names, {"val": 2, "test": 2}, seed=0, geometries=geometries)
    for split_name in ["val", "test"]:
        chosen = [geometries[n] for n in names if splits[n] == split_name]
        assert sorted(chosen) == ["asc", "desc"]

    # With one scene each, val and test get different geometries.
    splits = assign_splits(names, {"val": 1, "test": 1}, seed=0, geometries=geometries)
    val = [geometries[n] for n in names if splits[n] == "val"]
    test = [geometries[n] for n in names if splits[n] == "test"]
    assert val != test


def test_forced_split_is_kept():
    names = [f"s{k}" for k in range(10)]
    splits = assign_splits(names, {"val": 2, "test": 2}, seed=0, forced={"s3": "test"})
    assert splits["s3"] == "test"
    assert list(splits.values()).count("test") == 2


def test_tiling_output(dataset_dir, scenes_dir):
    patches = read_manifest(dataset_dir / "patches.csv")
    assert len(patches) == 5 * 6
    # One scene never spans two splits.
    assert patches.groupby("scene")["split"].nunique().max() == 1
    # 4 asc + 1 desc scenes -> one asc scene mirrored -> 3 asc-like, 2 desc-like.
    per_scene = patches.drop_duplicates("scene")
    assert per_scene["mirrored"].sum() == 1
    assert sorted(per_scene["geometry"]) == ["asc", "asc", "asc", "desc", "desc"]

    scenes = find_scenes(scenes_dir)
    for _, row in patches.iterrows():
        patch = read_tif(dataset_dir / "real" / f"{row['patch_id']}.tif")
        assert patch.shape == (256, 256)  # TEST_PATCH_SIZE of conftest.py
        assert (patch == 0).mean() <= 0.05

        # row / col point to the patch in the ORIGINAL scene, mirrored or not.
        scene = read_tif(scenes[row["scene"]])
        original = scene[row["row"]:row["row"] + 256, row["col"]:row["col"] + 256]
        if row["mirrored"]:
            original = mirror_east_west(original)
        assert np.array_equal(patch, original)
