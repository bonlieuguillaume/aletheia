"""Tile the preprocessed Sentinel-1 scenes into 512x512 patches (issue #28).

Input: one folder per scene, each holding one VV GeoTIFF (and the XML of the
SNAP graph that produced it), as put on Google Drive by the user:

    scenes_dir/
        <scene_name>/<anything>.tif
        <scene_name>/<anything>.xml      (not read here, kept with the data)

Output, in `out_dir`:

    real/<patch_id>.tif        authentic patches (float32, no georeferencing)
    patches.csv                one row per patch: patch_id, scene, row, col, split

Choices (see docs/report.md, section 2):
- crop, never resize (resizing would destroy fine-scale traces);
- a patch is rejected if more than 5 % of its pixels are no-data;
- a fixed number of patches is drawn at random in each scene, so that every
  scene weighs the same in the dataset;
- the train / val / test split is made by scene, never by patch.

Command line (from the repository root):

    python features/tiling/tiling.py --scenes-dir <dir> --out-dir <dir>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make "features.<name>" importable when this file is run directly or from
# its notebook (see "Conventions" in CLAUDE.md).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.sar_io.sar_io import read_tif, write_manifest, write_patch  # noqa: E402

# Default parameters (the notebook passes its own values).
PATCH_SIZE = 512
PATCHES_PER_SCENE = 200
MAX_NODATA_FRACTION = 0.05
SPLIT_COUNTS = {"val": 2, "test": 2}  # the remaining scenes go to "train"
SEED = 0


def find_scenes(scenes_dir):
    """List the scenes: one sub-folder per scene, holding one `.tif`.

    Parameters
    ----------
    scenes_dir : str or Path
        Folder holding one sub-folder per scene.

    Returns
    -------
    dict
        Scene name (the sub-folder name) -> path of its `.tif` file,
        sorted by scene name so that the order never depends on the disk.
    """
    scenes = {}
    for scene_dir in sorted(Path(scenes_dir).iterdir()):
        if not scene_dir.is_dir():
            continue

        tif_files = sorted(scene_dir.glob("*.tif")) + sorted(scene_dir.glob("*.tiff"))
        if len(tif_files) == 0:
            print(f"Skipping {scene_dir.name}: no .tif file")
            continue
        if len(tif_files) > 1:
            raise ValueError(f"{scene_dir}: expected one .tif file, found {len(tif_files)}")

        scenes[scene_dir.name] = tif_files[0]
    return scenes


def assign_splits(scene_names, split_counts=None, seed=SEED, forced=None):
    """Assign each scene to train, val or test.

    The split is made by scene, so that the test only contains landscapes
    never seen during training (docs/report.md, section 2.3).

    Parameters
    ----------
    scene_names : list of str
        All the scenes.
    split_counts : dict, optional
        Number of scenes for "val" and "test"; the others go to "train".
    seed : int
        Seed of the random draw of the val / test scenes.
    forced : dict, optional
        Scene name -> split, to choose some scenes by hand (e.g. keep a given
        scene for the test). The other scenes are drawn at random.

    Returns
    -------
    dict
        Scene name -> "train", "val" or "test".
    """
    if split_counts is None:
        split_counts = SPLIT_COUNTS
    if forced is None:
        forced = {}

    splits = dict(forced)

    # Scenes not chosen by hand, in a seeded random order.
    remaining = [name for name in sorted(scene_names) if name not in forced]
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)

    # Fill val, then test, up to the requested counts; the rest is train.
    for split_name in ["val", "test"]:
        already = sum(1 for s in splits.values() if s == split_name)
        missing = split_counts.get(split_name, 0) - already
        for _ in range(max(missing, 0)):
            if remaining:
                splits[remaining.pop(0)] = split_name
    for name in remaining:
        splits[name] = "train"

    return splits


def nodata_mask(image):
    """Return a boolean array, True where the pixel is no-data.

    No-data = 0 or not a finite number (NaN, inf). Terrain correction leaves
    large empty areas filled with 0 or NaN around the rotated footprint.
    """
    return (image == 0) | ~np.isfinite(image)


def valid_windows(image, patch_size=PATCH_SIZE, max_nodata_fraction=MAX_NODATA_FRACTION):
    """List the patch positions of a scene that pass the no-data rule.

    The scene is cut into a grid of non-overlapping windows. A window is kept
    if at most `max_nodata_fraction` of its pixels are no-data: isolated zeros
    are a normal part of SAR images and are kept, whereas the empty borders
    left by terrain correction fill whole windows and get them rejected.

    Returns
    -------
    list of (int, int)
        Top-left (row, col) of every valid window.
    """
    is_nodata = nodata_mask(image)
    n_rows = image.shape[0] // patch_size
    n_cols = image.shape[1] // patch_size

    positions = []
    for i in range(n_rows):
        for j in range(n_cols):
            row = i * patch_size
            col = j * patch_size
            window = is_nodata[row:row + patch_size, col:col + patch_size]
            if window.mean() <= max_nodata_fraction:
                positions.append((row, col))
    return positions


def tile_scenes(scenes_dir, out_dir, patch_size=PATCH_SIZE, patches_per_scene=PATCHES_PER_SCENE,
                max_nodata_fraction=MAX_NODATA_FRACTION, split_counts=None, forced_splits=None,
                seed=SEED):
    """Tile every scene and write the authentic patches and their manifest.

    Parameters
    ----------
    scenes_dir : str or Path
        Folder with one sub-folder per scene (see the module docstring).
    out_dir : str or Path
        Where `real/` and `patches.csv` are written.
    patch_size, patches_per_scene, max_nodata_fraction, seed
        See the module constants.
    split_counts, forced_splits
        See `assign_splits`.

    Returns
    -------
    pandas.DataFrame
        The manifest of the authentic patches (also written to
        `out_dir/patches.csv`).
    """
    out_dir = Path(out_dir)
    scenes = find_scenes(scenes_dir)
    if not scenes:
        raise RuntimeError(f"No scene found in {scenes_dir}")

    splits = assign_splits(list(scenes), split_counts, seed, forced_splits)
    print("Split by scene:")
    for name in sorted(scenes):
        print(f"  {splits[name]:5s}  {name}")

    rows = []
    patch_counter = 0
    for scene_index, (scene_name, tif_path) in enumerate(sorted(scenes.items())):
        print(f"Reading {tif_path.name} ...")
        image = read_tif(tif_path).astype(np.float32)

        positions = valid_windows(image, patch_size, max_nodata_fraction)
        print(f"  {len(positions)} valid windows of {patch_size}x{patch_size}")

        # Draw the patches at random among the valid windows. One random
        # generator per scene, seeded by (seed, scene index): the draw of a
        # scene does not change if another scene is added or removed.
        rng = np.random.default_rng([seed, scene_index])
        n_kept = min(patches_per_scene, len(positions))
        chosen = rng.choice(len(positions), size=n_kept, replace=False)
        chosen = sorted(chosen)  # keep the reading order of the scene

        for k in chosen:
            row, col = positions[k]
            patch = image[row:row + patch_size, col:col + patch_size]

            patch_id = f"{patch_counter:06d}"
            write_patch(out_dir / "real" / f"{patch_id}.tif", patch)
            rows.append({
                "patch_id": patch_id,
                "scene": scene_name,
                "row": row,  # position in the scene, kept for later
                "col": col,  # (e.g. incidence angle, issue #30)
                "split": splits[scene_name],
            })
            patch_counter += 1
        print(f"  {n_kept} patches written")

    manifest = pd.DataFrame(rows)
    write_manifest(out_dir / "patches.csv", manifest)
    print(f"Done: {len(manifest)} patches in {out_dir}")
    print(manifest.groupby("split").size().to_string())
    return manifest


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    parser.add_argument("--scenes-dir", required=True, help="one sub-folder per scene")
    parser.add_argument("--out-dir", required=True, help="where real/ and patches.csv go")
    parser.add_argument("--patches-per-scene", type=int, default=PATCHES_PER_SCENE)
    parser.add_argument("--max-nodata", type=float, default=MAX_NODATA_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    tile_scenes(args.scenes_dir, args.out_dir, patches_per_scene=args.patches_per_scene,
                max_nodata_fraction=args.max_nodata, seed=args.seed)


if __name__ == "__main__":
    main()
