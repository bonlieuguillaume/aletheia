"""Tile the preprocessed Sentinel-1 scenes into 512x512 patches (issue #28).

Input: one folder per scene, each holding one VV GeoTIFF (and the XML of the
SNAP graph that produced it), as put on Google Drive by the user:

    scenes_dir/
        <scene_name>/<anything>.tif
        <scene_name>/<anything>.xml      (not read here, kept with the data)

The orbit of each scene (ascending / descending) is read from its folder name
(containing "asc" or "desc") or given explicitly.

Output, in `out_dir`:

    real/<patch_id>.tif        authentic patches (float32, no georeferencing)
    patches.csv                one row per patch: patch_id, scene, row, col,
                               orbit, mirrored, geometry, split

Choices (see docs/report.md, section 2):
- crop, never resize (resizing would destroy fine-scale traces);
- a patch is rejected if more than 5 % of its pixels are no-data;
- a fixed number of patches is drawn at random in each scene, so that every
  scene weighs the same in the dataset;
- ascending and descending geometries are balanced: if needed, some scenes of
  the majority orbit are mirrored east-west, which turns their look direction
  into the one of the other orbit (see `mirror_east_west`);
- the train / val / test split is made by scene, never by patch, and val and
  test take as many scenes of each geometry.

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

ORBITS = ["asc", "desc"]


# ---------------------------------------------------------------------------
# Scenes and orbits
# ---------------------------------------------------------------------------

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


def orbit_from_name(scene_name):
    """Read the orbit from a scene name: "asc" or "desc", None if absent.

    "desc" is tested first: a name containing "desc" never means ascending.
    """
    name = scene_name.lower()
    if "desc" in name:
        return "desc"
    if "asc" in name:
        return "asc"
    return None


def scene_orbits(scene_names, orbits=None):
    """Orbit of every scene, from `orbits` if given, else from the names.

    Raises an error listing the scenes whose orbit is unknown: the balance of
    the dataset and the split depend on it, so it must not be guessed.
    """
    if orbits is None:
        orbits = {}

    result = {}
    unknown = []
    for name in scene_names:
        orbit = orbits.get(name, orbit_from_name(name))
        if orbit not in ORBITS:
            unknown.append(name)
        result[name] = orbit

    if unknown:
        raise ValueError(f"Orbit unknown for {unknown}: put 'asc' or 'desc' in the folder "
                         f"name, or give them in `orbits`")
    return result


def choose_mirrored_scenes(orbits, seed=SEED):
    """Choose the scenes to mirror so that both geometries are balanced.

    With n_asc ascending and n_desc descending scenes, mirroring k scenes of
    the majority moves them to the other geometry; k = |n_asc - n_desc| // 2
    gives the best balance (5-5 from 7-3 by mirroring 2 scenes). The scenes to
    mirror are drawn at random with the seed.

    Returns
    -------
    set of str
        Names of the scenes to mirror (empty if already balanced).
    """
    by_orbit = {orbit: sorted(n for n, o in orbits.items() if o == orbit) for orbit in ORBITS}
    majority = max(ORBITS, key=lambda orbit: len(by_orbit[orbit]))
    minority = [orbit for orbit in ORBITS if orbit != majority][0]
    n_to_mirror = (len(by_orbit[majority]) - len(by_orbit[minority])) // 2

    rng = np.random.default_rng(seed)
    chosen = rng.choice(by_orbit[majority], size=n_to_mirror, replace=False)
    return set(str(name) for name in chosen)


def mirror_east_west(image):
    """Mirror a north-up image east-west (left <-> right).

    Sentinel-1 looks to the right of its track. On a terrain-corrected,
    north-up image, an ascending pass looks towards ~78 deg (east-north-east)
    and a descending one towards ~282 deg (west-north-west). An east-west
    mirror maps a look direction d to 360 - d, i.e. 78 deg to 282 deg: the
    mirrored ascending image has exactly the look direction of a descending
    one (shadows and layover on the other side). A 180 deg rotation would be
    off by ~24 deg, and a north-south mirror or a 90 deg rotation would give
    look directions Sentinel-1 never has.
    """
    return image[:, ::-1]


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def assign_splits(scene_names, split_counts=None, seed=SEED, forced=None, geometries=None):
    """Assign each scene to train, val or test.

    The split is made by scene, so that the test only contains landscapes
    never seen during training (docs/report.md, section 2.3). If the
    geometries are given, val and test take as many scenes of each geometry
    (e.g. 2 val scenes = 1 ascending-like + 1 descending-like), so that both
    are measured.

    Parameters
    ----------
    scene_names : list of str
        All the scenes.
    split_counts : dict, optional
        Number of scenes for "val" and "test"; the others go to "train".
    seed : int
        Seed of the random draw of the val / test scenes.
    forced : dict, optional
        Scene name -> split, to choose some scenes by hand. The other scenes
        are drawn at random.
    geometries : dict, optional
        Scene name -> "asc" / "desc" (after mirroring). None = one group.

    Returns
    -------
    dict
        Scene name -> "train", "val" or "test".
    """
    if split_counts is None:
        split_counts = SPLIT_COUNTS
    if forced is None:
        forced = {}
    if geometries is None:
        geometries = {name: "all" for name in scene_names}

    splits = dict(forced)
    rng = np.random.default_rng(seed)

    # Scenes not chosen by hand, per geometry, in a seeded random order.
    groups = sorted(set(geometries.values()))
    remaining = {}
    for group in groups:
        names = [n for n in sorted(scene_names) if geometries[n] == group and n not in forced]
        rng.shuffle(names)
        remaining[group] = names

    # Share the val / test scenes between the geometries, one at a time in
    # turn. The turn carries on from val to test, so that with 1 val and 1
    # test scene, one is ascending-like and the other descending-like.
    group_index = 0
    for split_name in ["val", "test"]:
        already = sum(1 for s in splits.values() if s == split_name)
        missing = max(split_counts.get(split_name, 0) - already, 0)

        attempts = 0
        while missing > 0 and attempts < 10 * len(groups):
            group = groups[group_index % len(groups)]
            if remaining[group]:
                splits[remaining[group].pop(0)] = split_name
                missing -= 1
            group_index += 1
            attempts += 1

    for group in groups:
        for name in remaining[group]:
            splits[name] = "train"

    return splits


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

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
                orbits=None, balance_orbits=True, seed=SEED):
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
    orbits : dict, optional
        Scene name -> "asc" / "desc"; scenes not listed are read from their
        name (see `scene_orbits`).
    balance_orbits : bool
        Mirror scenes of the majority orbit to balance the two geometries.

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

    # Orbit of each scene, and the geometry it will have once tiled: the
    # opposite one for the mirrored scenes.
    orbit_of = scene_orbits(list(scenes), orbits)
    mirrored = choose_mirrored_scenes(orbit_of, seed) if balance_orbits else set()
    geometry_of = {}
    for name, orbit in orbit_of.items():
        if name in mirrored:
            geometry_of[name] = "desc" if orbit == "asc" else "asc"
        else:
            geometry_of[name] = orbit

    splits = assign_splits(list(scenes), split_counts, seed, forced_splits, geometry_of)
    print("Scenes:")
    for name in sorted(scenes):
        note = "  (mirrored east-west)" if name in mirrored else ""
        print(f"  {splits[name]:5s}  {orbit_of[name]:4s} -> {geometry_of[name]:4s}  {name}{note}")

    rows = []
    patch_counter = 0
    for scene_index, (scene_name, tif_path) in enumerate(sorted(scenes.items())):
        print(f"Reading {tif_path.name} ...")
        image = read_tif(tif_path).astype(np.float32)
        scene_width = image.shape[1]
        if scene_name in mirrored:
            image = mirror_east_west(image)

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

            # Position in the ORIGINAL scene, so it can be traced back (and
            # used later for the incidence angle, #30): for a mirrored scene,
            # the window starting at column `col` of the mirrored image covers
            # columns [width - col - patch_size, width - col) of the original.
            original_col = scene_width - col - patch_size if scene_name in mirrored else col

            patch_id = f"{patch_counter:06d}"
            write_patch(out_dir / "real" / f"{patch_id}.tif", patch)
            rows.append({
                "patch_id": patch_id,
                "scene": scene_name,
                "row": row,
                "col": original_col,
                "orbit": orbit_of[scene_name],          # orbit of the acquisition
                "mirrored": scene_name in mirrored,     # patch mirrored east-west
                "geometry": geometry_of[scene_name],    # look geometry of the patch
                "split": splits[scene_name],
            })
            patch_counter += 1
        print(f"  {n_kept} patches written")

    manifest = pd.DataFrame(rows)
    write_manifest(out_dir / "patches.csv", manifest)
    print(f"Done: {len(manifest)} patches in {out_dir}")
    print(manifest.groupby(["split", "geometry"]).size().to_string())
    return manifest


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    parser.add_argument("--scenes-dir", required=True, help="one sub-folder per scene, 'asc'/'desc' in its name")
    parser.add_argument("--out-dir", required=True, help="where real/ and patches.csv go")
    parser.add_argument("--patches-per-scene", type=int, default=PATCHES_PER_SCENE)
    parser.add_argument("--max-nodata", type=float, default=MAX_NODATA_FRACTION)
    parser.add_argument("--no-balance", action="store_true", help="do not mirror scenes to balance the orbits")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    tile_scenes(args.scenes_dir, args.out_dir, patches_per_scene=args.patches_per_scene,
                max_nodata_fraction=args.max_nodata, balance_orbits=not args.no_balance,
                seed=args.seed)


if __name__ == "__main__":
    main()
