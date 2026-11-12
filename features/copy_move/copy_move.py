"""Generate copy-move forgeries with their ground-truth masks (issues #5, #24).

For every authentic patch written by `tiling`, one forged twin is made: a
region of the patch is copied and pasted elsewhere in the same patch.

Output, in the dataset folder (next to `real/` and `patches.csv`):

    forged/<patch_id>.tif      forged twin of real/<patch_id>.tif
    masks/<patch_id>.tif       uint8 mask: 0 background, 1 pasted, 2 source
    manifest.csv               one row per sample (authentic AND forged)

Level 1 (#24): one rectangle of 2-5 % of the patch area, source and target
inside the patch, not overlapping, both at least 32 px from the border, no
transform, hard edges. Other levels (#25-#27) will be added to `LEVELS`.

Every forgery is drawn from a random generator seeded by (seed, patch index):
the same seed always gives the same dataset, and any single forgery can be
regenerated on its own.

Command line (from the repository root):

    python features/copy_move/copy_move.py --dataset-dir <dir>
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.sar_io.sar_io import (  # noqa: E402
    MASK_PASTED,
    MASK_SOURCE,
    read_manifest,
    read_tif,
    write_manifest,
    write_mask,
    write_patch,
)

SEED = 0
LEVEL1_AREA_RANGE = (0.02, 0.05)   # fraction of the patch area
LEVEL1_ASPECT_RANGE = (0.5, 2.0)   # height / width of the rectangle
LEVEL1_BORDER_MARGIN = 32          # px kept free along the patch border
MAX_TRIES = 1000                   # attempts to place two non-overlapping boxes


def boxes_overlap(box_a, box_b):
    """Tell whether two boxes (row_min, col_min, row_max, col_max) overlap.

    `row_max` and `col_max` are excluded, as in numpy slicing.
    """
    a_r0, a_c0, a_r1, a_c1 = box_a
    b_r0, b_c0, b_r1, b_c1 = box_b
    separated = a_r1 <= b_r0 or b_r1 <= a_r0 or a_c1 <= b_c0 or b_c1 <= a_c0
    return not separated


def random_rectangle_size(rng, patch_size, area_range, aspect_range):
    """Draw the height and width of a rectangle covering a given area fraction.

    The area fraction and the aspect ratio (height / width) are drawn
    uniformly; height and width follow from them:
        area = height * width  and  aspect = height / width
        =>  height = sqrt(area * aspect),  width = sqrt(area / aspect)
    """
    area_fraction = rng.uniform(area_range[0], area_range[1])
    aspect = rng.uniform(aspect_range[0], aspect_range[1])
    area_pixels = area_fraction * patch_size * patch_size

    height = int(round(math.sqrt(area_pixels * aspect)))
    width = int(round(math.sqrt(area_pixels / aspect)))
    return height, width


def random_position(rng, patch_size, height, width, margin):
    """Draw the top-left corner of a box that stays `margin` px from the border."""
    row = int(rng.integers(margin, patch_size - margin - height + 1))
    col = int(rng.integers(margin, patch_size - margin - width + 1))
    return row, col


def copy_move_level1(patch, rng, area_range=LEVEL1_AREA_RANGE, aspect_range=LEVEL1_ASPECT_RANGE,
                     margin=LEVEL1_BORDER_MARGIN):
    """Make a level-1 copy-move forgery of one patch (issue #24).

    Parameters
    ----------
    patch : numpy.ndarray
        Square authentic patch (2-D).
    rng : numpy.random.Generator
        Random generator: the whole forgery depends on it only.
    area_range, aspect_range, margin
        See the module constants.

    Returns
    -------
    forged : numpy.ndarray
        Copy of the patch with the source region pasted onto the target.
    mask : numpy.ndarray
        uint8 array, 0 background, 1 pasted (target), 2 source.
    params : dict
        What was done, for the manifest: shape, target box, source offset.
    """
    patch_size = patch.shape[0]
    height, width = random_rectangle_size(rng, patch_size, area_range, aspect_range)

    # Draw source and target positions until the two boxes do not overlap.
    for _ in range(MAX_TRIES):
        src_row, src_col = random_position(rng, patch_size, height, width, margin)
        tgt_row, tgt_col = random_position(rng, patch_size, height, width, margin)
        source_box = (src_row, src_col, src_row + height, src_col + width)
        target_box = (tgt_row, tgt_col, tgt_row + height, tgt_col + width)
        if not boxes_overlap(source_box, target_box):
            break
    else:
        raise RuntimeError("Could not place two non-overlapping rectangles")

    # The forgery itself: the target region receives an exact copy of the
    # source region. The pasted speckle is therefore identical, pixel for
    # pixel, to the source speckle (a cue in itself, see issue #13).
    forged = patch.copy()
    forged[tgt_row:tgt_row + height, tgt_col:tgt_col + width] = \
        patch[src_row:src_row + height, src_col:src_col + width]

    mask = np.zeros(patch.shape, dtype=np.uint8)
    mask[src_row:src_row + height, src_col:src_col + width] = MASK_SOURCE
    mask[tgt_row:tgt_row + height, tgt_col:tgt_col + width] = MASK_PASTED

    params = {
        "shape": "rectangle",
        # Bounding box of the pasted region (row_max / col_max excluded).
        "bbox_row_min": tgt_row,
        "bbox_col_min": tgt_col,
        "bbox_row_max": tgt_row + height,
        "bbox_col_max": tgt_col + width,
        # Where the paste comes from, relative to where it was pasted.
        "src_offset_row": src_row - tgt_row,
        "src_offset_col": src_col - tgt_col,
        "paste_fraction": (height * width) / (patch_size * patch_size),
    }
    return forged, mask, params


# Difficulty levels available (issues #24-#27); levels 2-4 to be added.
LEVELS = {1: copy_move_level1}


def build_dataset(dataset_dir, level=1, seed=SEED):
    """Make the forged twin of every authentic patch and write the manifest.

    Parameters
    ----------
    dataset_dir : str or Path
        Folder written by `tiling` (holds `real/` and `patches.csv`).
    level : int
        Difficulty level (see `LEVELS`).
    seed : int
        Global seed of the forgeries.

    Returns
    -------
    pandas.DataFrame
        The full manifest, one row per sample, also written to
        `dataset_dir/manifest.csv`. Columns:
        sample_id, patch_id, label (0 authentic / 1 forged), file, mask_file,
        scene, row, col, split, level, seed, and the forgery parameters.
    """
    dataset_dir = Path(dataset_dir)
    patches = read_manifest(dataset_dir / "patches.csv")
    make_forgery = LEVELS[level]

    rows = []
    for patch_index, patch_row in patches.iterrows():
        patch_id = patch_row["patch_id"]
        patch = read_tif(dataset_dir / "real" / f"{patch_id}.tif")

        # One generator per patch: forgery k only depends on (seed, k).
        forgery_seed = [seed, patch_index]
        rng = np.random.default_rng(forgery_seed)
        forged, mask, params = make_forgery(patch, rng)

        write_patch(dataset_dir / "forged" / f"{patch_id}.tif", forged)
        write_mask(dataset_dir / "masks" / f"{patch_id}.tif", mask)

        common = {
            "patch_id": patch_id,
            "scene": patch_row["scene"],
            "row": patch_row["row"],
            "col": patch_row["col"],
            "split": patch_row["split"],
        }
        # The authentic sample: no mask file, no forgery parameters.
        rows.append({"sample_id": f"{patch_id}_real", "label": 0,
                     "file": f"real/{patch_id}.tif", "mask_file": "", **common})
        # Its forged twin.
        rows.append({"sample_id": f"{patch_id}_forged", "label": 1,
                     "file": f"forged/{patch_id}.tif", "mask_file": f"masks/{patch_id}.tif",
                     **common, "level": level, "seed": f"{seed}-{patch_index}", **params})

    manifest = pd.DataFrame(rows)
    write_manifest(dataset_dir / "manifest.csv", manifest)
    print(f"Done: {len(manifest)} samples ({len(patches)} authentic + {len(patches)} forged)")
    return manifest


def show_forgery(dataset_dir, patch_id, input_scale="linear"):
    """Show an authentic patch, its forged twin and the mask, side by side.

    Used in the notebook to check by eye that the forgeries look as intended.
    Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    from features.evaluation.evaluation import to_display

    dataset_dir = Path(dataset_dir)
    real = read_tif(dataset_dir / "real" / f"{patch_id}.tif")
    forged = read_tif(dataset_dir / "forged" / f"{patch_id}.tif")
    mask = read_tif(dataset_dir / "masks" / f"{patch_id}.tif")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(to_display(real, input_scale), cmap="gray")
    axes[0].set_title(f"Authentic {patch_id}")
    axes[1].imshow(to_display(forged, input_scale), cmap="gray")
    axes[1].set_title("Forged")
    axes[2].imshow(mask, cmap="viridis", vmin=0, vmax=2)
    axes[2].set_title("Mask (0 background, 1 pasted, 2 source)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", required=True, help="folder written by tiling")
    parser.add_argument("--level", type=int, default=1, choices=sorted(LEVELS))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    build_dataset(args.dataset_dir, level=args.level, seed=args.seed)


if __name__ == "__main__":
    main()
