"""Generate copy-move forgeries with their ground-truth masks (issues #5, #24-#26).

For every authentic patch written by `tiling`, one forged twin is made: a
region of the patch is copied and pasted elsewhere in the same patch.

Output, in the dataset folder (next to `real/` and `patches.csv`):

    forged/<patch_id>.tif      forged twin of real/<patch_id>.tif
    masks/<patch_id>.tif       uint8 mask: 0 background, 1 pasted, 2 source
    manifest.csv               one row per sample (authentic AND forged)

Level 1 (#24): one rectangle of 2-5 % of the patch area, source and target
inside the patch, not overlapping, both at least 32 px from the border, no
transform, hard edges.
Level 2 (#25): the same, with a shape drawn among rectangle, ellipse and
slightly irregular polygon, and pastes allowed to touch the border.
Level 3 (#26): 2 to 4 pastes per patch, each as at level 2, none overlapping
another paste nor any source.
Level 4 (#27) will be added to `LEVELS`.

Every forgery is drawn from a random generator seeded by (seed, patch index):
the same seed always gives the same dataset, and any single forgery can be
regenerated on its own.

Command line (from the repository root):

    python features/copy_move/copy_move.py --dataset-dir <dir>
"""

import argparse
import json
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


# ---------------------------------------------------------------------------
# Level 2 (#25): varied simple shapes
# ---------------------------------------------------------------------------

def ellipse_mask(height, width):
    """Boolean mask of the ellipse inscribed in a height x width box.

    A pixel is inside if its centre satisfies (x / rx)^2 + (y / ry)^2 <= 1,
    with x, y measured from the centre of the box. A square box gives a disc.
    """
    rows, cols = np.mgrid[0:height, 0:width]
    # Pixel centres, scaled so that the box spans [-1, 1] in both directions.
    y = (rows + 0.5 - height / 2.0) / (height / 2.0)
    x = (cols + 0.5 - width / 2.0) / (width / 2.0)
    return x ** 2 + y ** 2 <= 1.0


def random_polygon(rng, n_vertices_range=(5, 8), radius_range=(0.7, 1.0)):
    """Draw a slightly irregular polygon, in coordinates normalized to [-1, 1].

    The vertices go once around the centre, at roughly evenly spaced angles,
    each at a random distance from the centre (0.7 to 1 of the half-size):
    the result is a rounded, slightly bumpy shape, never self-intersecting
    (the angles are sorted).

    Returns
    -------
    ys, xs : numpy.ndarray
        Vertex coordinates, in [-1, 1].
    """
    n_vertices = int(rng.integers(n_vertices_range[0], n_vertices_range[1] + 1))
    rotation = rng.uniform(0.0, 2.0 * math.pi)
    step = 2.0 * math.pi / n_vertices

    angles = []
    for k in range(n_vertices):
        # Evenly spaced, each moved by at most a quarter of a step.
        angles.append(rotation + k * step + rng.uniform(-0.25, 0.25) * step)
    radii = rng.uniform(radius_range[0], radius_range[1], size=n_vertices)

    ys = radii * np.sin(angles)
    xs = radii * np.cos(angles)
    return ys, xs


def polygon_mask(height, width, ys, xs):
    """Boolean mask of a polygon (normalized vertices) stretched over a box.

    Classic even-odd rule: from each pixel centre, a ray going right crosses
    the polygon edges an odd number of times if and only if the pixel is
    inside.
    """
    rows, cols = np.mgrid[0:height, 0:width]
    y = (rows + 0.5 - height / 2.0) / (height / 2.0)
    x = (cols + 0.5 - width / 2.0) / (width / 2.0)

    inside = np.zeros((height, width), dtype=bool)
    n_vertices = len(ys)
    for i in range(n_vertices):
        j = i - 1  # previous vertex (index -1 = the last one: closes the polygon)
        y_i, x_i, y_j, x_j = ys[i], xs[i], ys[j], xs[j]
        if y_i == y_j:
            continue  # a horizontal edge is never crossed by a horizontal ray
        straddles = (y_i > y) != (y_j > y)
        x_on_edge = x_i + (x_j - x_i) * (y - y_i) / (y_j - y_i)
        inside ^= straddles & (x < x_on_edge)
    return inside


def shape_mask(shape, height, width, polygon=None):
    """Boolean mask of `shape` ("rectangle", "ellipse" or "polygon") in a box."""
    if shape == "rectangle":
        return np.ones((height, width), dtype=bool)
    if shape == "ellipse":
        return ellipse_mask(height, width)
    if shape == "polygon":
        ys, xs = polygon
        return polygon_mask(height, width, ys, xs)
    raise ValueError(f"Unknown shape: {shape}")


def random_shape(rng, patch_size, area_range, aspect_range, shapes):
    """Draw a shape and a box size so that the SHAPE covers the drawn area.

    A shape does not fill its box (an ellipse fills ~79 % of it), so the box
    is sized in two passes: a first box with the target area, then a box
    rescaled by sqrt(target / measured) so that the shape itself covers the
    target fraction of the patch (up to rounding).

    Returns
    -------
    shape : str
    inside : numpy.ndarray of bool
        The shape, in its box (height x width).
    """
    shape = shapes[int(rng.integers(len(shapes)))]
    polygon = random_polygon(rng) if shape == "polygon" else None
    height, width = random_rectangle_size(rng, patch_size, area_range, aspect_range)
    target_pixels = height * width

    inside = shape_mask(shape, height, width, polygon)
    scale = math.sqrt(target_pixels / max(int(inside.sum()), 1))
    height = min(int(round(height * scale)), patch_size // 2)
    width = min(int(round(width * scale)), patch_size // 2)
    inside = shape_mask(shape, height, width, polygon)
    return shape, inside


def touches_border(row_min, col_min, row_max, col_max, patch_size):
    """Tell whether a box (row_max / col_max excluded) touches the patch border."""
    return row_min == 0 or col_min == 0 or row_max == patch_size or col_max == patch_size


LEVEL2_AREA_RANGE = (0.02, 0.05)              # fraction of the patch covered by the shape
LEVEL2_ASPECT_RANGE = (0.5, 2.0)              # height / width of the shape's box
LEVEL2_BORDER_MARGIN = 0                      # from level 2 on, pastes may touch the border
LEVEL2_SHAPES = ("rectangle", "ellipse", "polygon")


def copy_move_level2(patch, rng, area_range=LEVEL2_AREA_RANGE, aspect_range=LEVEL2_ASPECT_RANGE,
                     margin=LEVEL2_BORDER_MARGIN, shapes=LEVEL2_SHAPES):
    """Make a level-2 copy-move forgery: one paste of a varied simple shape (#25).

    Same as level 1 (one paste, 2-5 % of the patch, exact copy, hard edges,
    no transform), except that:
    - the shape is drawn among a rectangle, an ellipse (or disc) and a
      slightly irregular polygon;
    - source and target may touch the patch border (their effect is measured
      by the metrics without the 16-px border, #32).

    Only the pixels INSIDE the shape are copied: the copied region has the
    shape's outline, not its box's.

    Returns
    -------
    forged, mask, params
        As for `copy_move_level1`; params also hold `touches_border` (1 if the
        source or the target box touches the border).
    """
    patch_size = patch.shape[0]
    shape, inside = random_shape(rng, patch_size, area_range, aspect_range, shapes)
    height, width = inside.shape

    # Draw source and target boxes until they do not overlap (so neither do
    # the shapes, which lie inside them).
    for _ in range(MAX_TRIES):
        src_row, src_col = random_position(rng, patch_size, height, width, margin)
        tgt_row, tgt_col = random_position(rng, patch_size, height, width, margin)
        source_box = (src_row, src_col, src_row + height, src_col + width)
        target_box = (tgt_row, tgt_col, tgt_row + height, tgt_col + width)
        if not boxes_overlap(source_box, target_box):
            break
    else:
        raise RuntimeError("Could not place two non-overlapping shapes")

    # The forgery: inside the shape, the target receives an exact copy of the
    # source; outside the shape, the target box keeps its own pixels.
    forged = patch.copy()
    target_view = forged[tgt_row:tgt_row + height, tgt_col:tgt_col + width]
    source_view = patch[src_row:src_row + height, src_col:src_col + width]
    target_view[inside] = source_view[inside]

    mask = np.zeros(patch.shape, dtype=np.uint8)
    mask[src_row:src_row + height, src_col:src_col + width][inside] = MASK_SOURCE
    mask[tgt_row:tgt_row + height, tgt_col:tgt_col + width][inside] = MASK_PASTED

    on_border = (touches_border(*source_box, patch_size) or touches_border(*target_box, patch_size))
    params = {
        "shape": shape,
        "bbox_row_min": tgt_row,
        "bbox_col_min": tgt_col,
        "bbox_row_max": tgt_row + height,
        "bbox_col_max": tgt_col + width,
        "src_offset_row": src_row - tgt_row,
        "src_offset_col": src_col - tgt_col,
        "paste_fraction": int(inside.sum()) / (patch_size * patch_size),
        "touches_border": int(on_border),
    }
    return forged, mask, params


# ---------------------------------------------------------------------------
# Level 3 (#26): several pastes per patch
# ---------------------------------------------------------------------------

LEVEL3_N_PASTES = (2, 4)                      # number of pastes per patch (both included)
MAX_LAYOUTS = 200                             # attempts to fit all the pastes in one patch


def place_boxes(rng, patch_size, sizes, margin):
    """Place a source and a target box for each size, no two boxes overlapping.

    Boxes are placed one pair at a time; each new box must not overlap any
    box already placed (sources and targets alike), so no paste overlaps
    another paste nor any source.

    Parameters
    ----------
    sizes : list of (height, width)
        One entry per paste.

    Returns
    -------
    list of (source_box, target_box), or None if a pair could not be placed
    after MAX_TRIES attempts (the caller then starts a new layout).
    """
    placed = []      # every box already in the patch
    pairs = []
    for height, width in sizes:
        for _ in range(MAX_TRIES):
            src_row, src_col = random_position(rng, patch_size, height, width, margin)
            tgt_row, tgt_col = random_position(rng, patch_size, height, width, margin)
            source_box = (src_row, src_col, src_row + height, src_col + width)
            target_box = (tgt_row, tgt_col, tgt_row + height, tgt_col + width)

            free = not boxes_overlap(source_box, target_box)
            for box in placed:
                if boxes_overlap(box, source_box) or boxes_overlap(box, target_box):
                    free = False
                    break
            if free:
                placed.extend([source_box, target_box])
                pairs.append((source_box, target_box))
                break
        else:
            return None
    return pairs


def copy_move_level3(patch, rng, n_pastes_range=LEVEL3_N_PASTES, area_range=LEVEL2_AREA_RANGE,
                     aspect_range=LEVEL2_ASPECT_RANGE, margin=LEVEL2_BORDER_MARGIN, shapes=LEVEL2_SHAPES):
    """Make a level-3 copy-move forgery: several pastes in one patch (#26).

    2 to 4 pastes, each drawn as at level 2 (rectangle, ellipse or slightly
    irregular polygon, the shape covering 2-5 % of the patch, exact copy,
    hard edges, border allowed). No paste overlaps another paste nor any
    source. Every source is read from the untouched authentic patch.

    Returns
    -------
    forged, mask
        As for the other levels; the mask is the union of all the pastes (1)
        and of all their sources (2).
    params : dict
        For the manifest: `n_pastes`; `shape` (e.g. "ellipse+polygon");
        `bbox_*` = box enclosing all the pastes; `src_offset_*` left empty
        (one per paste, see `pastes`); `paste_fraction` = all the pasted
        pixels / patch pixels; `touches_border`; `pastes` = the details of
        every paste as JSON text (shape, target box, source offset, fraction).
    """
    patch_size = patch.shape[0]
    n_pastes = int(rng.integers(n_pastes_range[0], n_pastes_range[1] + 1))

    # Draw the shapes, then try to fit them all; if the patch is too crowded
    # for this draw, start again with new shapes.
    for _ in range(MAX_LAYOUTS):
        drawn = []
        for _ in range(n_pastes):
            drawn.append(random_shape(rng, patch_size, area_range, aspect_range, shapes))
        sizes = [inside.shape for _, inside in drawn]
        pairs = place_boxes(rng, patch_size, sizes, margin)
        if pairs is not None:
            break
    else:
        raise RuntimeError(f"Could not fit {n_pastes} non-overlapping pastes")

    forged = patch.copy()
    mask = np.zeros(patch.shape, dtype=np.uint8)
    details = []
    on_border = False
    for (shape, inside), (source_box, target_box) in zip(drawn, pairs):
        s_r0, s_c0, s_r1, s_c1 = source_box
        t_r0, t_c0, t_r1, t_c1 = target_box

        # Same copy as level 2, the source always read from the authentic patch.
        target_view = forged[t_r0:t_r1, t_c0:t_c1]
        target_view[inside] = patch[s_r0:s_r1, s_c0:s_c1][inside]
        mask[s_r0:s_r1, s_c0:s_c1][inside] = MASK_SOURCE
        mask[t_r0:t_r1, t_c0:t_c1][inside] = MASK_PASTED

        if touches_border(*source_box, patch_size) or touches_border(*target_box, patch_size):
            on_border = True
        details.append({
            "shape": shape,
            "bbox": [t_r0, t_c0, t_r1, t_c1],
            "src_offset": [s_r0 - t_r0, s_c0 - t_c0],
            "paste_fraction": round(int(inside.sum()) / (patch_size * patch_size), 5),
        })

    targets = [target_box for _, target_box in pairs]
    params = {
        "shape": "+".join(shape for shape, _ in drawn),
        "n_pastes": n_pastes,
        # Box enclosing all the pastes (one box per paste in `pastes`).
        "bbox_row_min": min(box[0] for box in targets),
        "bbox_col_min": min(box[1] for box in targets),
        "bbox_row_max": max(box[2] for box in targets),
        "bbox_col_max": max(box[3] for box in targets),
        "src_offset_row": None,   # one offset per paste: see `pastes`
        "src_offset_col": None,
        "paste_fraction": int((mask == MASK_PASTED).sum()) / (patch_size * patch_size),
        "touches_border": int(on_border),
        "pastes": json.dumps(details),
    }
    return forged, mask, params


# Difficulty levels available (issues #24-#27); level 4 to be added.
LEVELS = {1: copy_move_level1, 2: copy_move_level2, 3: copy_move_level3}


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
        `dataset_dir/manifest.csv`. Columns: sample_id, label (0 authentic /
        1 forged), file, mask_file, every column of patches.csv (patch_id,
        scene, row, col, orbit, mirrored, geometry, split), and for forged
        samples level, seed and the forgery parameters.
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

        # Every column of patches.csv (scene, position, orbit, split...) is
        # carried over to both samples of the pair.
        common = patch_row.to_dict()
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
