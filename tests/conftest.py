"""Shared fixtures of the tests (run by GitHub Actions, see docs/github_actions.md).

The tests never use real SAR data: they build small synthetic scenes whose
statistics look like Sentinel-1 GRD (gamma-distributed speckle, ENL ~4.4),
with an empty band like the one left by terrain correction.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.copy_move.copy_move import build_dataset  # noqa: E402
from features.sar_io.sar_io import write_patch  # noqa: E402
from features.tiling.tiling import tile_scenes  # noqa: E402

# Smaller patches than the real 512 to stay fast on a CPU (the code is
# size-agnostic), but large enough for the 32-px border margin of level 1.
TEST_PATCH_SIZE = 256


def make_synthetic_scene(rng, height, width, empty_columns):
    """Gamma speckle (ENL 4.4) with an empty band and 1 % sparse zeros."""
    scene = rng.gamma(shape=4.4, scale=0.05 / 4.4, size=(height, width)).astype(np.float32)
    scene[:, :empty_columns] = 0.0               # empty border (terrain correction)
    scene[rng.random(scene.shape) < 0.01] = 0.0  # isolated zeros: normal in SAR
    return scene


@pytest.fixture(scope="session")
def scenes_dir(tmp_path_factory):
    """Five synthetic scenes, one folder each, like on Google Drive.

    4 ascending + 1 descending: tiling mirrors one ascending scene to reach
    3 ascending-like + 2 descending-like.
    """
    root = tmp_path_factory.mktemp("scenes")
    rng = np.random.default_rng(0)
    for k in range(5):
        orbit = "asc" if k < 4 else "desc"
        # 3 x 4 windows of 256; the empty band removes the first column
        # (the last one once mirrored).
        scene = make_synthetic_scene(rng, height=800, width=1100, empty_columns=80)
        write_patch(root / f"scene_{k}_{orbit}" / f"scene_{k}.tif", scene)
    return root


@pytest.fixture(scope="session")
def dataset_dir(scenes_dir, tmp_path_factory):
    """A full small dataset: tiling + level-1 copy-move, 256x256 patches."""
    root = tmp_path_factory.mktemp("dataset")
    tile_scenes(scenes_dir, root, patch_size=TEST_PATCH_SIZE, patches_per_scene=6,
                split_counts={"val": 1, "test": 1}, seed=0)
    build_dataset(root, level=1, seed=0)
    return root
