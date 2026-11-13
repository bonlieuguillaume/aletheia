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


def write_envi(hdr_path, image, big_endian=True):
    """Write a single-band image like one band of a SNAP BEAM-DIMAP product.

    `.img` = raw pixels (big-endian by default, as SNAP writes them),
    `.hdr` = ENVI text header. Only the fields read by `read_envi` matter;
    the others mimic a real SNAP header.
    """
    hdr_path = Path(hdr_path)
    hdr_path.parent.mkdir(parents=True, exist_ok=True)
    byte_order = 1 if big_endian else 0
    stored = image.astype(">f4" if big_endian else "<f4")
    stored.tofile(hdr_path.with_suffix(".img"))
    header_lines = [
        "ENVI",
        "description = {Sentinel-1 IW Level-1 GRD Product - Unit: intensity}",
        f"samples = {image.shape[1]}",
        f"lines = {image.shape[0]}",
        "bands = 1",
        "header offset = 0",
        "file type = ENVI Standard",
        "data type = 4",
        "interleave = bsq",
        f"byte order = {byte_order}",
        "band names = { Sigma0_VV }",
        "map info = {UTM,1.0,1.0,264556.0,8920425.0,10.0,10.0,20,South,units=Meters}",
    ]
    header = "\n".join(header_lines) + "\n"
    hdr_path.write_text(header, encoding="utf-8")


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
    3 ascending-like + 2 descending-like. The descending scene is written as
    a BEAM-DIMAP band (`.data/Sigma0_VV.hdr` + `.img`), the others as GeoTIFF.
    """
    root = tmp_path_factory.mktemp("scenes")
    rng = np.random.default_rng(0)
    for k in range(5):
        orbit = "asc" if k < 4 else "desc"
        # 3 x 4 windows of 256; the empty band removes the first column
        # (the last one once mirrored).
        scene = make_synthetic_scene(rng, height=800, width=1100, empty_columns=80)
        scene_dir = root / f"scene_{k}_{orbit}"
        if orbit == "desc":
            write_envi(scene_dir / f"scene_{k}.data" / "Sigma0_VV.hdr", scene)
        else:
            write_patch(scene_dir / f"scene_{k}.tif", scene)
    return root


@pytest.fixture(scope="session")
def dataset_dir(scenes_dir, tmp_path_factory):
    """A full small dataset: tiling + level-1 copy-move, 256x256 patches."""
    root = tmp_path_factory.mktemp("dataset")
    tile_scenes(scenes_dir, root, patch_size=TEST_PATCH_SIZE, patches_per_scene=6,
                split_counts={"val": 1, "test": 1}, seed=0)
    build_dataset(root, level=1, seed=0)
    return root
