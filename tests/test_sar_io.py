"""Tests of features/sar_io: reading the scenes (GeoTIFF or BEAM-DIMAP)."""

import numpy as np
import pytest

from conftest import write_envi
from features.sar_io.sar_io import read_scene, write_patch
from features.tiling.tiling import find_scenes


def test_beam_dimap_band_is_read_in_native_byte_order(tmp_path):
    image = np.arange(12, dtype=np.float32).reshape(3, 4) / 7.0
    for big_endian in (True, False):
        hdr_path = tmp_path / f"big_{big_endian}" / "Sigma0_VV.hdr"
        write_envi(hdr_path, image, big_endian=big_endian)

        result = read_scene(hdr_path)
        assert result.shape == (3, 4)          # (lines, samples)
        assert result.dtype == np.float32
        assert result.dtype.isnative           # bytes swapped after reading
        assert np.array_equal(result, image)


def test_truncated_img_file_is_refused(tmp_path):
    hdr_path = tmp_path / "Sigma0_VV.hdr"
    write_envi(hdr_path, np.ones((3, 4), dtype=np.float32))
    img_path = hdr_path.with_suffix(".img")
    img_path.write_bytes(img_path.read_bytes()[:-4])  # one pixel missing
    with pytest.raises(ValueError):
        read_scene(hdr_path)


def test_find_scenes_accepts_both_formats_but_not_both_at_once(tmp_path):
    image = np.ones((8, 8), dtype=np.float32)
    write_patch(tmp_path / "a_asc" / "a.tif", image)
    write_envi(tmp_path / "b_desc" / "b.data" / "Sigma0_VV.hdr", image)
    scenes = find_scenes(tmp_path)
    assert scenes["a_asc"].suffix == ".tif"
    assert scenes["b_desc"].suffix == ".hdr"

    # A GeoTIFF and a BEAM-DIMAP of the same scene: ambiguous, refused.
    write_patch(tmp_path / "b_desc" / "b.tif", image)
    with pytest.raises(ValueError):
        find_scenes(tmp_path)
