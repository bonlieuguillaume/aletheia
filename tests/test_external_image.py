"""Tests of features/external_image (#11), on CPU with a stand-in model.

The stand-in returns its normalized input as logits, so the probability of
a pixel only depends on that pixel: the stitched map can be checked exactly.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from features.external_image import external_image
from features.unet.unet import to_model_scale

CONFIG = {
    "input_scale": "linear",
    "normalization_mean": -12.0,   # dB, like Sentinel-1 VV over forest
    "normalization_std": 3.0,
    "score_method": "max",
    "image_threshold": 0.9,
    "pixel_threshold": 0.5,
}


class IdentityLogits(nn.Module):
    """Logits = normalized input; one parameter so that the device can be read."""

    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x


def test_colour_image_becomes_the_float_mean_of_its_channels(tmp_path):
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    rgb[..., 0] = 10
    rgb[..., 1] = 20
    rgb[..., 2] = 31            # mean 20.333: must not be rounded
    path = tmp_path / "image.png"
    Image.fromarray(rgb).save(path)

    gray = external_image.read_image_gray(path)
    assert gray.shape == (10, 12)
    assert np.allclose(gray, 61 / 3)


def test_pseudo_sar_has_the_training_statistics_in_db():
    rng = np.random.default_rng(0)
    gray = rng.uniform(0, 255, size=(64, 64))
    raw = external_image.to_pseudo_sar(gray, CONFIG)
    in_db = to_model_scale(raw, "linear")
    assert abs(in_db.mean() - CONFIG["normalization_mean"]) < 1e-3
    assert abs(in_db.std() - CONFIG["normalization_std"]) < 1e-3


def test_windows_cover_the_whole_axis():
    assert external_image.window_starts(1000, 512, 256) == [0, 256, 488]
    assert external_image.window_starts(512, 512, 256) == [0]
    assert external_image.window_starts(300, 512, 256) == [0]


def test_stitched_map_matches_every_pixel_and_keeps_the_image_size():
    rng = np.random.default_rng(1)
    model = IdentityLogits()
    for shape in [(100, 150), (40, 30)]:          # larger and smaller than a window
        raw = rng.gamma(4.4, 0.05 / 4.4, size=shape).astype(np.float32)
        probability_map, windows = external_image.predict_large_image(model, CONFIG, raw, tile=64, stride=32)

        normalized = (to_model_scale(raw, "linear") - CONFIG["normalization_mean"]) / CONFIG["normalization_std"]
        expected = 1.0 / (1.0 + np.exp(-normalized))
        assert probability_map.shape == shape
        assert np.allclose(probability_map, expected, atol=1e-5)
        assert len(windows) >= 1


def test_analyse_image_decides_on_the_highest_window(tmp_path):
    # Dark image with one bright square: after the matching, the square is
    # far above the mean, its probability is close to 1 with the stand-in.
    gray = np.full((128, 160), 50, dtype=np.uint8)
    gray[70:80, 100:110] = 250
    path = tmp_path / "image.png"
    Image.fromarray(gray).save(path)

    result = external_image.analyse_image(IdentityLogits(), CONFIG, path, tile=64, stride=32)
    assert result["is_forged"]
    assert result["score"] == max(window["score"] for window in result["windows"])
    assert result["zone"][75, 105] and not result["zone"][10, 10]

    fig = external_image.plot_result(result, "test")
    assert len(fig.axes) == 4      # three panels + the colour bar
