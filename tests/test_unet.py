"""Tests of features/unet (issues #9, #10, #31), on CPU with tiny data.

The encoder is not pretrained here (no download in the tests) and is a
ResNet-18 to stay fast; the code path is the same as on Colab.
"""

import numpy as np
import torch
import torch.nn as nn

from features.unet import unet

TEST_CONFIG = {
    "encoder": "resnet18",
    "pretrained": False,
    "batch_size": 4,
    "epochs": 2,
    "num_workers": 0,
    "mixed_precision": False,
}


def test_output_has_the_input_size_and_padding_is_reflect():
    config = {**unet.DEFAULT_CONFIG, **TEST_CONFIG}
    model = unet.build_model(config)
    padded = [m for m in model.modules() if isinstance(m, nn.Conv2d) and any(p > 0 for p in m.padding)]
    assert padded and all(m.padding_mode == "reflect" for m in padded)

    with torch.no_grad():
        output = model(torch.randn(2, 1, 64, 64))
    assert output.shape == (2, 1, 64, 64)


def test_loss_ignores_source_pixels():
    target = torch.zeros(1, 1, 8, 8)
    target[..., :2, :2] = 1
    valid = torch.ones(1, 1, 8, 8)
    valid[..., 6:, 6:] = 0  # source pixels

    good = torch.where(target > 0, 10.0, -10.0)
    flags_source = good.clone()
    flags_source[..., 6:, 6:] = 10.0  # also flags the (ignored) source
    assert torch.isclose(unet.loss_function(good, target, valid),
                         unet.loss_function(flags_source, target, valid), atol=1e-6)


def test_augmentation_is_an_east_west_mirror_only(dataset_dir, monkeypatch):
    from features.sar_io.sar_io import read_manifest

    manifest = read_manifest(dataset_dir / "manifest.csv")
    dataset = unet.SARPatchDataset(dataset_dir, manifest, "train", 0.0, 1.0, "linear", augment=True)
    forged_index = int(dataset.rows.index[dataset.rows["label"] == 1][0])

    # Force the random draw so that the mirror is applied.
    monkeypatch.setattr(unet.torch, "rand", lambda *args, **kwargs: torch.zeros(1))
    image, target, valid = dataset[forged_index]
    _, reference, mask = dataset.load(forged_index)

    assert np.array_equal(image[0].numpy(), reference[:, ::-1])          # columns reversed
    assert np.array_equal(target[0].numpy(), (mask == 1)[:, ::-1])       # mask follows
    assert np.array_equal(valid[0].numpy(), (mask != 2)[:, ::-1])


def test_dB_conversion_handles_zeros():
    values = unet.to_model_scale(np.array([0.0, 1.0, np.nan], dtype=np.float32), "linear")
    assert np.all(np.isfinite(values))
    assert values[1] == 0.0  # 1 in linear scale = 0 dB


def test_train_resume_and_predict(dataset_dir, tmp_path):
    run_dir = tmp_path / "run"
    unet.train(dataset_dir, run_dir, TEST_CONFIG)
    assert (run_dir / "best.pt").exists() and (run_dir / "log.csv").exists()

    # A second call resumes from last.pt and has nothing left to do.
    unet.train(dataset_dir, run_dir, TEST_CONFIG)
    assert len((run_dir / "log.csv").read_text().strip().splitlines()) == 1 + 2

    model, config = unet.load_model(run_dir)
    patch = np.random.default_rng(0).gamma(4.4, 0.01, size=(64, 64)).astype(np.float32)
    score, probability_map = unet.predict(model, config, patch)
    assert probability_map.shape == (64, 64)
    assert 0.0 <= score <= 1.0

    predictions = list(unet.iterate_predictions(model, config, dataset_dir, "test"))
    assert len(predictions) == 12  # 6 patches x (authentic + forged)
