"""Tests of features/optical (#47), on tiny synthetic Sentinel-2 products.

A fake `.SAFE` holds three small 16-bit JPEG 2000 bands, written with
Pillow like the real ones are read.
"""

import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from features.optical import optical
from features.sar_io.sar_io import read_manifest, read_tif, write_patch
from features.unet.unet import to_model_scale


def write_fake_safe(root, name, bands, granule="L2A_T00XXX_A000000_20200101T000000"):
    """Write B02 / B03 / B04 as `GRANULE/<granule>/IMG_DATA/R10m/*_<band>_10m.jp2`."""
    folder = root / name / f"{name}.SAFE" / "GRANULE" / granule / "IMG_DATA" / "R10m"
    folder.mkdir(parents=True)
    for band, values in zip(optical.BANDS, bands):
        Image.fromarray(values.astype(np.uint16)).save(folder / f"T00XXX_20200101_{band}_10m.jp2")
    return root / name / f"{name}.SAFE"


def fake_bands(rng, height, width, empty_columns=0):
    """Forest-like digital numbers (with the L2A offset of 1000), an empty band on the left."""
    bands = []
    for mean in (1200, 1400, 1300):
        values = rng.normal(mean, 60, size=(height, width)).clip(1001, None)
        values[:, :empty_columns] = 0
        bands.append(values)
    return bands


def fake_reference(rng):
    """A SAR-like reference: log-gamma values around -8 dB."""
    values = to_model_scale(rng.gamma(4.4, 0.16 / 4.4, size=100_000), "linear").astype(np.float64)
    levels = np.linspace(0, 1, optical.N_QUANTILES)
    return {"quantiles": np.quantile(values, levels).tolist(), "mean": float(values.mean()),
            "std": float(values.std()), "n_patches": 1, "source": "fake"}


def test_gray_is_the_float_mean_and_zeros_are_no_data(tmp_path):
    rng = np.random.default_rng(0)
    bands = fake_bands(rng, 40, 50, empty_columns=5)
    safe = write_fake_safe(tmp_path, "scene", bands)

    gray, valid = optical.s2_gray(safe)
    expected = np.mean([b.astype(np.uint16).astype(np.float32) for b in bands], axis=0)
    assert not valid[:, :5].any() and valid[:, 5:].all()
    assert np.allclose(gray[valid], expected[valid])
    assert (gray[~valid] == 0).all()


def test_histogram_matching_gives_the_sar_distribution_and_keeps_the_order():
    rng = np.random.default_rng(1)
    reference = fake_reference(rng)
    gray = rng.uniform(1000, 3000, size=(200, 200)).astype(np.float32)   # flat, not SAR-like
    valid = np.ones(gray.shape, dtype=bool)

    values_db = optical.to_sar_db(gray, valid, reference, "histogram")
    levels = np.linspace(0, 1, optical.N_QUANTILES)
    for q in (0.1, 0.5, 0.9):
        assert abs(np.quantile(values_db, q) - np.interp(q, levels, reference["quantiles"])) < 0.05
    # Increasing map: a brighter grey pixel never gets a darker value.
    order = np.argsort(gray.ravel())
    assert (np.diff(values_db.ravel()[order]) >= -1e-5).all()


def test_affine_matches_mean_and_std_only():
    rng = np.random.default_rng(2)
    reference = fake_reference(rng)
    gray = rng.uniform(1000, 3000, size=(100, 100)).astype(np.float32)
    values_db = optical.to_sar_db(gray, np.ones(gray.shape, dtype=bool), reference, "affine")
    assert abs(values_db.mean() - reference["mean"]) < 1e-3
    assert abs(values_db.std() - reference["std"]) < 1e-3


def test_speckle_has_the_sentinel1_enl():
    rng = np.random.default_rng(3)
    gray = np.full((300, 300), 50.0, dtype=np.float32)
    speckled = optical.add_speckle(gray, np.ones(gray.shape, dtype=bool), rng)
    enl = speckled.mean() ** 2 / speckled.var()        # ENL = mean^2 / variance of intensity
    assert abs(enl - optical.ENL) < 0.2
    assert abs(speckled.mean() - 50.0) < 0.5           # speckle of mean 1


def test_sar_reference_reads_train_authentic_patches_from_the_zip(tmp_path):
    rng = np.random.default_rng(4)
    rows = []
    for k, split in enumerate(["train", "train", "test"]):
        patch = rng.gamma(4.4, 0.1 / 4.4, size=(32, 32)).astype(np.float32)
        if split == "test":
            patch *= 1000.0          # must not be read
        write_patch(tmp_path / "data" / "real" / f"{k:06d}.tif", patch)
        rows.append({"sample_id": f"{k:06d}_real", "label": 0, "file": f"real/{k:06d}.tif", "split": split})
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "manifest.csv", index=False)
    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in (tmp_path / "data").rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(tmp_path / "data").as_posix())

    reference = optical.sar_reference(zip_path)
    assert reference["n_patches"] == 2
    assert -20 < reference["mean"] < -5            # ~ -13 dB, the test patches (+30 dB) excluded
    assert len(reference["quantiles"]) == optical.N_QUANTILES


def test_whole_pipeline_gives_test_only_datasets(tmp_path):
    rng = np.random.default_rng(5)
    safe_dirs = {}
    for name in ["ds_forest_1", "ds_forest_9"]:
        safe_dirs[name] = write_fake_safe(tmp_path / "s2", name, fake_bands(rng, 600, 800, empty_columns=40))

    datasets = optical.build_optical_datasets(safe_dirs, fake_reference(rng), tmp_path / "work",
                                              levels=(1,), patch_size=256, patches_per_scene=3)
    assert set(datasets) == {("_optical", 1), ("_optical_speckle", 1)}
    for dataset_dir in datasets.values():
        manifest = read_manifest(dataset_dir / "manifest.csv")
        assert set(manifest["split"]) == {"test"}
        assert set(manifest["scene"]) == {"ds_forest_1", "ds_forest_9"}
        assert not manifest["mirrored"].any()
        patch = read_tif(dataset_dir / manifest.iloc[0]["file"])
        assert patch.shape == (256, 256) and (patch > 0).mean() > 0.95
