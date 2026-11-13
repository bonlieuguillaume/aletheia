"""Tests of features/evaluation (issues #6-#8, #32)."""

import numpy as np

from features.evaluation import evaluation as ev
from features.sar_io.sar_io import read_manifest, read_tif


def test_mcc_and_f1_on_the_worked_example():
    # Example of docs/architecture_v0.1.md: 370 TP, 20 FP, 30 FN, 380 TN.
    assert abs(ev.mcc(370, 20, 30, 380) - 0.8752) < 1e-3
    assert abs(ev.f1(370, 20, 30) - 740 / 790) < 1e-12


def test_flagging_nothing_scores_zero():
    # 97 % accuracy, but useless: MCC 0, F1 0.
    assert ev.mcc(0, 0, 300, 9700) == 0.0
    assert ev.f1(0, 0, 300) == 0.0


def test_source_pixels_are_ignored():
    mask = np.zeros((64, 64), np.uint8)
    mask[10:20, 10:20] = 1  # pasted
    mask[40:50, 40:50] = 2  # source
    flags_source_too = (mask > 0).astype(float)
    stats = ev.sample_statistics(flags_source_too, mask, label=1, margin=4)
    index = int(np.argmin(np.abs(ev.PIXEL_THRESHOLDS - 0.5)))
    tp, fp, fn, tn = stats["pixel_counts"][index]
    assert (tp, fp, fn) == (100, 0, 0)


def test_image_score_methods():
    probability_map = np.zeros((100, 100))
    probability_map[0, 0] = 1.0  # one noisy pixel
    assert ev.image_score(probability_map, "max") == 1.0
    assert ev.image_score(probability_map, "top1%") == 0.01  # 1 pixel among the top 100
    assert ev.image_score(probability_map, "p99.5") < 0.5


def test_full_procedure_with_a_perfect_model(dataset_dir):
    manifest = read_manifest(dataset_dir / "manifest.csv")

    def perfect_predictions(split):
        for _, row in manifest[manifest["split"] == split].iterrows():
            mask = read_tif(dataset_dir / row["mask_file"]) if row["label"] == 1 else None
            if mask is None:
                shape = read_tif(dataset_dir / row["file"]).shape
                probability_map = np.full(shape, 0.02)
            else:
                probability_map = np.where(mask == 1, 0.95, 0.02)
            yield row, probability_map, mask

    settings = ev.choose_thresholds(ev.collect_statistics(perfect_predictions("val")))
    summary, table = ev.compute_metrics(ev.collect_statistics(perfect_predictions("test")), settings, manifest)
    assert summary["image_mcc"] == 1.0
    assert summary["pixel_mcc_mean"] == 1.0
    assert summary["false_alarm_mean"] == 0.0
    assert "scene" in table.columns  # joined with the manifest


def test_image_threshold_is_the_middle_of_the_best_interval():
    # Authentic scores up to 0.3, forged from 0.8: every threshold in
    # (0.3, 0.8] is perfect. The middle, 0.55, is kept, not the edge 0.8.
    scores = [0.1, 0.2, 0.3, 0.8, 0.9]
    labels = [0, 0, 0, 1, 1]
    threshold, value, margin = ev.best_image_threshold(scores, labels)
    assert value == 1.0
    assert abs(threshold - 0.55) < 1e-9
    assert abs(margin - 0.5) < 1e-9


def test_tied_score_methods_are_split_by_their_margin():
    # Every method separates the classes perfectly, but only by 0.005, except
    # top2% (by 0.5): at equal MCC, the widest margin wins, not the first method.
    stats = []
    for label, narrow, wide in [(0, 0.980, 0.2), (0, 0.985, 0.3), (1, 0.990, 0.8), (1, 0.995, 0.9)]:
        sample = {"label": label, "pixel_counts": [(10, 0, 0, 90)] * len(ev.PIXEL_THRESHOLDS)}
        for method in ev.SCORE_METHODS:
            sample[f"score_{method}"] = narrow
        sample["score_top2%"] = wide
        stats.append(sample)

    settings = ev.choose_thresholds(stats)
    assert settings["score_method"] == "top2%"
    assert abs(settings["image_threshold"] - 0.55) < 1e-9


def test_score_methods_are_described_in_words():
    for method in ev.SCORE_METHODS:
        assert ev.describe_score_method(method).startswith("score = ")
    assert "0.5 % highest" in ev.describe_score_method("top0.5%")
