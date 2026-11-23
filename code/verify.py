import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import common

SUBMISSION_PATH = common.ROOT / "code" / "artifacts" / "submission_check.csv"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def a1_split():
    split = common.load_split()
    train = common.load_train()
    assert len(split) == len(train) and split.id.is_unique
    assert set(split.id) == set(train.id)
    counts = split.part.value_counts()
    assert abs(counts["val"] / len(split) - 0.15) < 0.002, counts
    assert 59000 <= counts["train"] <= 61000, counts
    assert set(split.part) == {"train", "val", "pool"}
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, str(common.ROOT / "code" / "make_split.py"), "again.csv"],
            check=True,
            capture_output=True,
            env={**__import__("os").environ, "ARTIFACTS_DIR": tmp},
        )
        assert digest(Path(tmp) / "again.csv") == digest(common.ARTIFACTS / "split.csv")


def a1_weights():
    split = common.load_split()
    meta = common.load_train()[["id", "domain", "lang"]].merge(split, on="id")
    val = meta[meta.part == "val"]
    test = common.load_test()
    target = (test.domain + "|" + test.lang).value_counts(normalize=True)
    cell = val.domain + "|" + val.lang
    weighted = val.groupby(cell).weight.sum()
    weighted = weighted / weighted.sum()
    gap = (weighted - target).abs().max()
    assert gap < 0.005, gap
    assert (meta[meta.part != "val"].weight == 0).all()


def a2_eda():
    needed = [
        common.RESULTS / "eda_class_balance.csv",
        common.RESULTS / "eda_length_stats.csv",
        common.RESULTS / "eda_examples.csv",
        common.IMAGES / "eda_class_balance.png",
        common.IMAGES / "eda_lengths.png",
    ]
    for path in needed:
        assert path.exists() and path.stat().st_size > 0, path
    examples = pd.read_csv(common.RESULTS / "eda_examples.csv")
    assert set(examples.label) == set(common.LABELS)


def read_metrics(name):
    return json.loads((common.RESULTS / name).read_text())


def b1_baseline():
    metrics = read_metrics("baseline.json")
    for key in ["macro_f1", "weighted_macro_f1", "per_class_f1", "confusion_matrix", "fit_on", "best_params"]:
        assert key in metrics, key
    assert metrics["fit_on"] == "train"
    assert set(metrics["per_class_f1"]) == set(common.LABELS)
    assert 0.3 < metrics["macro_f1"] <= 1.0
    assert (common.RESULTS / "baseline_confusion.csv").exists()


def b2_features():
    from features import stylometric_features

    train, val = common.split_frames()
    a = stylometric_features(train.text.head(200))
    b = stylometric_features(val.text.head(200))
    assert list(a.columns) == list(b.columns)
    assert not a.isna().any().any() and not b.isna().any().any()
    assert a.shape[1] >= 10


def f1_submission():
    sample = common.load_sample_submission()
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else SUBMISSION_PATH
    sub = pd.read_csv(path)
    assert list(sub.columns) == ["id", "label"]
    assert len(sub) == len(sample) == 25000
    assert sub.id.is_unique
    assert (sub.id.values == sample.id.values).all()
    assert set(sub.label) <= set(common.LABELS)


def logits_gate(prefix, check_seed=True):
    metrics = read_metrics(prefix + ".json")
    split = common.load_split()
    counts = split.part.value_counts()
    val = np.load(common.ARTIFACTS / (prefix + "_val.npy"))
    test = np.load(common.ARTIFACTS / (prefix + "_test.npy"))
    assert val.shape == (counts["val"], 3), val.shape
    assert test.shape == (25000, 3), test.shape
    assert np.isfinite(val).all() and np.isfinite(test).all()
    for key in ["macro_f1", "weighted_macro_f1", "model", "seed", "train_size", "max_length"]:
        assert key in metrics, key
    if check_seed:
        assert metrics["seed"] == common.SEED
    return metrics


def c1_smoke():
    metrics = read_metrics("smoke.json")
    assert metrics["gpu"] and metrics["samples_per_sec"] > 0
    assert metrics["internet"] is True


def c2_ref():
    logits_gate("ref_head")


def c3_chunks():
    ref = logits_gate("ref_head")
    chunks = logits_gate("chunks")
    for key in ["model", "seed", "train_size", "max_length"]:
        assert ref[key] == chunks[key], key
    assert chunks["mode"] == "chunks" and ref["mode"] == "head"


def d1_stack():
    metrics = read_metrics("stack.json")
    for key in ["nested_cv_macro_f1", "reference_macro_f1", "n_splits"]:
        assert key in metrics, key
    assert metrics["n_splits"] >= 5


def d2_thresholds():
    metrics = read_metrics("stack_thresholds.json")
    for key in ["before", "after", "class_weights", "selected_by"]:
        assert key in metrics, key
    assert metrics["selected_by"] == "cv_on_val"


def e1_explain():
    for name in ["explain_linear_top_tokens.csv", "explain_transformer_attributions.csv", "explain_summary.md"]:
        path = common.RESULTS / name
        assert path.exists() and path.stat().st_size > 0, name
    assert any(common.IMAGES.glob("xai_*.png"))


def g1_final():
    final = (common.RESULTS / "final.md").read_text()
    for token in ["candidate_1", "candidate_2", "reproduced"]:
        assert token in final, token


CHECKS = {
    "a1_split": a1_split,
    "a1_weights": a1_weights,
    "a2_eda": a2_eda,
    "b1_baseline": b1_baseline,
    "b2_features": b2_features,
    "f1_submission": f1_submission,
    "c1_smoke": c1_smoke,
    "c2_ref": c2_ref,
    "c3_chunks": c3_chunks,
    "d1_stack": d1_stack,
    "d2_thresholds": d2_thresholds,
    "e1_explain": e1_explain,
    "g1_final": g1_final,
}


LEGACY = [
    "a1_split",
    "a1_weights",
    "a2_eda",
    "b1_baseline",
    "b2_features",
    "f1_submission",
    "c2_ref",
    "d1_stack",
    "d2_thresholds",
]


def legacy():
    for name in LEGACY:
        CHECKS[name]()


CHECKS["legacy"] = legacy


def load_extra_checks():
    folder = Path(__file__).resolve().parent / "checks"
    for path in sorted(folder.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"checks_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        CHECKS.update(module.CHECKS)


def main():
    name = sys.argv[1]
    load_extra_checks()
    CHECKS[name]()
    print(name, "verification passed")


if __name__ == "__main__":
    main()
