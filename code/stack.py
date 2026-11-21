import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ARTIFACTS, LABELS, LABEL2ID, RESULTS, SEED, load_test, macro_f1, split_frames
from features import stylometric_features

N_SPLITS = 5
INNER_SPLITS = 3
CANDIDATES = [
    ("logreg_c0.1", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))),
    ("logreg_c1", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))),
    ("hgb", lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=SEED)),
]


def cached_style(name, texts):
    path = ARTIFACTS / f"style_{name}.npy"
    if path.exists():
        return np.load(path)
    values = stylometric_features(texts).values
    np.save(path, values)
    return values


def blocks(sources):
    _, val = split_frames()
    test = load_test()
    val_parts, test_parts = [], []
    for source in sources:
        if source == "style":
            val_parts.append(cached_style("val", val.text))
            test_parts.append(cached_style("test", test.text))
        else:
            val_parts.append(np.load(ARTIFACTS / f"{source}_val.npy"))
            test_parts.append(np.load(ARTIFACTS / f"{source}_test.npy"))
    return val, np.hstack(val_parts), np.hstack(test_parts)


def score(y_true, probs, weights):
    pred = np.array(LABELS)[probs.argmax(axis=1)]
    return macro_f1(y_true, pred), macro_f1(y_true, pred, weights)


def fit_best(X, y, weights, y_str):
    inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=SEED)
    best = None
    for name, factory in CANDIDATES:
        oof = np.zeros((len(y), 3))
        for tr, te in inner.split(X, y):
            oof[te] = factory().fit(X[tr], y[tr]).predict_proba(X[te])
        value = score(y_str, oof, weights)[1]
        if best is None or value > best[0]:
            best = (value, name, factory)
    return best[1], best[2]().fit(X, y)


def nested_oof(X, y, y_str, weights):
    outer = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), 3))
    chosen = []
    for tr, te in outer.split(X, y):
        name, model = fit_best(X[tr], y[tr], weights[tr], y_str[tr])
        oof[te] = model.predict_proba(X[te])
        chosen.append(name)
    return oof, chosen


def tune_class_weights(probs, y_str, weights):
    w = np.ones(3)
    grid = np.linspace(0.5, 2.0, 31)
    for _ in range(3):
        for k in range(3):
            best_value, best_w = -1.0, w[k]
            for candidate in grid:
                trial = w.copy()
                trial[k] = candidate
                value = score(y_str, probs * trial, weights)[1]
                if value > best_value:
                    best_value, best_w = value, candidate
            w[k] = best_w
    return w / w.sum() * 3


def nested_threshold(oof, y_str, weights):
    folds = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED + 1)
    adjusted = np.zeros_like(oof)
    for tr, te in folds.split(oof, y_str):
        w = tune_class_weights(oof[tr], y_str[tr], weights[tr])
        adjusted[te] = oof[te] * w
    return adjusted


def main():
    sources = sys.argv[1].split(",") if len(sys.argv) > 1 else ["ref_head", "baseline", "style"]
    tag = sys.argv[2] if len(sys.argv) > 2 else "stack"
    reference = sys.argv[3] if len(sys.argv) > 3 else "ref_head"
    val, X, X_test = blocks(sources)
    y_str = val.label.values
    y = val.label.map(LABEL2ID).values
    weights = val.weight.values
    ref_scores = np.load(ARTIFACTS / f"{reference}_val.npy")
    ref_macro, ref_weighted = score(y_str, ref_scores, weights)
    oof, chosen = nested_oof(X, y, y_str, weights)
    macro, weighted = score(y_str, oof, weights)
    name, final_model = fit_best(X, y, weights, y_str)
    test_probs = final_model.predict_proba(X_test)
    np.save(ARTIFACTS / f"{tag}_oof.npy", oof)
    np.save(ARTIFACTS / f"{tag}_test.npy", test_probs)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(
        json.dumps(
            {
                "sources": sources,
                "reference": reference,
                "reference_macro_f1": ref_macro,
                "reference_weighted_macro_f1": ref_weighted,
                "nested_cv_macro_f1": macro,
                "nested_cv_weighted_macro_f1": weighted,
                "n_splits": N_SPLITS,
                "inner_splits": INNER_SPLITS,
                "chosen_per_outer_fold": chosen,
                "final_model": name,
            },
            indent=2,
        )
    )
    w = tune_class_weights(oof, y_str, weights)
    adjusted = nested_threshold(oof, y_str, weights)
    after_macro, after_weighted = score(y_str, adjusted, weights)
    np.save(ARTIFACTS / f"{tag}_thresholded_test.npy", test_probs * w)
    (RESULTS / f"{tag}_thresholds.json").write_text(
        json.dumps(
            {
                "before": {"macro_f1": macro, "weighted_macro_f1": weighted},
                "after": {"macro_f1": after_macro, "weighted_macro_f1": after_weighted},
                "class_weights": dict(zip(LABELS, w.tolist())),
                "selected_by": "cv_on_val",
            },
            indent=2,
        )
    )
    print(tag, sources, "ref", round(ref_weighted, 4), "stack", round(weighted, 4), "thresholded", round(after_weighted, 4))


if __name__ == "__main__":
    main()
