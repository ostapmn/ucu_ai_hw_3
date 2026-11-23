import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ARTIFACTS, LABELS, LABEL2ID, RESULTS, SEED, load_test, macro_f1, split_frames
from features import drift_features, stylometric_features

INNER_SPLITS = 3
CANDIDATES = [
    ("logreg_c1", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))),
    ("hgb", lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=SEED)),
    ("hgb_slow", lambda: HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, l2_regularization=1.0, min_samples_leaf=40, max_leaf_nodes=24, early_stopping=False, random_state=SEED)),
]
EXPERIMENTS = {
    "v1_regression": ["ref_head", "baseline", "style"],
    "pool_views": ["ref_head", "baseline", "style", "bp_full", "bp_head", "bp_tail"],
    "pool_views_drift": ["ref_head", "baseline", "style", "bp_full", "bp_head", "bp_tail", "drift"],
    "headtail_core": ["headtail_a", "style", "drift", "bp_full", "bp_head", "bp_tail"],
    "headtail_all": ["headtail_a", "ref_head", "baseline", "style", "drift", "bp_full", "bp_head", "bp_tail"],
    "headtail_ab_core": ["headtail_a", "headtail_b", "style", "drift", "bp_full", "bp_head", "bp_tail"],
    "headtail_ab_all": ["headtail_a", "headtail_b", "ref_head", "baseline", "style", "drift", "bp_full", "bp_head", "bp_tail"],
    "headtail_ab_only": ["headtail_a", "headtail_b"],
    "headtail_abcd_only": ["headtail_a", "headtail_b", "headtail_c", "headtail_d"],
    "headtail_abcd_core": ["headtail_a", "headtail_b", "headtail_c", "headtail_d", "style", "drift", "bp_full", "bp_head", "bp_tail"],
    "headtail_abcdf_all": ["headtail_a", "headtail_b", "headtail_c", "headtail_d", "headtail_f", "ref_head", "baseline", "style", "drift", "bp_full", "bp_head", "bp_tail"],
    "headtail_all6": ["headtail_a", "headtail_b", "headtail_c", "headtail_d", "headtail_e", "headtail_f", "ref_head", "baseline", "style", "drift", "bp_full", "bp_head", "bp_tail"],
    "headtail_abcd_win": ["headtail_a", "headtail_b", "headtail_c", "headtail_d", "ref_head", "baseline", "style", "drift", "bp_full", "bp_head", "bp_tail", "winscan_a"],
    "headtail_all6_win": ["headtail_a", "headtail_b", "headtail_c", "headtail_d", "headtail_e", "headtail_f", "ref_head", "baseline", "style", "drift", "bp_full", "bp_head", "bp_tail", "winscan_a"],
    "headtail_abcd_win2": ["headtail_a", "headtail_b", "headtail_c", "headtail_d", "ref_head", "baseline", "style", "drift", "bp_full", "bp_head", "bp_tail", "winscan_a", "winscan_g"],
    "headtail_abcdg_win2": ["headtail_a", "headtail_b", "headtail_c", "headtail_d", "headtail_g", "ref_head", "baseline", "style", "drift", "bp_full", "bp_head", "bp_tail", "winscan_a", "winscan_g"],
    "headtail_abcdg_win3": ["headtail_a", "headtail_b", "headtail_c", "headtail_d", "headtail_g", "ref_head", "baseline", "style", "drift", "bp_full", "bp_head", "bp_tail", "winscan_a", "winscan_g"],
    "headtail_abcdg_win3_meta": ["headtail_a", "headtail_b", "headtail_c", "headtail_d", "headtail_g", "ref_head", "baseline", "style", "drift", "bp_full", "bp_head", "bp_tail", "winscan_a", "winscan_g", "meta"],
    "headtail_abcdg_win3_blog": ["headtail_a", "headtail_b", "headtail_c", "headtail_d", "headtail_g", "ref_head", "baseline", "style", "drift", "bp_full", "bp_head", "bp_tail", "winscan_a", "winscan_g", "blog_spec"],
    "headtail_abcd_all": ["headtail_a", "headtail_b", "headtail_c", "headtail_d", "ref_head", "baseline", "style", "drift", "bp_full", "bp_head", "bp_tail"],
}
CACHED = {"style": stylometric_features, "drift": drift_features}


def cached(name, split, texts):
    path = ARTIFACTS / f"{name}_{split}.npy"
    if path.exists():
        return np.load(path)
    values = CACHED[name](texts).values
    np.save(path, values)
    return values


DOMAINS = ["arxiv_papers", "blog_authorship_corpus", "bundestag", "cnn_news", "euro_court_cases", "gutenberg", "house_of_commons", "student_essays"]


def meta_features(frame):
    columns = {f"domain_{d}": (frame.domain == d).astype(np.float32) for d in DOMAINS}
    columns["lang_en"] = (frame.lang == "en-EN").astype(np.float32)
    return np.column_stack(list(columns.values()))


def blocks(sources, val):
    test = load_test()
    val_parts, test_parts = [], []
    for source in sources:
        if source == "meta":
            val_parts.append(meta_features(val))
            test_parts.append(meta_features(test))
        elif source in CACHED:
            val_parts.append(cached(source, "val", val.text))
            test_parts.append(cached(source, "test", test.text))
        else:
            val_parts.append(np.load(ARTIFACTS / f"{source}_val.npy"))
            test_parts.append(np.load(ARTIFACTS / f"{source}_test.npy"))
    return np.hstack(val_parts), np.hstack(test_parts)


def score(y_str, probs, weights):
    pred = np.array(LABELS)[probs.argmax(axis=1)]
    return macro_f1(y_str, pred), macro_f1(y_str, pred, weights)


SLOW_TAGS = {"headtail_abcdg_win3", "headtail_abcdg_win3_meta", "headtail_abcdg_win3_blog"}


def fit_best(X, y, y_str, weights, groups, candidates):
    inner = StratifiedGroupKFold(INNER_SPLITS, shuffle=True, random_state=SEED)
    best = None
    for name, factory in candidates:
        oof = np.zeros((len(y), 3))
        for tr, te in inner.split(X, y, groups):
            oof[te] = factory().fit(X[tr], y[tr]).predict_proba(X[te])
        value = score(y_str, oof, weights)[1]
        if best is None or value > best[0]:
            best = (value, name, factory)
    return best[1], best[2]().fit(X, y)


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


def run(tag, sources):
    candidates = CANDIDATES if tag in SLOW_TAGS else [c for c in CANDIDATES if c[0] != "hgb_slow"]
    _, val = split_frames()
    X, X_test = blocks(sources, val)
    y_str = val.label.values
    y = val.label.map(LABEL2ID).values
    weights = val.weight.values
    groups = val.key.values
    dev = (val.lock == 0).values
    lock = ~dev
    folds = val.fold.values
    oof = np.zeros((len(val), 3))
    chosen = []
    for fold in range(5):
        tr, te = dev & (folds != fold), dev & (folds == fold)
        assert not set(groups[tr]) & set(groups[te])
        name, model = fit_best(X[tr], y[tr], y_str[tr], weights[tr], groups[tr], candidates)
        oof[te] = model.predict_proba(X[te])
        chosen.append(name)
    dev_macro, dev_weighted = score(y_str[dev], oof[dev], weights[dev])
    lock_name, lock_model = fit_best(X[dev], y[dev], y_str[dev], weights[dev], groups[dev], candidates)
    lock_probs = lock_model.predict_proba(X[lock])
    lock_macro, lock_weighted = score(y_str[lock], lock_probs, weights[lock])
    class_weights = tune_class_weights(oof[dev], y_str[dev], weights[dev])
    adjusted = np.zeros_like(oof)
    for fold in range(5):
        tr, te = dev & (folds != fold), dev & (folds == fold)
        adjusted[te] = oof[te] * tune_class_weights(oof[tr], y_str[tr], weights[tr])
    thr_dev = score(y_str[dev], adjusted[dev], weights[dev])
    thr_lock = score(y_str[lock], lock_probs * class_weights, weights[lock])
    final_name, final_model = fit_best(X, y, y_str, weights, groups, candidates)
    test_probs = final_model.predict_proba(X_test)
    np.save(ARTIFACTS / f"stack_v2_{tag}_oof.npy", oof)
    np.save(ARTIFACTS / f"stack_v2_{tag}_lock.npy", lock_probs)
    np.save(ARTIFACTS / f"stack_v2_{tag}_test.npy", test_probs)
    return {
        "tag": tag,
        "sources": sources,
        "n_dev": int(dev.sum()),
        "n_lock": int(lock.sum()),
        "dev_cv_macro_f1": dev_macro,
        "dev_cv_weighted_macro_f1": dev_weighted,
        "lock_macro_f1": lock_macro,
        "lock_weighted_macro_f1": lock_weighted,
        "thresholded_dev_cv_weighted_macro_f1": thr_dev[1],
        "thresholded_lock_weighted_macro_f1": thr_lock[1],
        "class_weights": dict(zip(LABELS, class_weights.tolist())),
        "chosen_per_outer_fold": chosen,
        "final_model": final_name,
        "folds_disjoint_by_key": True,
    }


def main():
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(EXPERIMENTS)
    out = RESULTS / "stack_v2.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    for tag in names:
        row = run(tag, EXPERIMENTS[tag])
        results[tag] = row
        out.write_text(json.dumps(results, indent=2))
        print(tag, "dev", round(row["dev_cv_weighted_macro_f1"], 4), "lock", round(row["lock_weighted_macro_f1"], 4), flush=True)


if __name__ == "__main__":
    main()
