import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from common import ARTIFACTS, LABELS, LABEL2ID, RESULTS, SEED, load_split, load_test, load_train, macro_f1

CHARS = 1500
VIEWS = {
    "full": lambda s: s,
    "head": lambda s: s.str[:CHARS],
    "tail": lambda s: s.str[-CHARS:],
}
HASHER = HashingVectorizer(ngram_range=(1, 2), n_features=2**21, alternate_sign=False, norm=None, dtype=np.float32)


def hashed(texts, jobs=8):
    texts = list(texts)
    chunks = [texts[i : i + 5000] for i in range(0, len(texts), 5000)]
    parts = Parallel(n_jobs=jobs)(delayed(HASHER.transform)(chunk) for chunk in chunks)
    return sp.vstack(parts).tocsr()


def main():
    split = load_split()
    train = load_train().merge(split, on="id", validate="one_to_one")
    val_keys = set(train.key[train.part == "val"])
    fit = train[(train.part != "val") & ~train.key.isin(val_keys)].reset_index(drop=True)
    val = train[train.part == "val"].reset_index(drop=True)
    test = load_test()
    assert not fit.key.isin(val_keys).any() and len(fit) >= 150000
    y_fit = fit.label.map(LABEL2ID).values
    order = np.array(LABELS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {"fit_rows": len(fit), "fit_excludes_val_keys": True, "fit_part_counts": fit.part.value_counts().to_dict(), "views": {}}
    for name, view in VIEWS.items():
        started = time.time()
        X_fit = hashed(view(fit.text))
        tfidf = TfidfTransformer(sublinear_tf=True).fit(X_fit)
        X_fit = tfidf.transform(X_fit)
        X_val = tfidf.transform(hashed(view(val.text)))
        X_test = tfidf.transform(hashed(view(test.text)))
        model = LinearSVC(C=0.3, random_state=SEED).fit(X_fit, y_fit)
        val_scores = model.decision_function(X_val).astype(np.float32)
        test_scores = model.decision_function(X_test).astype(np.float32)
        pred = order[val_scores.argmax(axis=1)]
        np.save(ARTIFACTS / f"bp_{name}_val.npy", val_scores)
        np.save(ARTIFACTS / f"bp_{name}_test.npy", test_scores)
        summary["views"][name] = {
            "macro_f1": macro_f1(val.label.values, pred),
            "weighted_macro_f1": macro_f1(val.label.values, pred, val.weight.values),
            "per_class_f1": dict(zip(LABELS, f1_score(val.label.values, pred, labels=LABELS, average=None).tolist())),
            "seconds": round(time.time() - started),
        }
        print(name, summary["views"][name], flush=True)
    (RESULTS / "baseline_pool.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
