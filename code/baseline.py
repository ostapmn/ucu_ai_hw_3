import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import LinearSVC

from common import ARTIFACTS, LABELS, LABEL2ID, RESULTS, SEED, load_test, macro_f1, split_frames

GRID = [
    ("linear_svc", {"C": 0.1}),
    ("linear_svc", {"C": 0.3}),
    ("linear_svc", {"C": 1.0}),
    ("logreg", {"C": 3.0}),
    ("logreg", {"C": 10.0}),
]


def build_model(name, params):
    if name == "linear_svc":
        return LinearSVC(random_state=SEED, **params)
    return LogisticRegression(max_iter=300, random_state=SEED, **params)


def scores_of(model, X):
    return model.decision_function(X)


def main():
    train, val = split_frames()
    test = load_test()
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=400000, sublinear_tf=True, dtype=np.float32)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=400000, sublinear_tf=True, dtype=np.float32)
    start = time.time()
    X_train = sp.hstack([word.fit_transform(train.text), char.fit_transform(train.text)]).tocsr()
    transform = lambda texts: sp.hstack([word.transform(texts), char.transform(texts)]).tocsr()
    X_val = transform(val.text)
    X_test = transform(test.text)
    print("vectorized", round(time.time() - start), "s", X_train.shape, flush=True)
    y_train = train.label.map(LABEL2ID).values
    y_val = val.label.values
    order = np.array(LABELS)
    results, best = [], None
    for name, params in GRID:
        model = build_model(name, params).fit(X_train, y_train)
        pred = order[scores_of(model, X_val).argmax(axis=1)]
        row = {
            "model": name,
            "params": params,
            "macro_f1": macro_f1(y_val, pred),
            "weighted_macro_f1": macro_f1(y_val, pred, val.weight.values),
        }
        results.append(row)
        print(row, flush=True)
        if best is None or row["weighted_macro_f1"] > best[0]["weighted_macro_f1"]:
            best = (row, model)
    row, model = best
    val_scores = scores_of(model, X_val)
    pred = order[val_scores.argmax(axis=1)]
    per_class = f1_score(y_val, pred, labels=LABELS, average=None)
    matrix = confusion_matrix(y_val, pred, labels=LABELS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(RESULTS / "baseline_confusion.csv")
    pd.DataFrame(results).to_csv(RESULTS / "baseline_grid.csv", index=False)
    np.save(ARTIFACTS / "baseline_val.npy", val_scores)
    np.save(ARTIFACTS / "baseline_test.npy", scores_of(model, X_test))
    summary = {
        "model": row["model"],
        "best_params": row["params"],
        "macro_f1": row["macro_f1"],
        "weighted_macro_f1": row["weighted_macro_f1"],
        "per_class_f1": dict(zip(LABELS, per_class.tolist())),
        "confusion_matrix": matrix.tolist(),
        "fit_on": "train",
        "train_size": len(train),
        "val_size": len(val),
    }
    (RESULTS / "baseline.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
