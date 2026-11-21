import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "code" / "artifacts"))
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "code" / "results"))
IMAGES = ROOT / "images"
LABELS = ["human", "ai", "fusion"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
SEED = 42


def load_train():
    return pd.read_csv(DATA / "train.csv")


def load_test():
    return pd.read_csv(DATA / "test.csv")


def load_sample_submission():
    return pd.read_csv(DATA / "sample_submission.csv")


def load_split():
    return pd.read_csv(ARTIFACTS / "split.csv")


def split_frames():
    train = load_train().merge(load_split(), on="id", validate="one_to_one")
    return (
        train[train.part == "train"].reset_index(drop=True),
        train[train.part == "val"].reset_index(drop=True),
    )


def macro_f1(y_true, y_pred, weights=None):
    return f1_score(y_true, y_pred, labels=LABELS, average="macro", sample_weight=weights)
