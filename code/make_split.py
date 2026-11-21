import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import ARTIFACTS, SEED, load_test, load_train

VAL_FRACTION = 0.15
TRAIN_SIZE = 60000


def capped_total(counts, cap):
    return int(np.minimum(counts, cap).sum())


def find_cap(counts, target):
    low, high = 1, int(counts.max())
    while low < high:
        mid = (low + high) // 2
        if capped_total(counts, mid) < target:
            low = mid + 1
        else:
            high = mid
    return low


def build_split():
    train = load_train()[["id", "lang", "domain", "label"]]
    test = load_test()[["id", "lang", "domain"]]
    key = train.domain + "|" + train.lang + "|" + train.label
    pool_idx, val_idx = train_test_split(
        np.arange(len(train)), test_size=VAL_FRACTION, stratify=key, random_state=SEED
    )
    train["part"] = "pool"
    train.loc[val_idx, "part"] = "val"
    pool = train.loc[pool_idx]
    cell = pool.domain + "|" + pool.label
    counts = cell.value_counts()
    cap = find_cap(counts.values, TRAIN_SIZE)
    rng = np.random.RandomState(SEED)
    chosen = []
    for name, group in pool.groupby(cell):
        take = min(len(group), cap)
        chosen.extend(rng.choice(group.index.values, size=take, replace=False))
    train.loc[chosen, "part"] = "train"
    test_mix = (test.domain + "|" + test.lang).value_counts(normalize=True)
    val = train[train.part == "val"]
    val_mix = (val.domain + "|" + val.lang).value_counts(normalize=True)
    ratio = (test_mix / val_mix).fillna(0.0)
    train["weight"] = 0.0
    val_cell = val.domain + "|" + val.lang
    train.loc[val.index, "weight"] = val_cell.map(ratio).values
    return train[["id", "part", "weight"]].sort_values("id").reset_index(drop=True), cap


def main():
    split, cap = build_split()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / (sys.argv[1] if len(sys.argv) > 1 else "split.csv")
    split.to_csv(out, index=False)
    print(split.part.value_counts().to_dict(), "cap", cap)


if __name__ == "__main__":
    main()
