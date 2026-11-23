import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from common import ARTIFACTS, SEED, doc_keys, load_test, load_train

VAL_FRACTION = 0.15
TRAIN_SIZE = 60000
LOCK_SIZE = 8000
N_FOLDS = 5
SEED_B = 7


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


def sample_train(frame, size, seed):
    cell = frame.domain + "|" + frame.label
    cap = find_cap(cell.value_counts().values, size)
    rng = np.random.RandomState(seed)
    chosen = []
    for _, group in frame.groupby(cell):
        chosen.extend(rng.choice(group.index.values, size=min(len(group), cap), replace=False))
    return chosen


def build_split():
    raw = load_train()
    train = raw[["id", "lang", "domain", "label"]].copy()
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
    train["key"] = doc_keys(raw.text)
    allowed = (train.part != "val") & ~train.key.isin(set(train.key[train.part == "val"]))
    train["part2"] = "other"
    chosen2 = sample_train(train[allowed], TRAIN_SIZE, SEED)
    train.loc[chosen2, "part2"] = "train"
    train["part3"] = "other"
    chosen3 = sample_train(train[allowed & (train.part2 != "train")], TRAIN_SIZE, SEED_B)
    train.loc[chosen3, "part3"] = "train"
    val = train[train.part == "val"]
    group_cell = val.groupby("key").apply(lambda g: g.domain.iloc[0] + "|" + g.lang.iloc[0] + "|" + g.label.iloc[0], include_groups=False)
    keys = group_cell.index.values
    fraction = LOCK_SIZE / len(val)
    rest_keys, lock_keys = train_test_split(keys, test_size=fraction, stratify=group_cell.values, random_state=SEED)
    train["lock"] = 0
    train.loc[val.index[val.key.isin(set(lock_keys))], "lock"] = 1
    train["fold"] = -1
    rest = train[(train.part == "val") & (train.lock == 0)]
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    strata = rest.domain + "|" + rest.lang + "|" + rest.label
    for fold, (_, test_idx) in enumerate(splitter.split(rest, strata, groups=rest.key)):
        train.loc[rest.index[test_idx], "fold"] = fold
    columns = ["id", "part", "weight", "key", "part2", "part3", "lock", "fold"]
    return train[columns].sort_values("id").reset_index(drop=True), cap


def main():
    split, cap = build_split()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / (sys.argv[1] if len(sys.argv) > 1 else "split.csv")
    split.to_csv(out, index=False)
    print(split.part.value_counts().to_dict(), "cap", cap)
    print("part2", split.part2.value_counts().to_dict(), "part3", split.part3.value_counts().to_dict())
    print("lock", int(split.lock.sum()), "folds", split.fold.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
