import common


def split_v2_disjoint():
    split = common.load_split()
    train = common.load_train()[["id", "domain", "label", "lang"]].merge(split, on="id")
    val_keys = set(split.key[split.part == "val"])
    assert split.key.notna().all()
    for column in ["part2", "part3"]:
        chosen = split[split[column] == "train"]
        assert 59000 <= len(chosen) <= 61000, (column, len(chosen))
        assert not chosen.key.isin(val_keys).any(), column
        assert (chosen.part != "val").all(), column
    both = (split.part2 == "train") & (split.part3 == "train")
    assert not both.any()
    cells = train[train.part2 == "train"]
    assert cells.domain.nunique() == 8 and set(cells.label) == set(common.LABELS)
    test = common.load_test()
    test_keys = set(common.doc_keys(test.text))
    assert not (test_keys & set(split.key)), "test key overlaps train"


def split_v2_lock():
    split = common.load_split()
    train = common.load_train()[["id", "domain", "label", "lang"]].merge(split, on="id")
    val = train[train.part == "val"]
    lock = val[val.lock == 1]
    assert 7500 <= len(lock) <= 8500, len(lock)
    assert set(split.lock[split.part != "val"]) == {0}
    rest_keys = set(val.key[val.lock == 0])
    assert not (set(lock.key) & rest_keys), "key group split between lock and rest"
    assert set(val.fold[val.lock == 0]) == set(range(5))
    assert set(val.fold[val.lock == 1]) == {-1}
    for fold in range(5):
        part = val[val.fold == fold]
        other = val[(val.fold != fold) & (val.lock == 0)]
        assert not (set(part.key) & set(other.key)), fold
    cell = val.domain + "|" + val.lang + "|" + val.label
    share_all = cell.value_counts(normalize=True)
    share_lock = cell[val.lock == 1].value_counts(normalize=True).reindex(share_all.index).fillna(0.0)
    assert (share_all - share_lock).abs().max() < 0.01


CHECKS = {"split_v2_disjoint": split_v2_disjoint, "split_v2_lock": split_v2_lock}
