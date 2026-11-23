import common
from features import drift_features, stylometric_features


def drift():
    train, val = common.split_frames()
    a = drift_features(train.text.head(300))
    b = drift_features(val.text.head(300))
    assert list(a.columns) == list(b.columns)
    assert not a.isna().any().any() and not b.isna().any().any()
    assert a.shape[1] >= 20, a.shape
    whole = stylometric_features(train.text.head(5))
    assert not set(whole.columns) & set(a.columns)
    short = drift_features(["a", "", "two words here. and more."])
    assert short.shape[0] == 3 and not short.isna().any().any()
    same = drift_features(["same text. " * 50])
    assert float(same.filter(like="delta_").abs().max().max()) < 0.2


CHECKS = {"drift": drift}
