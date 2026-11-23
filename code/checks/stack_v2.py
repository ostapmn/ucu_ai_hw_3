import json

import numpy as np

import common


def stack_v2():
    results = json.loads((common.RESULTS / "stack_v2.json").read_text())
    assert {"v1_regression", "pool_views", "pool_views_drift"} <= set(results), set(results)
    split = common.load_split()
    val = split[split.part == "val"]
    assert val.lock.sum() > 7500 and set(val.fold[val.lock == 0]) == set(range(5))
    for row in results.values():
        for key in ["dev_cv_weighted_macro_f1", "lock_weighted_macro_f1", "class_weights", "sources"]:
            assert key in row, key
        assert row["folds_disjoint_by_key"] is True
        assert 0.3 < row["dev_cv_weighted_macro_f1"] <= 1.0
        oof = np.load(common.ARTIFACTS / f"stack_v2_{row['tag']}_oof.npy")
        lock = np.load(common.ARTIFACTS / f"stack_v2_{row['tag']}_lock.npy")
        test = np.load(common.ARTIFACTS / f"stack_v2_{row['tag']}_test.npy")
        assert oof.shape == (len(val), 3) and lock.shape == (row["n_lock"], 3) and test.shape == (25000, 3)
        assert np.isfinite(oof).all() and np.isfinite(lock).all() and np.isfinite(test).all()
    regression = results["v1_regression"]["dev_cv_weighted_macro_f1"]
    assert abs(regression - 0.923) < 0.02, regression


CHECKS = {"stack_v2": stack_v2}
