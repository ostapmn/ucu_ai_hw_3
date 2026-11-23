import json

import numpy as np

import common


def baseline_pool():
    summary = json.loads((common.RESULTS / "baseline_pool.json").read_text())
    split = common.load_split()
    val_keys = set(split.key[split.part == "val"])
    allowed = (split.part != "val") & ~split.key.isin(val_keys)
    assert summary["fit_excludes_val_keys"] is True
    assert summary["fit_rows"] == int(allowed.sum()) >= 150000
    assert set(summary["views"]) == {"full", "head", "tail"}
    n_val = int((split.part == "val").sum())
    for name, row in summary["views"].items():
        val = np.load(common.ARTIFACTS / f"bp_{name}_val.npy")
        test = np.load(common.ARTIFACTS / f"bp_{name}_test.npy")
        assert val.shape == (n_val, 3) and test.shape == (25000, 3), name
        assert np.isfinite(val).all() and np.isfinite(test).all()
        assert 0.3 < row["macro_f1"] <= 1.0 and "weighted_macro_f1" in row


CHECKS = {"baseline_pool": baseline_pool}
