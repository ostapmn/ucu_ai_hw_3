import json

import numpy as np
import pandas as pd

import common


def stack_v2_final():
    final = json.loads((common.RESULTS / "stack_v2_final.json").read_text())
    results = json.loads((common.RESULTS / "stack_v2.json").read_text())
    assert final["selected_by"] == "dev_cv_and_lock" and final["public_leaderboard_used"] is False
    assert set(final["candidates"]) == {"candidate_1", "candidate_2"}
    assert final["candidates"]["candidate_1"]["tag"] != final["candidates"]["candidate_2"]["tag"]
    sample = common.load_sample_submission()
    for row in final["candidates"].values():
        recorded = results[row["tag"]]
        assert row["dev_cv_weighted_macro_f1"] == recorded["dev_cv_weighted_macro_f1"]
        assert row["lock_weighted_macro_f1"] == recorded["lock_weighted_macro_f1"]
        assert row["dev_cv_weighted_macro_f1"] > 0.95 and row["lock_weighted_macro_f1"] > 0.95
        sub = pd.read_csv(common.ROOT / row["submission_file"])
        assert list(sub.columns) == ["id", "label"] and len(sub) == 25000
        assert (sub.id.values == sample.id.values).all() and set(sub.label) <= set(common.LABELS)
        test = np.load(common.ARTIFACTS / f"stack_v2_{row['tag']}_test.npy")
        assert (np.array(common.LABELS)[test.argmax(1)] == sub.label.values).all()


CHECKS = {"stack_v2_final": stack_v2_final}
