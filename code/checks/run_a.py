import numpy as np

import common
from verify import logits_gate, read_metrics


def check_run(prefix, split_col, same_seed):
    metrics = logits_gate(prefix, check_seed=same_seed)
    reference = read_metrics("ref_head.json")
    assert metrics["model"] == reference["model"]
    assert metrics["max_length"] == reference["max_length"]
    assert abs(metrics["train_size"] - reference["train_size"]) <= 100
    assert (metrics["seed"] == reference["seed"]) == same_seed
    assert metrics["mode"] == "headtail" and metrics["split_col"] == split_col
    assert metrics["head_tokens"] == 128 and metrics["epochs"] == reference["epochs"]
    assert metrics["stopped_early"] is False
    split = common.load_split()
    val_keys = set(split.key[split.part == "val"])
    assert not split.key[split[split_col] == "train"].isin(val_keys).any()
    return metrics


def run_a():
    check_run("headtail_a", "part2", True)


def run_b():
    check_run("headtail_b", "part3", False)
    a = np.load(common.ARTIFACTS / "headtail_a_val.npy")
    b = np.load(common.ARTIFACTS / "headtail_b_val.npy")
    assert not np.allclose(a, b, atol=1e-3)


CHECKS = {"run_a": run_a, "run_b": run_b}
