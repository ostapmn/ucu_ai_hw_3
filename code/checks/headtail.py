import os
import subprocess
import tempfile
from pathlib import Path

import common
from windows import head_tail

TINY = "google/bert_uncased_L-2_H-128_A-2"
FALLBACK = "/private/tmp/claude-501/-Users-ostapmnykh-AI-Course-ucu-ai-hw-3/c2bf751e-19f9-4780-bf68-77591b6a0c22/scratchpad/venv/bin/python"


def headtail():
    tokens = list(range(1000))
    window = head_tail(tokens, 510, 128)
    assert len(window) == 510
    assert window[:128] == tokens[:128]
    assert window[128:] == tokens[-382:]
    assert head_tail(tokens[:300], 510, 128) == tokens[:300]
    assert head_tail(tokens[:510], 510, 128) == tokens[:510]
    assert len(head_tail(tokens, 510, 509)) == 510
    python = os.environ.get("HW3_TORCH_PYTHON", FALLBACK)
    assert Path(python).exists(), python
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "RESULTS_DIR": tmp, "ARTIFACTS_DIR": str(common.ARTIFACTS)}
        command = [
            python,
            str(common.ROOT / "code" / "train_encoder.py"),
            "--smoke",
            "--mode",
            "headtail",
            "--model",
            TINY,
            "--max-length",
            "128",
            "--head-tokens",
            "32",
            "--split-col",
            "part2",
            "--epochs",
            "2",
            "--time-budget-hours",
            "0.00003",
            "--checkpoint-dir",
            tmp,
            "--checkpoint-every",
            "10",
            "--prefix",
            "smoke_check",
        ]
        first = subprocess.run(command, env=env, capture_output=True, text=True)
        assert first.returncode == 0, first.stderr[-2000:]
        assert (Path(tmp) / "smoke_check_ckpt.pt").exists()
        second = subprocess.run(command, env=env, capture_output=True, text=True)
        assert second.returncode == 0, second.stderr[-2000:]
        assert "resumed epoch" in second.stdout, second.stdout[-1000:]


CHECKS = {"headtail": headtail}
