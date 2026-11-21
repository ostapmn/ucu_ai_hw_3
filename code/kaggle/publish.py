import json
import shutil
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
BUILD = Path(__file__).resolve().parent / "build"
USER = "ostapmn"
DATASET = "hw3-code"
COMPETITION = "ai-generated-text-detection-hw3-ucu-ai-2026"
FILES = ["common.py", "train_encoder.py"]

RUNNER = """import glob
import os
import shutil
import subprocess
import sys

SRC = os.path.dirname(glob.glob("/kaggle/input/**/common.py", recursive=True)[0])
WORK = "/kaggle/working"
os.makedirs(WORK + "/artifacts", exist_ok=True)
os.makedirs(WORK + "/results", exist_ok=True)
for name in %(files)r:
    shutil.copy(SRC + "/" + name, WORK + "/" + name)
shutil.copy(SRC + "/split.csv", WORK + "/artifacts/split.csv")
train_csv = glob.glob("/kaggle/input/**/train.csv", recursive=True)[0]
env = dict(os.environ, DATA_DIR=os.path.dirname(train_csv), ARTIFACTS_DIR=WORK + "/artifacts", RESULTS_DIR=WORK + "/results")
subprocess.run([sys.executable, "train_encoder.py"] + %(args)r, cwd=WORK, env=env, check=True)
"""


def run(cmd):
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def publish_dataset():
    folder = BUILD / "dataset"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for name in FILES:
        shutil.copy(CODE / name, folder / name)
    shutil.copy(CODE / "artifacts" / "split.csv", folder / "split.csv")
    meta = {"title": DATASET, "id": f"{USER}/{DATASET}", "licenses": [{"name": "other"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    exists = subprocess.run(["kaggle", "datasets", "status", f"{USER}/{DATASET}"], capture_output=True).returncode == 0
    if exists:
        run(["kaggle", "datasets", "version", "-p", str(folder), "-m", "update", "-r", "zip"])
    else:
        run(["kaggle", "datasets", "create", "-p", str(folder), "-r", "zip"])


def publish_kernel(slug, args):
    folder = BUILD / slug
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "run.py").write_text(RUNNER % {"files": FILES, "args": args})
    meta = {
        "id": f"{USER}/{slug}",
        "title": slug,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{USER}/{DATASET}"],
        "competition_sources": [COMPETITION],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    run(["kaggle", "kernels", "push", "-p", str(folder), "--accelerator", "NvidiaTeslaT4"])


def main():
    if sys.argv[1] == "--dataset":
        publish_dataset()
    else:
        publish_kernel(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    main()
