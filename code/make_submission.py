import sys

import numpy as np
import pandas as pd

from common import LABELS, load_sample_submission


def write_submission(scores, out_path):
    sample = load_sample_submission()
    scores = np.asarray(scores)
    assert scores.shape == (len(sample), len(LABELS)), scores.shape
    labels = np.array(LABELS)[scores.argmax(axis=1)]
    pd.DataFrame({"id": sample.id, "label": labels}).to_csv(out_path, index=False)
    return pd.Series(labels).value_counts(normalize=True).round(4).to_dict()


def main():
    scores_path, out_path = sys.argv[1], sys.argv[2]
    print(write_submission(np.load(scores_path), out_path))


if __name__ == "__main__":
    main()
