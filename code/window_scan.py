import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import ARTIFACTS, RESULTS, load_test, split_frames

BODY = 510
STRIDE = 255
MAX_WINDOWS = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prefix", default="winscan_a")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def scan_items(texts, tokenizer):
    items = []
    for start in range(0, len(texts), 512):
        batch = tokenizer(list(texts[start : start + 512]), add_special_tokens=False, return_attention_mask=False)["input_ids"]
        for offset, tokens in enumerate(batch):
            doc = start + offset
            if len(tokens) <= BODY:
                items.append((doc, 0, tokens))
                continue
            count = min(MAX_WINDOWS, int(np.ceil((len(tokens) - BODY) / STRIDE)) + 1)
            for index, first in enumerate(np.linspace(0, len(tokens) - BODY, count).astype(int)):
                items.append((doc, index, tokens[first : first + BODY]))
    return items


@torch.no_grad()
def probabilities(model, items, tokenizer, batch_size, device):
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    order = sorted(range(len(items)), key=lambda i: len(items[i][2]))
    out = np.zeros((len(items), 3), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        rows = [[cls_id] + list(items[i][2]) + [sep_id] for i in chunk]
        width = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for k, r in enumerate(rows):
            input_ids[k, : len(r)] = torch.tensor(r)
            mask[k, : len(r)] = 1
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
        out[chunk] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return out


def aggregate(items, probs, n_docs):
    per_doc = [[] for _ in range(n_docs)]
    for (doc, index, _), p in zip(items, probs):
        per_doc[doc].append((index, p))
    features = np.zeros((n_docs, 17), dtype=np.float32)
    for doc, rows in enumerate(per_doc):
        rows.sort(key=lambda r: r[0])
        matrix = np.stack([p for _, p in rows])
        position = matrix[:, 2].argmax() / max(len(rows) - 1, 1)
        features[doc] = np.concatenate([matrix.mean(0), matrix.max(0), matrix.min(0), matrix[0], matrix[-1], [len(rows), position]])
    return features


def run(model, tokenizer, texts, batch_size, device):
    items = scan_items(texts, tokenizer)
    return aggregate(items, probabilities(model, items, tokenizer, batch_size, device), len(texts))


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    _, val = split_frames("part2")
    test = load_test()
    started = time.time()
    val_texts, test_texts = val.text.tolist(), test.text.tolist()
    if args.smoke:
        val_texts, test_texts = val_texts[:64], test_texts[:64]
    val_features = run(model, tokenizer, val_texts, args.batch_size, device)
    print("val done", time.time() - started, flush=True)
    test_features = run(model, tokenizer, test_texts, args.batch_size, device)
    print("test done", time.time() - started, flush=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS / f"{args.prefix}_val.npy", val_features)
    np.save(ARTIFACTS / f"{args.prefix}_test.npy", test_features)
    (RESULTS / f"{args.prefix}.json").write_text(json.dumps({"model_dir": args.model_dir, "body": BODY, "stride": STRIDE, "max_windows": MAX_WINDOWS, "seconds": time.time() - started, "mean_windows_val": float(val_features[:, 15].mean())}, indent=2))


if __name__ == "__main__":
    main()
