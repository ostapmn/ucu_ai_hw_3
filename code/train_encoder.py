import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import ARTIFACTS, LABEL2ID, RESULTS, SEED, domain_frames, load_test, macro_f1, split_frames, LABELS
from windows import head_tail


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["head", "chunks", "headtail"], default="head")
    parser.add_argument("--model", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--prefix", default="ref_head")
    parser.add_argument("--train-size", type=int, default=60000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval-windows", type=int, default=3)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--time-budget-hours", type=float, default=6.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--head-tokens", type=int, default=128)
    parser.add_argument("--split-col", default="part")
    parser.add_argument("--train-domain", default=None)
    parser.add_argument("--init-model", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    return parser.parse_args()


def save_checkpoint(path, model, optimizer, scheduler, scaler, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "state": state,
        },
        tmp,
    )
    tmp.replace(path)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tokenizer, texts, body, mode, head):
    kwargs = {"add_special_tokens": False, "return_attention_mask": False}
    if mode == "head":
        kwargs.update(truncation=True, max_length=body)
    ids = []
    for start in range(0, len(texts), 512):
        batch = tokenizer(list(texts[start : start + 512]), **kwargs)["input_ids"]
        if mode == "headtail":
            batch = [head_tail(tokens, body, head) for tokens in batch]
        ids.extend(batch)
    return ids


class WindowDataset(Dataset):
    def __init__(self, ids, labels, body, random_window, cls_id, sep_id):
        self.ids, self.labels, self.body = ids, labels, body
        self.random_window, self.cls_id, self.sep_id = random_window, cls_id, sep_id

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        tokens = self.ids[index]
        if len(tokens) > self.body:
            start = np.random.randint(0, len(tokens) - self.body + 1) if self.random_window else 0
            tokens = tokens[start : start + self.body]
        return [self.cls_id] + list(tokens) + [self.sep_id], self.labels[index]


def collate(batch, pad_id):
    width = max(len(x) for x, _ in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (x, _) in enumerate(batch):
        input_ids[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = 1
    return input_ids, mask, torch.tensor([y for _, y in batch], dtype=torch.long)


def eval_windows(ids, body, windows):
    items = []
    for doc, tokens in enumerate(ids):
        if len(tokens) <= body or windows == 1:
            items.append((doc, tokens[:body]))
            continue
        for start in np.linspace(0, len(tokens) - body, windows).astype(int):
            items.append((doc, tokens[start : start + body]))
    return items


@torch.no_grad()
def predict(model, items, n_docs, cls_id, sep_id, pad_id, batch_size, device):
    model.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i][1]))
    sums = np.zeros((n_docs, 3), dtype=np.float64)
    counts = np.zeros(n_docs)
    for start in range(0, len(order), batch_size):
        chunk = [items[i] for i in order[start : start + batch_size]]
        batch = [([cls_id] + list(tokens) + [sep_id], 0) for _, tokens in chunk]
        input_ids, mask, _ = collate(batch, pad_id)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
        logits = logits.float().cpu().numpy()
        for (doc, _), row in zip(chunk, logits):
            sums[doc] += row
            counts[doc] += 1
    return (sums / counts[:, None]).astype(np.float32)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    body = args.max_length - 2
    train, val = domain_frames(args.train_domain) if args.train_domain else split_frames(args.split_col)
    if args.smoke:
        train = train.sample(n=512, random_state=SEED).reset_index(drop=True)
    elif args.train_size < len(train):
        train = train.sample(n=args.train_size, random_state=args.seed).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(args.init_model or args.model)
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(args.init_model or args.model, num_labels=3, dtype=torch.float32)
    model.to(device)
    parallel = nn.DataParallel(model) if device == "cuda" and torch.cuda.device_count() > 1 else model
    started = time.time()
    train_ids = encode(tokenizer, train.text.tolist(), body, args.mode, args.head_tokens)
    train_labels = train.label.map(LABEL2ID).tolist()
    dataset = WindowDataset(train_ids, train_labels, body, args.mode == "chunks", cls_id, sep_id)
    n_batches = len(dataset) // args.batch_size
    steps = n_batches * args.epochs // args.accum
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    state = {"epoch": 0, "step": 0, "seen": 0, "elapsed": 0.0, "stopped_early": False}
    checkpoint = Path(args.checkpoint_dir) / f"{args.prefix}_ckpt.pt" if args.checkpoint_dir else None
    if checkpoint is not None and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        print(f"resumed epoch {state['epoch']} step {state['step']} elapsed {state['elapsed']:.0f}s", flush=True)
    seen, base_elapsed, train_started = state["seen"], state["elapsed"], time.time()
    stopped_early = state["stopped_early"]
    for epoch in range(state["epoch"], args.epochs):
        if stopped_early:
            break
        parallel.train()
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        batches = [order[i : i + args.batch_size] for i in range(0, n_batches * args.batch_size, args.batch_size)]
        first = state["step"] if epoch == state["epoch"] else 0
        loader = DataLoader(dataset, batch_sampler=batches[first:], collate_fn=partial(collate, pad_id=pad_id), num_workers=2)
        for step, (input_ids, mask, labels) in enumerate(loader, start=first):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = parallel(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
            loss = loss_fn(logits.float(), labels.to(device)) / args.accum
            scaler.scale(loss).backward()
            seen += len(labels)
            if (step + 1) % args.accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            elapsed = base_elapsed + time.time() - train_started
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{n_batches} loss {loss.item() * args.accum:.4f} elapsed {elapsed:.0f}s", flush=True)
            if elapsed > args.time_budget_hours * 3600:
                stopped_early = True
                break
            if checkpoint is not None and (step + 1) % args.checkpoint_every == 0:
                state = {"epoch": epoch, "step": step + 1, "seen": seen, "elapsed": elapsed, "stopped_early": False}
                save_checkpoint(checkpoint, model, optimizer, scheduler, scaler, state)
                print(f"checkpoint epoch {epoch} step {step + 1}", flush=True)
        state = {"epoch": epoch + 1, "step": 0, "seen": seen, "elapsed": base_elapsed + time.time() - train_started, "stopped_early": stopped_early}
    train_seconds = base_elapsed + time.time() - train_started
    if checkpoint is not None:
        save_checkpoint(checkpoint, model, optimizer, scheduler, scaler, state)
    samples_per_sec = seen / train_seconds
    RESULTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        info = {
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "",
            "n_gpus": torch.cuda.device_count(),
            "samples_per_sec": samples_per_sec,
            "internet": True,
            "model": args.model,
            "max_length": args.max_length,
            "estimated_hours_full_run": args.train_size * args.epochs / samples_per_sec / 3600,
        }
        (RESULTS / "smoke.json").write_text(json.dumps(info, indent=2))
        print(info)
        return
    if args.save_model:
        model.half().save_pretrained(ARTIFACTS / f"{args.prefix}_model")
        tokenizer.save_pretrained(ARTIFACTS / f"{args.prefix}_model")
        model.float()
    val_ids = encode(tokenizer, val.text.tolist(), body, args.mode, args.head_tokens)
    windows = args.eval_windows if args.mode == "chunks" else 1
    val_logits = predict(model, eval_windows(val_ids, body, windows), len(val_ids), cls_id, sep_id, pad_id, args.eval_batch_size, device)
    pred = np.array(LABELS)[val_logits.argmax(axis=1)]
    metrics = {
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "train_size": len(train),
        "max_length": args.max_length,
        "split_col": args.split_col,
        "train_domain": args.train_domain,
        "init_model": bool(args.init_model),
        "head_tokens": args.head_tokens,
        "epochs": args.epochs,
        "lr": args.lr,
        "eval_windows": windows,
        "stopped_early": stopped_early,
        "train_seconds": train_seconds,
        "samples_per_sec": samples_per_sec,
        "macro_f1": macro_f1(val.label.values, pred),
        "weighted_macro_f1": macro_f1(val.label.values, pred, val.weight.values),
    }
    print(metrics, flush=True)
    np.save(ARTIFACTS / f"{args.prefix}_val.npy", val_logits)
    (RESULTS / f"{args.prefix}.json").write_text(json.dumps(metrics, indent=2))
    test = load_test()
    test_ids = encode(tokenizer, test.text.tolist(), body, args.mode, args.head_tokens)
    test_logits = predict(model, eval_windows(test_ids, body, windows), len(test_ids), cls_id, sep_id, pad_id, args.eval_batch_size, device)
    np.save(ARTIFACTS / f"{args.prefix}_test.npy", test_logits)
    print("done", time.time() - started, flush=True)


if __name__ == "__main__":
    main()
