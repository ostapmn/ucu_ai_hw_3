import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import ARTIFACTS, IMAGES, LABELS, LABEL2ID, RESULTS, SEED, split_frames

TOP = 25


def linear_tokens():
    train, _ = split_frames()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200000, sublinear_tf=True, dtype=np.float32)
    X = vectorizer.fit_transform(train.text)
    model = LogisticRegression(C=3.0, max_iter=300, random_state=SEED).fit(X, train.label.map(LABEL2ID))
    names = np.array(vectorizer.get_feature_names_out())
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    for k, label in enumerate(LABELS):
        order = np.argsort(model.coef_[k])[::-1][:TOP]
        for rank, index in enumerate(order):
            rows.append({"label": label, "rank": rank + 1, "token": names[index], "coef": float(model.coef_[k][index])})
        axes[k].barh(names[order][::-1], model.coef_[k][order][::-1])
        axes[k].set_title(f"Top tokens for {label}")
    fig.tight_layout()
    fig.savefig(IMAGES / "xai_linear_top_tokens.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(RESULTS / "explain_linear_top_tokens.csv", index=False)


def integrated_gradients(model, input_ids, mask, target, steps=24):
    import torch

    embeddings = model.get_input_embeddings()(input_ids).detach()
    baseline = torch.zeros_like(embeddings)
    total = torch.zeros_like(embeddings)
    for alpha in np.linspace(0.05, 1.0, steps):
        point = (baseline + alpha * (embeddings - baseline)).requires_grad_(True)
        logits = model(inputs_embeds=point, attention_mask=mask).logits
        model.zero_grad()
        logits[0, target].backward()
        total += point.grad
    return ((embeddings - baseline) * total / steps).sum(-1)[0].detach().numpy()


def transformer_attributions(model_dir, per_class=8):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _, val = split_frames()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32).eval()
    sample = val[val.lang == "en-EN"].groupby("label", group_keys=False).sample(n=per_class, random_state=SEED)
    rows = []
    for _, row in sample.iterrows():
        enc = tokenizer(row.text, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            predicted = int(model(**enc).logits.argmax())
        attr = integrated_gradients(model, enc["input_ids"], enc["attention_mask"], predicted)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        for position, (token, value) in enumerate(zip(tokens, attr)):
            rows.append(
                {
                    "id": row.id,
                    "label": row.label,
                    "predicted": LABELS[predicted],
                    "position": position,
                    "token": token,
                    "attribution": float(value),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "explain_transformer_attributions.csv", index=False)
    clean = frame[~frame.token.isin(["[CLS]", "[SEP]"])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for k, label in enumerate(LABELS):
        part = clean[clean.predicted == label]
        top = part.groupby("token").attribution.agg(["sum", "count"])
        top = top[top["count"] >= 3].sort_values("sum", ascending=False).head(20)
        axes[k].barh(top.index[::-1], top["sum"].values[::-1])
        axes[k].set_title(f"Tokens driving prediction: {label}")
    fig.tight_layout()
    fig.savefig(IMAGES / "xai_transformer_tokens.png", dpi=130)
    plt.close(fig)
    position_share = clean.assign(bucket=pd.cut(clean.position, [0, 64, 128, 256, 384, 512], include_lowest=True))
    position_share = position_share.groupby(["predicted", "bucket"], observed=True).attribution.apply(lambda s: s.abs().mean()).unstack(0)
    position_share.to_csv(RESULTS / "explain_position_profile.csv")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    linear_tokens()
    if len(sys.argv) > 1:
        transformer_attributions(sys.argv[1])
    print("explain done")


if __name__ == "__main__":
    main()
