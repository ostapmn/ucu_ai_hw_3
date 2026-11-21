import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import IMAGES, LABELS, RESULTS, SEED, load_test, load_train
from features import stylometric_features


def class_balance(train):
    by_lang = pd.crosstab(train.lang, train.label, normalize="index")[LABELS]
    by_domain = pd.crosstab(train.domain, train.label, normalize="index")[LABELS]
    counts = pd.crosstab([train.lang, train.domain], train.label)[LABELS]
    table = pd.concat({"lang_share": by_lang, "domain_share": by_domain}, names=["view", "key"])
    table.to_csv(RESULTS / "eda_class_balance.csv")
    counts.to_csv(RESULTS / "eda_class_counts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    by_domain.plot.barh(stacked=True, ax=axes[0], title="Class share by domain")
    by_lang.plot.barh(stacked=True, ax=axes[1], title="Class share by language")
    for ax in axes:
        ax.set_xlabel("share")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.tight_layout()
    fig.savefig(IMAGES / "eda_class_balance.png", dpi=130)
    plt.close(fig)


def lengths(train, test):
    train = train.assign(chars=train.text.str.len(), words=train.text.str.split().str.len())
    test = test.assign(chars=test.text.str.len(), words=test.text.str.split().str.len())
    stats = pd.concat(
        {
            "label": train.groupby("label")[["chars", "words"]].describe(),
            "domain": train.groupby("domain")[["chars", "words"]].describe(),
        }
    )
    stats.to_csv(RESULTS / "eda_length_stats.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for label in LABELS:
        train[train.label == label].chars.clip(upper=15000).plot.hist(bins=60, alpha=0.5, ax=axes[0], label=label, density=True)
    axes[0].set_title("Characters per text by class (train)")
    axes[0].legend()
    train.chars.clip(upper=15000).plot.hist(bins=60, alpha=0.5, ax=axes[1], label="train", density=True)
    test.chars.clip(upper=15000).plot.hist(bins=60, alpha=0.5, ax=axes[1], label="test", density=True)
    axes[1].set_title("Characters per text: train vs test")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "eda_lengths.png", dpi=130)
    plt.close(fig)


def examples(train):
    rows = []
    rng_train = train.sample(frac=1.0, random_state=SEED)
    for label in LABELS:
        for _, row in rng_train[(rng_train.label == label) & (rng_train.lang == "en-EN")].head(3).iterrows():
            rows.append({"label": label, "domain": row.domain, "lang": row.lang, "chars": len(row.text), "text": row.text[:600]})
    pd.DataFrame(rows).to_csv(RESULTS / "eda_examples.csv", index=False)


def fusion_style(train):
    sample = train.groupby("label", group_keys=False).sample(n=4000, random_state=SEED).reset_index(drop=True)
    feats = stylometric_features(sample.text)
    feats["label"] = sample.label
    feats["domain"] = sample.domain
    table = feats.groupby("label").mean(numeric_only=True).loc[LABELS]
    table.to_csv(RESULTS / "eda_style_by_class.csv")
    normalized = (table - table.mean()) / table.std().replace(0, 1)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    normalized.T.plot.bar(ax=ax, title="Stylometric features by class (z-scored across classes)")
    fig.tight_layout()
    fig.savefig(IMAGES / "eda_fusion_style.png", dpi=130)
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    train = load_train()
    test = load_test()
    class_balance(train)
    lengths(train, test)
    examples(train)
    fusion_style(train)
    print("eda done")


if __name__ == "__main__":
    main()
