import numpy as np
import pandas as pd


def stylometric_features(texts):
    texts = pd.Series(texts).fillna("").astype(str).reset_index(drop=True)
    n_chars = texts.str.len().clip(lower=1)
    words = texts.str.findall(r"\w+")
    n_words = words.str.len().clip(lower=1)
    word_len = words.apply(lambda w: np.mean([len(x) for x in w]) if w else 0.0)
    unique = words.apply(lambda w: len({x.lower() for x in w}))
    n_sentences = texts.str.count(r"[.!?]+(?:\s|$)").clip(lower=1)
    n_paragraphs = texts.str.count(r"\n\s*\n").add(1)
    feats = pd.DataFrame(
        {
            "log_chars": np.log1p(n_chars),
            "log_words": np.log1p(n_words),
            "avg_word_len": word_len,
            "avg_sentence_len": n_words / n_sentences,
            "type_token_ratio": unique / n_words,
            "punct_ratio": texts.str.count(r"[^\w\s]") / n_chars,
            "comma_per_word": texts.str.count(",") / n_words,
            "upper_ratio": texts.str.count(r"[A-ZÄÖÜ]") / n_chars,
            "digit_ratio": texts.str.count(r"\d") / n_chars,
            "newline_ratio": texts.str.count("\n") / n_chars,
            "non_ascii_ratio": texts.str.count(r"[^\x00-\x7f]") / n_chars,
            "quote_ratio": texts.str.count(r"[\"'“”‘’„]") / n_chars,
            "long_word_ratio": words.apply(lambda w: np.mean([len(x) > 8 for x in w]) if w else 0.0),
            "paragraphs": np.log1p(n_paragraphs),
            "space_ratio": texts.str.count(" ") / n_chars,
        }
    )
    return feats.astype(np.float32)


def drift_features(texts):
    texts = pd.Series(texts).fillna("").astype(str).reset_index(drop=True)
    size = texts.str.len()
    cut = (size // 3).astype(int)
    first = pd.Series([t[:c] for t, c in zip(texts, cut)])
    last = pd.Series([t[len(t) - c :] if c else "" for t, c in zip(texts, cut)])
    a = stylometric_features(first).drop(columns=["log_chars", "log_words", "paragraphs"])
    b = stylometric_features(last).drop(columns=["log_chars", "log_words", "paragraphs"])
    out = pd.concat([a.add_prefix("first_"), b.add_prefix("last_"), (b - a).add_prefix("delta_")], axis=1)
    return out.astype(np.float32)
