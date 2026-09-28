"""Shared cleaning and fixed train/val/test split for the whole group.

Run from the repo root:
    python -m src.preprocessing
or from a notebook:
    from src.preprocessing import make_splits, load_splits
"""
import os
import re

import emoji
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/Train.csv"
OUT_DIR = "data/processed"
SEED = 42
VALID_LABELS = [-1, 0, 1]

PLACEHOLDER_RE = re.compile(r"<(user|url)>", flags=re.I)
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
RT_RE = re.compile(r"^\s*rt\s+", flags=re.I)
REPEAT_RE = re.compile(r"(.)\1{2,}")
NON_TEXT_RE = re.compile(r"[^a-z0-9_'!? ]")
WS_RE = re.compile(r"\s+")


def clean_text(text: str, demojize: bool = True) -> str:
    """Shared cleaning used by every model in the project."""
    text = str(text)
    text = RT_RE.sub("", text)
    text = PLACEHOLDER_RE.sub(" ", text)        # drop <user>/<url> anonymisation tokens
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("&amp;", " and ")
    if demojize:                                 # keep emoji meaning as words
        text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.replace("#", " ")                # keep hashtag words, drop the symbol
    text = text.lower()
    text = REPEAT_RE.sub(r"\1\1", text)          # "soooo" -> "soo"
    text = NON_TEXT_RE.sub(" ", text)            # keep ! and ? as sentiment cues
    return WS_RE.sub(" ", text).strip()


def make_splits(raw_path=RAW_PATH, out_dir=OUT_DIR, seed=SEED):
    """Clean, deduplicate and create the fixed stratified 70/15/15 train/val/test split."""
    df = pd.read_csv(raw_path)
    n0 = len(df)
    df = df.dropna(subset=["safe_text", "label", "agreement"])
    df = df[df["label"].isin(VALID_LABELS)].copy()
    print(f"Dropped {n0 - len(df)} rows with missing/odd label or missing agreement")

    df["label"] = df["label"].astype(int)
    df["clean_text"] = df["safe_text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    # deduplicate on cleaned text BEFORE splitting; keep the highest-agreement copy
    n1 = len(df)
    df = (df.sort_values("agreement", ascending=False, kind="stable")
            .drop_duplicates("clean_text", keep="first"))
    print(f"Dropped {n1 - len(df)} duplicates after cleaning; {len(df)} tweets remain")

    df["label_id"] = df["label"] + 1             # -1,0,1 -> 0,1,2 for neural models

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed)

    os.makedirs(out_dir, exist_ok=True)
    cols = ["tweet_id", "safe_text", "clean_text", "label", "label_id", "agreement"]
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part[cols].to_csv(f"{out_dir}/{name}.csv", index=False)
        print(name, len(part), part["label"].value_counts(normalize=True).round(3).to_dict())


def load_splits(data_dir=OUT_DIR):
    """Return (train, val, test) DataFrames from the shared split files."""
    return (pd.read_csv(f"{data_dir}/train.csv"),
            pd.read_csv(f"{data_dir}/val.csv"),
            pd.read_csv(f"{data_dir}/test.csv"))


if __name__ == "__main__":
    make_splits()
