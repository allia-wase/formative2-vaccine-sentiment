"""Shared helpers for error analysis and final model comparison notebooks."""
from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error

from src.evaluation import LABEL_NAMES, LABELS
from src.nn_utils import NEGATION_RE

MODEL_NAMES = [
    "tfidf_logreg",
    "fasttext",
    "bilstm_glove",
    "transformer_scratch",
    "bertweet",
]

# Heuristic sarcasm / irony cues (quoted short span, polite or exaggerated praise, /s).
SARCASM_RE = (
    r"(?:\"[^\"]{1,40}\"|'[^']{1,40}'|"
    r"\bthanks\b|\bthank\s+you\b|\bgreat\b|\blove\s+how\b|\bgenius\b|\bbrilliant\b|"
    r"\bnice\b|\boh\b|\bwow\b|\bso\s+proud\b|\blol\b|#sarcasm|/s\b|"
    r"\byeah\s+right\b|\bsure\b|\bas\s+if\b|\btotally\b|\bobviously\b|"
    r"\bwhat\s+a\s+joke\b|\bgive\s+me\s+a\s+break\b|\bso\s+smart\b)"
)

# First and second person pronouns used to reject the news-style slice.
PERSON_RE = r"\b(?:i|me|my|mine|we|us|our|ours|you|your|yours)\b"
URL_TOKEN_RE = r"(?:<url>|httpurl|https?://|www\.)"


def load_predictions(model_names: Sequence[str] = MODEL_NAMES,
                     pred_dir: str = "reports/predictions") -> Dict[str, pd.DataFrame]:
    """Load prediction CSVs and check they share size, tweet order, y_true and agreement."""
    frames = {name: pd.read_csv(f"{pred_dir}/{name}.csv") for name in model_names}
    base_name = model_names[0]
    base = frames[base_name]
    n = len(base)
    for name, df in frames.items():
        if len(df) != n:
            raise ValueError(f"{name} has {len(df)} rows; expected {n}")
        if list(df["tweet_id"]) != list(base["tweet_id"]):
            raise ValueError(f"{name} tweet_id order differs from {base_name}")
        if not (df["y_true"].to_numpy() == base["y_true"].to_numpy()).all():
            raise ValueError(f"{name} y_true differs from {base_name}")
        if not np.allclose(df["agreement"].to_numpy(dtype=float),
                           base["agreement"].to_numpy(dtype=float), equal_nan=True):
            raise ValueError(f"{name} agreement differs from {base_name}")
    return frames


def agreement_bin(agreement: pd.Series) -> pd.Series:
    """Map continuous agreement to the three reported values 1.0, 0.667, 0.333."""
    a = agreement.astype(float)
    out = pd.Series(np.nan, index=a.index, dtype=float)
    out[(a - 1.0).abs() < 1e-6] = 1.0
    out[(a - 2 / 3).abs() < 1e-2] = 0.667
    out[(a - 1 / 3).abs() < 1e-2] = 0.333
    return out


def negation_mask(text: pd.Series) -> pd.Series:
    return text.astype(str).str.lower().str.contains(NEGATION_RE, regex=True)


def sarcasm_mask(text: pd.Series) -> pd.Series:
    return text.astype(str).str.lower().str.contains(SARCASM_RE, regex=True)


def news_style_mask(text: pd.Series) -> pd.Series:
    """URL token present, no first/second person pronouns, no ? or !."""
    t = text.astype(str)
    lower = t.str.lower()
    has_url = lower.str.contains(URL_TOKEN_RE, regex=True)
    has_person = lower.str.contains(PERSON_RE, regex=True)
    has_punct = t.str.contains(r"[?!]")
    return has_url & ~has_person & ~has_punct


def n_models_wrong(frames: Dict[str, pd.DataFrame],
                   model_names: Sequence[str] = MODEL_NAMES) -> pd.Series:
    wrong = np.zeros(len(next(iter(frames.values()))), dtype=int)
    for name in model_names:
        df = frames[name]
        wrong += (df["y_pred"].to_numpy() != df["y_true"].to_numpy()).astype(int)
    return pd.Series(wrong, name="n_wrong")


def metrics_from_pred(df: pd.DataFrame) -> dict:
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    y_score = df["y_score"].to_numpy()
    p, r, f, _ = __import__("sklearn.metrics", fromlist=["precision_recall_fscore_support"]).precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_score))),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_negative": float(f[0]),
        "f1_neutral": float(f[1]),
        "f1_positive": float(f[2]),
    }


def metrics_on_mask(df: pd.DataFrame, mask: np.ndarray) -> dict:
    sub = df.loc[mask]
    if len(sub) == 0:
        return {"n": 0, "macro_f1": np.nan, "accuracy": np.nan}
    m = metrics_from_pred(sub)
    return {"n": int(len(sub)), "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}


def agreement_metrics_table(frames: Dict[str, pd.DataFrame],
                            model_names: Sequence[str] = MODEL_NAMES) -> pd.DataFrame:
    base = frames[model_names[0]]
    bins = agreement_bin(base["agreement"])
    rows = []
    for b in [1.0, 0.667, 0.333]:
        mask = (bins == b).to_numpy()
        for name in model_names:
            m = metrics_on_mask(frames[name], mask)
            rows.append(dict(agreement=b, model=name, n=int(mask.sum()),
                             macro_f1=m["macro_f1"], accuracy=m["accuracy"]))
    return pd.DataFrame(rows)


def slice_metrics_table(frames: Dict[str, pd.DataFrame],
                        slices: Dict[str, np.ndarray],
                        model_names: Sequence[str] = MODEL_NAMES) -> pd.DataFrame:
    rows = []
    for name in model_names:
        overall = metrics_from_pred(frames[name])
        rows.append(dict(slice="overall", model=name, n=len(frames[name]),
                         macro_f1=overall["macro_f1"], accuracy=overall["accuracy"]))
        for sl_name, mask in slices.items():
            m = metrics_on_mask(frames[name], mask)
            rows.append(dict(slice=sl_name, model=name, n=m["n"],
                             macro_f1=m["macro_f1"], accuracy=m["accuracy"]))
    return pd.DataFrame(rows)


def error_directions(df: pd.DataFrame) -> pd.DataFrame:
    """Row-normalised confusion: share of each true class predicted as each label."""
    cm = confusion_matrix(df["y_true"], df["y_pred"], labels=LABELS).astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    norm = cm / row_sum
    rows = []
    for i, true_name in enumerate(LABEL_NAMES):
        for j, pred_name in enumerate(LABEL_NAMES):
            if i == j:
                continue
            rows.append(dict(true=true_name, pred=pred_name,
                             rate=norm[i, j], count=int(cm[i, j])))
    return pd.DataFrame(rows).sort_values("rate", ascending=False)


def most_confident_errors(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    err = df[df["y_pred"] != df["y_true"]].copy()
    err["abs_err"] = (err["y_score"] - err["y_true"]).abs()
    return err.sort_values("abs_err", ascending=False).head(k)


def hard_examples(frames: Dict[str, pd.DataFrame], mask: np.ndarray,
                  model_names: Sequence[str] = MODEL_NAMES, k: int = 6,
                  min_wrong: int = 3) -> Tuple[pd.DataFrame, int]:
    """Return up to k hard examples and the total count with >= min_wrong models wrong."""
    n_wrong = n_models_wrong({n: frames[n].loc[mask].reset_index(drop=True)
                              for n in model_names}, model_names)
    idx = np.flatnonzero(mask)
    hard_pos = np.flatnonzero(n_wrong.to_numpy() >= min_wrong)
    total = int(len(hard_pos))
    keep_pos = hard_pos[:k]
    rows = []
    for pos in keep_pos:
        i = idx[pos]
        row = {
            "tweet_id": frames[model_names[0]].at[i, "tweet_id"],
            "text": frames[model_names[0]].at[i, "text"],
            "y_true": int(frames[model_names[0]].at[i, "y_true"]),
            "agreement": float(frames[model_names[0]].at[i, "agreement"]),
            "n_wrong": int(n_wrong.iloc[pos]),
        }
        for name in model_names:
            row[name] = int(frames[name].at[i, "y_pred"])
        rows.append(row)
    return pd.DataFrame(rows), total


def sample_all_wrong_for_coding(frames: Dict[str, pd.DataFrame],
                                n: int = 30, seed: int = 42,
                                model_names: Sequence[str] = MODEL_NAMES) -> pd.DataFrame:
    """Stratified sample of tweets that all models get wrong; empty manual columns."""
    wrong = n_models_wrong(frames, model_names)
    base = frames[model_names[0]]
    pool = base.loc[wrong.to_numpy() == len(model_names)].copy()
    for name in model_names:
        pool[name] = frames[name].loc[pool.index, "y_pred"].to_numpy()
    rng = np.random.default_rng(seed)
    labels = sorted(pool["y_true"].unique().tolist())
    # proportional allocation, then fill remainder so the sample size is exactly n
    sizes = {lab: int(np.floor(n * (pool["y_true"] == lab).sum() / len(pool)))
             for lab in labels}
    while sum(sizes.values()) < n:
        remainders = {
            lab: (pool["y_true"] == lab).sum() / len(pool) - sizes[lab] / n
            for lab in labels
            if sizes[lab] < (pool["y_true"] == lab).sum()
        }
        if not remainders:
            break
        lab = max(remainders, key=remainders.get)
        sizes[lab] += 1
    parts = []
    for lab in labels:
        sub = pool[pool["y_true"] == lab]
        take = min(sizes[lab], len(sub))
        if take == 0:
            continue
        chosen = sub.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
        parts.append(chosen)
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    cols = ["tweet_id", "text", "y_true", "agreement"] + list(model_names)
    out = out[cols].copy()
    out["manual_category"] = ""
    out["notes"] = ""
    return out


def negative_confusion_rates(frames: Dict[str, pd.DataFrame],
                             model_names: Sequence[str] = MODEL_NAMES) -> pd.DataFrame:
    """Row-normalised rates from true negative to neutral and to positive."""
    rows = []
    for name in model_names:
        df = frames[name]
        cm = confusion_matrix(df["y_true"], df["y_pred"], labels=LABELS).astype(float)
        row = cm[0]
        denom = row.sum() if row.sum() else 1.0
        rows.append(dict(
            model=name,
            n_negative=int(row.sum()),
            neg_to_neu=row[1] / denom,
            neg_to_pos=row[2] / denom,
        ))
    return pd.DataFrame(rows)


def bootstrap_metrics(df: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> dict:
    """Bootstrap means and 95% intervals for macro-F1, RMSE, accuracy and per-class F1."""
    rng = np.random.default_rng(seed)
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    y_score = df["y_score"].to_numpy()
    n = len(df)
    f1s, rmses, accs = [], [], []
    f1_neg, f1_neu, f1_pos = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp, ys = y_true[idx], y_pred[idx], y_score[idx]
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        rmses.append(float(np.sqrt(mean_squared_error(yt, ys))))
        accs.append(accuracy_score(yt, yp))
        _, _, f, _ = __import__(
            "sklearn.metrics", fromlist=["precision_recall_fscore_support"]
        ).precision_recall_fscore_support(yt, yp, labels=LABELS, zero_division=0)
        f1_neg.append(f[0]); f1_neu.append(f[1]); f1_pos.append(f[2])
    def ci(a):
        a = np.asarray(a, dtype=float)
        return float(a.mean()), float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))
    out = {}
    for key, arr in [
        ("macro_f1", f1s), ("rmse", rmses), ("accuracy", accs),
        ("f1_negative", f1_neg), ("f1_neutral", f1_neu), ("f1_positive", f1_pos),
    ]:
        mean, lo, hi = ci(arr)
        out[f"{key}_mean"] = mean
        out[f"{key}_lo"] = lo
        out[f"{key}_hi"] = hi
    point = metrics_from_pred(df)
    out.update({f"{k}_point": v for k, v in point.items()})
    return out


def paired_bootstrap_f1_diff(df_a: pd.DataFrame, df_b: pd.DataFrame,
                             n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired bootstrap of macro-F1(A) - macro-F1(B) on the same resampled tweets."""
    rng = np.random.default_rng(seed)
    n = len(df_a)
    ya, pa = df_a["y_true"].to_numpy(), df_a["y_pred"].to_numpy()
    yb, pb = df_b["y_true"].to_numpy(), df_b["y_pred"].to_numpy()
    # y_true must match
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fa = f1_score(ya[idx], pa[idx], average="macro", zero_division=0)
        fb = f1_score(yb[idx], pb[idx], average="macro", zero_division=0)
        diffs.append(fa - fb)
    diffs = np.asarray(diffs, dtype=float)
    return {
        "mean_diff": float(diffs.mean()),
        "lo": float(np.quantile(diffs, 0.025)),
        "hi": float(np.quantile(diffs, 0.975)),
        "share_le_0": float((diffs <= 0).mean()),
    }


def plot_agreement_heatmap(table: pd.DataFrame, path: str, value: str = "macro_f1"):
    pivot = table.pivot(index="model", columns="agreement", values=value)
    pivot = pivot[[1.0, 0.667, 0.333]]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", ax=ax)
    ax.set_title(f"{value} by annotator agreement")
    ax.set_xlabel("agreement")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.show()
    plt.close()


def plot_slice_bars(table: pd.DataFrame, path: str,
                    model_names: Sequence[str] = MODEL_NAMES):
    sub = table[table["slice"] != "overall"].copy()
    overall = table[table["slice"] == "overall"].set_index("model")["macro_f1"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    slices = list(dict.fromkeys(sub["slice"]))
    x = np.arange(len(model_names))
    width = 0.2
    for i, sl in enumerate(slices):
        vals = [sub[(sub.model == m) & (sub.slice == sl)]["macro_f1"].values[0]
                for m in model_names]
        ax.bar(x + (i - 1) * width, vals, width, label=sl)
    ax.plot(x, [overall.loc[m] for m in model_names], "ko-", label="overall")
    ax.set_xticks(x, model_names, rotation=20)
    ax.set_ylabel("macro-F1")
    ax.set_title("Macro-F1 on heuristic slices")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.show()
    plt.close()


def plot_confusion_grid(frames: Dict[str, pd.DataFrame], path: str,
                        model_names: Sequence[str] = MODEL_NAMES):
    fig, axes = plt.subplots(1, len(model_names), figsize=(3.2 * len(model_names), 3.4))
    if len(model_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        df = frames[name]
        cm = confusion_matrix(df["y_true"], df["y_pred"], labels=LABELS).astype(float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.show()
    plt.close()


def plot_learning_curve_panel(paths: Dict[str, str], out_path: str):
    """paths: label -> image path. Missing files are skipped and reported."""
    present = {k: v for k, v in paths.items() if os.path.exists(v)}
    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if not present:
        print("No learning-curve images found; skipped panel.")
        return missing
    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, (label, path) in zip(axes, present.items()):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    plt.close()
    return missing


def plot_macro_f1_ci(final_table: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(final_table))
    ax.bar(x, final_table["macro_f1_mean"],
           yerr=[final_table["macro_f1_mean"] - final_table["macro_f1_lo"],
                 final_table["macro_f1_hi"] - final_table["macro_f1_mean"]],
           capsize=4, color="#4c72b0")
    ax.set_xticks(x, final_table["model"], rotation=20)
    ax.set_ylabel("macro-F1")
    ax.set_title("Test macro-F1 with 95% bootstrap intervals")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.show()
    plt.close()


def final_table_to_tex(df: pd.DataFrame, path: str):
    cols = ["model", "macro_f1_mean", "macro_f1_lo", "macro_f1_hi",
            "rmse_mean", "rmse_lo", "rmse_hi", "accuracy_point",
            "f1_negative_point", "f1_neutral_point", "f1_positive_point"]
    sub = df[cols].copy()
    lines = [
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\hline",
        r"model & F1 & F1 lo & F1 hi & RMSE & RMSE lo & RMSE hi & acc & F1- & F1 0 & F1+ \\",
        r"\hline",
    ]
    for _, r in sub.iterrows():
        lines.append(
            f"{r['model']} & {r['macro_f1_mean']:.4f} & {r['macro_f1_lo']:.4f} & {r['macro_f1_hi']:.4f} & "
            f"{r['rmse_mean']:.4f} & {r['rmse_lo']:.4f} & {r['rmse_hi']:.4f} & "
            f"{r['accuracy_point']:.4f} & {r['f1_negative_point']:.4f} & "
            f"{r['f1_neutral_point']:.4f} & {r['f1_positive_point']:.4f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    with open(path, "w") as f:
        f.write("\n".join(lines))
