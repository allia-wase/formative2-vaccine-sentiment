"""Shared evaluation, prediction format and experiment log for the whole group.

Every model must:
  1. tune on the validation split and call log_run() for each training run
  2. call evaluate() once on the test split (writes reports/results.csv + confusion matrix)
  3. call save_predictions() so the error analysis can use all five models
"""
import datetime as dt
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_squared_error, precision_recall_fscore_support)

LABELS = [-1, 0, 1]
LABEL_NAMES = ["negative", "neutral", "positive"]


def score(y_true, y_pred, y_score=None):
    """Quick metrics for tuning on validation: returns (macro_f1, rmse)."""
    f1 = f1_score(y_true, y_pred, average="macro")
    rmse = float(np.sqrt(mean_squared_error(y_true, y_score if y_score is not None else y_pred)))
    return f1, rmse


def evaluate(y_true, y_pred, model_name, y_score=None, fig_dir="reports/figures",
             results_path="reports/results.csv", show=True):
    """Final test metrics + confusion matrix + row in the shared results table.

    y_score = continuous prediction in [-1, 1] (regression output or expected value).
    RMSE uses y_score if given, otherwise the predicted class.
    """
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_score if y_score is not None else y_pred)))
    row = {
        "model": model_name,
        "rmse": rmse,
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "f1_negative": f[0], "f1_neutral": f[1], "f1_positive": f[2],
    }

    if show:
        print(f"== {model_name} | RMSE {rmse:.4f} ==")
        print(classification_report(y_true, y_pred, labels=LABELS,
                                    target_names=LABEL_NAMES, digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, fmt, title in [(axes[0], cm, "d", "Counts"), (axes[1], cm_norm, ".2f", "Row-normalised")]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", cbar=False,
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{model_name}: {title}")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/cm_{model_name}.png", dpi=200)
    if show:
        plt.show()
    plt.close()

    new = pd.DataFrame([row])
    if os.path.exists(results_path):
        old = pd.read_csv(results_path)
        old = old[old["model"] != model_name]
        new = pd.concat([old, new], ignore_index=True)
    new.round(4).to_csv(results_path, index=False)
    return row


def save_predictions(model_name, tweet_ids, texts, y_true, y_pred, agreement=None, y_score=None,
                     out_dir="reports/predictions"):
    """Shared prediction format for ALL models (used for the error analysis).

    Columns: tweet_id, text, y_true, y_pred, y_score, agreement
    """
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({
        "tweet_id": list(tweet_ids),
        "text": list(texts),
        "y_true": list(y_true),
        "y_pred": list(y_pred),
        "y_score": list(y_score) if y_score is not None else list(y_pred),
        "agreement": list(agreement) if agreement is not None else np.nan,
    }).to_csv(f"{out_dir}/{model_name}.csv", index=False)


def log_run(model_name, params, val_macro_f1, val_rmse, author="", notes="",
            log_path="reports/experiment_log.csv"):
    """Append one row per training run to the shared experiment log."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    row = pd.DataFrame([{
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "author": author,
        "model": model_name,
        "params": json.dumps(params, default=str),
        "val_macro_f1": round(val_macro_f1, 4),
        "val_rmse": round(val_rmse, 4),
        "notes": notes,
    }])
    row.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)


def plot_history(history: dict, model_name, fig_dir="reports/figures"):
    """Learning curves. history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}"""
    os.makedirs(fig_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{model_name}: loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{model_name}: accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/curves_{model_name}.png", dpi=200)
    plt.show()
