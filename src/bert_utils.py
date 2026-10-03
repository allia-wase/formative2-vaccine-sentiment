"""BERTweet fine-tuning helpers for Model 5.

Builds on the shared preprocessing, evaluation and Experiments runner.
"""
from __future__ import annotations

import copy
import gc
import logging
import os
import re
import time
import warnings
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.evaluation import log_run, score
from src.experiments import Experiments
from src.nn_utils import DEVICE, n_params, set_seed

try:
    from IPython.display import display
except ImportError:
    display = print

BERT_UTILS_VERSION = 3

# Quiet Hugging Face download bars and model-load chatter.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch(\.|$)")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch(\.|$)")


def _quiet_hf():
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass


_quiet_hf()


def _amp_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _amp_autocast(enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (TypeError, AttributeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def format_lr_for_display(value) -> str:
    """Readable learning-rate label for tables (display only)."""
    v = float(value)
    if abs(v) >= 1e-3:
        return f"{v:g}"
    return f"{v:.0e}"

CLASSES = np.array([-1, 0, 1], dtype=np.float64)
BERTWEET_NAME = "vinai/bertweet-base"
COVID_BERT_NAME = "digitalepidemiologylab/covid-twitter-bert-v2"
MAX_LEN = 80
USER_RE = re.compile(r"<user>", flags=re.I)
URL_RE = re.compile(r"<url>", flags=re.I)


def to_bertweet_raw(text: str) -> str:
    """Map Zindi `<user>` / `<url>` to `@USER` / `HTTPURL` as in BERTweet pretraining."""
    text = USER_RE.sub("@USER", str(text))
    text = URL_RE.sub("HTTPURL", text)
    return text


def texts_for_variant(df, variant: str):
    """Return model inputs for variant `raw` or `clean`."""
    if variant == "raw":
        return df["safe_text"].map(to_bertweet_raw)
    if variant == "clean":
        return df["clean_text"].astype(str)
    raise ValueError(f"unknown text variant: {variant!r}")


def token_length_stats(tokenizer, texts, max_len: Optional[int] = None) -> Dict[str, float]:
    """Token-length percentiles with no truncation."""
    lengths = []
    for t in texts:
        ids = tokenizer.encode(str(t), add_special_tokens=True, truncation=False)
        lengths.append(len(ids))
    arr = np.asarray(lengths, dtype=np.float64)
    out = {
        "n": float(len(arr)),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }
    if max_len is not None:
        out["frac_over_max_len"] = float((arr > max_len).mean())
    return out


class BertStanceDataset(Dataset):
    """Tokenise texts once; the collate function pads each batch to its own max length."""

    def __init__(self, texts, label_ids, tokenizer, max_len: int = MAX_LEN):
        self.label_ids = list(map(int, label_ids))
        self.encodings = [
            tokenizer(
                str(t),
                truncation=True,
                max_length=max_len,
                padding=False,
                return_attention_mask=True,
            )
            for t in texts
        ]

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, i):
        enc = self.encodings[i]
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": self.label_ids[i],
        }


def _pad_batch(batch, pad_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attention_mask, labels = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = max_len - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        attention_mask.append(b["attention_mask"] + [0] * pad)
        labels.append(b["labels"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def make_bert_loaders(train_ds, val_ds, test_ds=None, tokenizer=None, batch_size: int = 16, seed: int = 42):
    """Return train, val and optional test DataLoaders with dynamic padding."""
    if tokenizer is None:
        raise TypeError("tokenizer is required")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate(batch):
        return _pad_batch(batch, pad_id)

    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=g)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate)
    test_dl = (
        None if test_ds is None
        else DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate)
    )
    return train_dl, val_dl, test_dl


def load_tokenizer(model_name: str = BERTWEET_NAME, **kwargs):
    from transformers import AutoTokenizer
    # BERTweet's tweet normaliser needs normalization=True (and the emoji package).
    if model_name == BERTWEET_NAME and "normalization" not in kwargs:
        kwargs["normalization"] = True
    return AutoTokenizer.from_pretrained(model_name, **kwargs)


def build_model(model_name: str = BERTWEET_NAME, freeze_encoder: bool = False, num_labels: int = 3):
    from transformers import AutoConfig, AutoModelForSequenceClassification
    _quiet_hf()
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    if freeze_encoder:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False
    return model


def build_model_from_config(config, freeze_encoder: bool = False):
    """Build a randomly initialised sequence classifier from a config object."""
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_config(config)
    if freeze_encoder:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False
    return model


def _move_batch(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}


@torch.no_grad()
def predict_bert(model, loader, loss_fn=None):
    """Return y_pred in {-1,0,1}, y_score = softmax @ [-1,0,1], class probs, and optional mean loss."""
    model.eval()
    probs, total, n = [], 0.0, 0
    use_amp = DEVICE.type == "cuda"
    for batch in loader:
        batch = _move_batch(batch)
        labels = batch["labels"]
        with _amp_autocast(use_amp):
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = out.logits
            if loss_fn is not None:
                total += loss_fn(logits, labels).item() * len(labels)
                n += len(labels)
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
    P = np.concatenate(probs)
    yp = CLASSES[P.argmax(1)].astype(int)
    ys = P @ CLASSES
    if loss_fn is not None:
        return yp, ys, P, total / max(n, 1)
    return yp, ys, P


def fit_bert(
    model,
    train_dl,
    val_dl,
    y_val,
    epochs: int = 5,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    class_weight_tensor=None,
    patience: int = 2,
    clip: float = 1.0,
    warmup: float = 0.1,
    verbose: bool = True,
):
    """Train with AdamW, linear warmup then decay, fp16 on CUDA, grad clip, early stop on val macro-F1."""
    model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_dl))
    warm_steps = max(1, int(warmup * total_steps)) if warmup > 0 else 0

    def lr_lambda(step):
        if warm_steps <= 0:
            return max(0.0, (total_steps - step) / max(1, total_steps))
        if step < warm_steps:
            return float(step + 1) / float(warm_steps)
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warm_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    weight = None if class_weight_tensor is None else class_weight_tensor.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    use_amp = DEVICE.type == "cuda"
    scaler = _amp_scaler(use_amp)

    y_val = np.asarray(y_val)
    history = {k: [] for k in ["loss", "val_loss", "acc", "val_acc", "val_macro_f1", "val_rmse"]}
    best = {"val_macro_f1": -1.0}
    best_state, bad = None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total, correct, n = 0.0, 0, 0
        for batch in train_dl:
            batch = _move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with _amp_autocast(use_amp):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = loss_fn(out.logits, batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(batch["labels"])
            correct += (out.logits.argmax(1) == batch["labels"]).sum().item()
            n += len(batch["labels"])
        tr_loss, tr_acc = total / max(n, 1), correct / max(n, 1)

        yp, ys, _, va_loss = predict_bert(model, val_dl, loss_fn)
        va_acc = float((yp == y_val).mean())
        f1, rmse = score(y_val, yp, ys)
        for k, v in zip(history, [tr_loss, va_loss, tr_acc, va_acc, f1, rmse]):
            history[k].append(v)
        if verbose:
            print(
                f"ep {ep:2d} | loss {tr_loss:.3f}/{va_loss:.3f} | acc {tr_acc:.3f}/{va_acc:.3f} "
                f"| val F1 {f1:.4f} RMSE {rmse:.4f} | {time.time() - t0:.1f}s"
            )
        if f1 > best["val_macro_f1"]:
            best = {"best_epoch": ep, "val_macro_f1": float(f1), "val_rmse": float(rmse)}
            best_state, bad = copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return history, best


def ckpt_path_for(cfg: dict, seed: int, ckpt_dir: str) -> str:
    """Return a checkpoint path that encodes the config and seed."""
    os.makedirs(ckpt_dir, exist_ok=True)
    text = cfg.get("text", "raw")
    lr = cfg.get("lr", 0.0)
    cw = int(bool(cfg.get("class_weight", False)))
    freeze = int(bool(cfg.get("freeze", False)))
    tag = f"seed{seed}_{text}_lr{lr:g}_cw{cw}_fr{freeze}"
    return os.path.join(ckpt_dir, f"{tag}.pt")


def free_model(model=None):
    """Delete a model reference and clear the CUDA cache."""
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class BertExperiments(Experiments):
    """Experiments runner that frees each model after training and saves seed-42 weights to disk.

    Cache entries are (row, checkpoint path or None, history). load_cached_model() reloads
    the saved weights for the final evaluation.
    """

    def __init__(
        self,
        train_fn,
        model_name,
        defaults,
        seeds=(42, 1, 2),
        author="",
        ckpt_dir="models/bertweet_ckpts",
        log_path="reports/experiment_log.csv",
        save_seeds: Sequence[int] = (42,),
    ):
        super().__init__(train_fn, model_name, defaults, seeds=seeds, author=author)
        self.ckpt_dir = ckpt_dir
        self.log_path = log_path
        self.save_seeds = tuple(save_seeds)
        os.makedirs(ckpt_dir, exist_ok=True)

    def run(self, exp, seed=42, **overrides):
        cfg = {**self.best, **overrides}
        key = (tuple(sorted(cfg.items())), seed)
        if key not in self.cache:
            set_seed(seed)
            _quiet_hf()
            model, hist, best = self.train_fn(cfg, seed)
            params = dict(exp=exp, **cfg, seed=seed, best_epoch=best["best_epoch"])
            log_run(
                self.model_name,
                params,
                best["val_macro_f1"],
                best["val_rmse"],
                author=self.author,
                log_path=self.log_path,
            )
            row = dict(
                params,
                n_params=n_params(model),
                val_macro_f1=best["val_macro_f1"],
                val_rmse=best["val_rmse"],
            )
            path = None
            if seed in self.save_seeds:
                path = ckpt_path_for(cfg, seed, self.ckpt_dir)
                torch.save({"state_dict": model.state_dict(), "cfg": cfg, "best": best}, path)
            free_model(model)
            self.cache[key] = (row, path, hist)
            print(
                f"{exp} | {cfg} | seed={seed} | val F1 {best['val_macro_f1']:.4f} | "
                f"val RMSE {best['val_rmse']:.4f} | best_epoch {best['best_epoch']}"
            )
        return self.cache[key]

    def sweep(self, exp, grid):
        """Same aggregation as Experiments.sweep; display formats lr for readability."""
        df = pd.DataFrame([self.run(exp, seed=s, **g)[0] for g in grid for s in self.seeds])
        keys = list(grid[0])
        agg = (df.groupby(keys, sort=False)
                 .agg(val_macro_f1=("val_macro_f1", "mean"), f1_std=("val_macro_f1", "std"),
                      val_rmse=("val_rmse", "mean"), rmse_std=("val_rmse", "std"),
                      best_epoch=("best_epoch", "mean"), n_params=("n_params", "first"))
                 .reset_index())
        shown = agg.copy()
        for col in shown.columns:
            if col == "lr":
                continue
            if pd.api.types.is_numeric_dtype(shown[col]):
                shown[col] = shown[col].round(4)
        if "lr" in shown.columns:
            shown["lr"] = agg["lr"].map(format_lr_for_display)
        display(shown)
        return agg

    def load_cached_model(self, cfg=None, seed=42, builder=None):
        """Reload saved weights for `cfg` (default: current best) at `seed`."""
        cfg = dict(self.best if cfg is None else cfg)
        key = (tuple(sorted(cfg.items())), seed)
        if key not in self.cache:
            raise KeyError(f"no cached run for cfg={cfg} seed={seed}")
        _, path, _ = self.cache[key]
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"no checkpoint on disk for cfg={cfg} seed={seed}")
        blob = torch.load(path, map_location="cpu")
        if builder is None:
            raise ValueError("builder(cfg) -> model is required to reload weights")
        model = builder(blob.get("cfg", cfg))
        model.load_state_dict(blob["state_dict"])
        return model, blob.get("best"), path
