"""Shared PyTorch utilities for the neural sequence models (BiLSTM, Transformer).

Used by notebooks 03 (BiLSTM + GloVe) and 04 (Transformer from scratch). Every model:
  - builds its vocabulary on the training split only,
  - early-stops on validation macro-F1,
  - reports y_score = expected label (softmax @ [-1, 0, 1]) for RMSE, like the baselines.
Runs on CPU or GPU (DEVICE is picked automatically).
"""
import collections
import copy
import os
import random
import re
import time
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.evaluation import score

MAX_LEN = 40                 # cleaned tweets: 99th pct 27 tokens, max 36
PAD, UNK = 0, 1
CLASSES = np.array([-1, 0, 1])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLOVE_URLS = [                   # same file; the Hugging Face mirror is usually much faster
    "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.twitter.27B.zip",
    "https://nlp.stanford.edu/data/glove.twitter.27B.zip",
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- tokenisation
TOKEN_RE = re.compile(r"[a-z0-9_']+|[!?]")


def tokenize(text):
    """Split clean_text into words, with ! and ? as their own tokens ("kids?" -> "kids", "?").

    clean_text keeps ! and ? attached to words, which would give "kids", "kids?" and "kids!"
    separate embeddings (1,180 such types in train) and none of them a GloVe vector.
    """
    return TOKEN_RE.findall(str(text))


class Vocab:
    """Word vocabulary built on the training split only (ids 0/1 = <pad>/<unk>)."""

    def __init__(self, texts, min_freq=2):
        counts = collections.Counter(w for t in texts for w in tokenize(t))
        self.itos = ["<pad>", "<unk>"] + [w for w, c in counts.most_common() if c >= min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len=MAX_LEN):
        return [self.stoi.get(w, UNK) for w in tokenize(text)][:max_len]


class BPEVocab:
    """Byte-pair-encoding subword vocabulary trained on the training split only."""

    def __init__(self, texts, vocab_size=5000):
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        self.tok = Tokenizer(models.BPE(unk_token="<unk>"))
        self.tok.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                      special_tokens=["<pad>", "<unk>"])
        self.tok.train_from_iterator([str(t) for t in texts], trainer)

    def __len__(self):
        return self.tok.get_vocab_size()

    def encode(self, text, max_len=64):
        return self.tok.encode(str(text)).ids[:max_len]


# ---------------------------------------------------------------- data loading
class TweetDataset(Dataset):
    def __init__(self, df, vocab, max_len=MAX_LEN, text_col="clean_text"):
        self.ids = [vocab.encode(t, max_len) or [UNK] for t in df[text_col]]
        self.y = df["label_id"].tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.ids[i], self.y[i]


def collate(batch):
    seqs, ys = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    x = torch.full((len(seqs), int(lengths.max())), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x, lengths, torch.tensor(ys)


def make_datasets(train, val, test, vocab, max_len=MAX_LEN):
    return tuple(TweetDataset(df, vocab, max_len) for df in (train, val, test))


def make_loaders(datasets, batch_size=64, seed=42):
    """Train loader is shuffled with its own seeded generator, so every run sees the same order."""
    g = torch.Generator().manual_seed(seed)
    tr, va, te = datasets
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=g),
            DataLoader(va, batch_size=256, shuffle=False, collate_fn=collate),
            DataLoader(te, batch_size=256, shuffle=False, collate_fn=collate))


def class_weights(train):
    """Inverse-frequency weights n / (k * n_c), same as sklearn's 'balanced'."""
    counts = np.bincount(train["label_id"], minlength=3)
    return torch.tensor(len(train) / (3 * counts), dtype=torch.float)


# ---------------------------------------------------------------- GloVe
def glove_keys(w):
    """GloVe-Twitter spellings to try for a word: as-is, without apostrophes, <number>.

    GloVe-Twitter has no "don't"/"can't"/"isn't" but does have "dont"/"cant"/"isnt",
    and it replaced every number with <number> before training.
    """
    keys = [w]
    if "'" in w:
        keys.append(w.replace("'", ""))
    if w.isdigit():
        keys.append("<number>")
    return keys


def download_glove(dim, glove_dir="/content/glove"):
    """Download glove.twitter.27B.zip once and extract the file for `dim`. Returns its path."""
    os.makedirs(glove_dir, exist_ok=True)
    name = f"glove.twitter.27B.{dim}d.txt"
    path = os.path.join(glove_dir, name)
    if os.path.exists(path):
        return path
    zpath = os.path.join(glove_dir, "glove.twitter.27B.zip")
    if not os.path.exists(zpath):
        part = zpath + ".part"                  # renamed only when complete, so a crash never leaves a bad zip
        for url in GLOVE_URLS:
            try:
                print("Downloading", url, "(~1.5 GB)")
                urllib.request.urlretrieve(url, part)
                os.replace(part, zpath)
                break
            except Exception as e:
                print("  failed:", e)
                if os.path.exists(part):
                    os.remove(part)
        else:
            raise RuntimeError("Could not download GloVe-Twitter from any mirror")
    with zipfile.ZipFile(zpath) as z:
        z.extract(name, glove_dir)
    return path


def read_glove(path, dim, words=None):
    """Read GloVe vectors (optionally only for `words`) into {word: vector}."""
    vecs = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) == dim + 1 and (words is None or parts[0] in words):
                vecs[parts[0]] = np.asarray(parts[1:], dtype="float32")
    return vecs


def load_glove_subset(dim, texts, cache_dir="data/embeddings", glove_dir="/content/glove"):
    """GloVe vectors for every spelling of every word in `texts`.

    The small subset is cached (on Drive in Colab) so reruns skip the 1.5 GB download.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"glove_twitter_{dim}d_subset_v2.txt")
    if not os.path.exists(cache):
        words = {k for t in texts for w in tokenize(t) for k in glove_keys(w)}
        vecs = read_glove(download_glove(dim, glove_dir), dim, words=words)
        with open(cache, "w", encoding="utf-8") as f:
            for w, v in vecs.items():
                f.write(w + " " + " ".join(f"{x:.5f}" for x in v) + "\n")
    return read_glove(cache, dim)


def glove_lookup(w, vecs):
    for k in glove_keys(w):
        if k in vecs:
            return vecs[k]
    return None


def embedding_matrix(vocab, vecs, dim, seed=42):
    """Rows for words found in GloVe; others ~ N(mean, std) of the GloVe vectors; <pad> = 0."""
    M = np.stack(list(vecs.values()))
    E = np.random.default_rng(seed).normal(M.mean(), M.std(), (len(vocab), dim)).astype("float32")
    E[PAD] = 0
    for w, i in vocab.stoi.items():
        v = glove_lookup(w, vecs)
        if v is not None:
            E[i] = v
    return E


def glove_coverage(vocab, vecs, texts):
    types = vocab.itos[2:]
    tokens = [w for t in texts for w in tokenize(t)]
    found = {w for w in set(types) | set(tokens) if glove_lookup(w, vecs) is not None}
    return {"vocab_size": len(types),
            "vocab_coverage": float(np.mean([w in found for w in types])),
            "token_coverage": float(np.mean([w in found for w in tokens])),
            "top_missing": [w for w in types if w not in found][:30]}


# ---------------------------------------------------------------- training
def train_epoch(model, loader, loss_fn, optimizer, scheduler=None, clip=1.0):
    model.train()
    total, correct, n = 0.0, 0, 0
    for x, lengths, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x, lengths)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total / n, correct / n


@torch.no_grad()
def predict(model, loader, loss_fn=None):
    """Returns (y_pred in {-1,0,1}, y_score = expected label in [-1,1], class probabilities),
    plus the mean loss if loss_fn is given."""
    model.eval()
    probs, total, n = [], 0.0, 0
    for x, lengths, y in loader:
        logits = model(x.to(DEVICE), lengths)
        if loss_fn is not None:
            total += loss_fn(logits, y.to(DEVICE)).item() * len(y)
            n += len(y)
        probs.append(torch.softmax(logits, 1).cpu().numpy())
    P = np.concatenate(probs)
    out = (CLASSES[P.argmax(1)], P @ CLASSES, P)
    return out + (total / n,) if loss_fn is not None else out


def fit(model, train_dl, val_dl, y_val, epochs=20, lr=1e-3, weight_decay=0.0, class_weights=None,
        patience=5, clip=1.0, warmup=0.0, verbose=True):
    """Train with AdamW; early-stop on validation macro-F1 and restore the best epoch.

    warmup > 0: linear warm-up over that fraction of all steps, then linear decay (for Transformers).
    Returns (history, best) where history has the keys plot_history() expects.
    """
    model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=None if class_weights is None else class_weights.to(DEVICE))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=weight_decay)
    scheduler = None
    if warmup > 0:
        total = epochs * len(train_dl)
        warm = max(1, int(warmup * total))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: min((s + 1) / warm, max(0.0, (total - s) / (total - warm))))

    y_val = np.asarray(y_val)
    history = {k: [] for k in ["loss", "val_loss", "acc", "val_acc", "val_macro_f1", "val_rmse"]}
    best = {"val_macro_f1": -1.0}
    best_state, bad = None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_dl, loss_fn, optimizer, scheduler, clip)
        yp, ys, _, va_loss = predict(model, val_dl, loss_fn)
        va_acc = float((yp == y_val).mean())
        f1, rmse = score(y_val, yp, ys)
        for k, v in zip(history, [tr_loss, va_loss, tr_acc, va_acc, f1, rmse]):
            history[k].append(v)
        if verbose:
            print(f"ep {ep:2d} | loss {tr_loss:.3f}/{va_loss:.3f} | acc {tr_acc:.3f}/{va_acc:.3f} "
                  f"| val F1 {f1:.4f} RMSE {rmse:.4f} | {time.time() - t0:.1f}s")
        if f1 > best["val_macro_f1"]:
            best = {"best_epoch": ep, "val_macro_f1": f1, "val_rmse": rmse}
            best_state, bad = copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return history, best


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- analysis
NEGATION_RE = r"\b(?:not|no|never|don't|dont|doesn't|isn't|won't|can't|without)\b"   # same as the EDA


def slice_report(model_names, pred_dir="reports/predictions"):
    """Test macro-F1 of each model on slices that probe word order (negation) and label noise (agreement)."""
    import pandas as pd
    from sklearn.metrics import f1_score
    rows = []
    for name in model_names:
        p = pd.read_csv(f"{pred_dir}/{name}.csv")
        neg = p["text"].str.lower().str.contains(NEGATION_RE, regex=True)
        high = p["agreement"] >= 0.7
        for sl, m in [("all", np.ones(len(p), bool)), ("with negation", neg), ("no negation", ~neg),
                      ("agreement >= 0.7", high), ("agreement < 0.7", ~high)]:
            rows.append(dict(slice=f"{sl} (n={int(m.sum())})", model=name,
                             macro_f1=f1_score(p["y_true"][m], p["y_pred"][m], average="macro")))
    return pd.DataFrame(rows).pivot(index="slice", columns="model", values="macro_f1")[list(model_names)]


class ShuffledDataset(Dataset):
    """The same tweets with the words of each one in a random order (labels unchanged)."""

    def __init__(self, ds, seed=0):
        rng = random.Random(seed)
        self.ids = [rng.sample(s, len(s)) for s in ds.ids]
        self.y = ds.y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.ids[i], self.y[i]


def shuffle_test(model, ds, y_true, n_shuffles=5, masks=None):
    """Word-order test: macro-F1 on the original tweets vs on n_shuffles copies with shuffled words.

    A model that ignores word order loses nothing; a model that uses it drops. `masks` adds
    slices ({name: boolean array}) that are scored separately. Returns one row per slice.
    """
    import pandas as pd
    from sklearn.metrics import f1_score
    y_true = np.asarray(y_true)
    masks = {"all": np.ones(len(y_true), bool), **(masks or {})}

    def slice_f1(d):
        yp = predict(model, DataLoader(d, batch_size=256, shuffle=False, collate_fn=collate))[0]
        return {k: f1_score(y_true[m], yp[m], average="macro") for k, m in masks.items()}

    orig = slice_f1(ds)
    shuffled = [slice_f1(ShuffledDataset(ds, seed=s)) for s in range(n_shuffles)]
    rows = []
    for k, m in masks.items():
        v = np.array([s[k] for s in shuffled])
        rows.append(dict(slice=k, n=int(m.sum()), original=orig[k], shuffled=v.mean(),
                         shuffled_std=v.std(), drop=orig[k] - v.mean()))
    return pd.DataFrame(rows)
