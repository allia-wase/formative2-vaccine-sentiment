"""Seeded, cached experiment runner shared by the neural-model notebooks (03, 04).

Each config is trained with several seeds and configs are compared on the mean, because a
single run's validation macro-F1 moves by about as much as the differences between configs.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import log_run
from src.nn_utils import n_params, set_seed

try:
    from IPython.display import display
except ImportError:                      # plain Python
    display = print


class Experiments:
    """train_fn(cfg, seed) -> (model, history, best) trains one config; this class does the rest.

    - run():    one config + one seed, logged with log_run(); cached, so a config that two
                experiments share is trained only once.
    - sweep():  every override in a grid x every seed, on top of the current best config.
    - update(): carry the winner forward and say whether its lead is larger than the seed noise.
    """

    def __init__(self, train_fn, model_name, defaults, seeds=(42, 1, 2), author=""):
        self.train_fn, self.model_name, self.author = train_fn, model_name, author
        self.best, self.seeds = dict(defaults), tuple(seeds)
        self.best_f1 = -1.0
        self.cache = {}

    def run(self, exp, seed=42, **overrides):
        """Returns (row, model, history) for the current best config updated with `overrides`."""
        cfg = {**self.best, **overrides}
        key = (tuple(sorted(cfg.items())), seed)
        if key not in self.cache:
            set_seed(seed)
            model, hist, best = self.train_fn(cfg, seed)
            params = dict(exp=exp, **cfg, seed=seed, best_epoch=best["best_epoch"])
            log_run(self.model_name, params, best["val_macro_f1"], best["val_rmse"], author=self.author)
            row = dict(params, n_params=n_params(model),
                       val_macro_f1=best["val_macro_f1"], val_rmse=best["val_rmse"])
            self.cache[key] = (row, model, hist)
        return self.cache[key]

    def sweep(self, exp, grid):
        """Mean and std over seeds of every config in `grid` (a list of override dicts)."""
        df = pd.DataFrame([self.run(exp, seed=s, **g)[0] for g in grid for s in self.seeds])
        keys = list(grid[0])
        agg = (df.groupby(keys, sort=False)
                 .agg(val_macro_f1=("val_macro_f1", "mean"), f1_std=("val_macro_f1", "std"),
                      val_rmse=("val_rmse", "mean"), rmse_std=("val_rmse", "std"),
                      best_epoch=("best_epoch", "mean"), n_params=("n_params", "first"))
                 .reset_index())
        display(agg.round(4))
        return agg

    def update(self, agg, keys):
        """Carry the winning values of `keys` forward if they beat the best mean so far."""
        d = agg.sort_values("val_macro_f1", ascending=False)
        top = d.iloc[0]
        if len(d) > 1:
            gap = top["val_macro_f1"] - d.iloc[1]["val_macro_f1"]
            noise = max(top["f1_std"], d.iloc[1]["f1_std"])
            print(f"Winner leads the runner-up by {gap:.4f}; seed std {noise:.4f} -> "
                  f"{'clear difference' if gap > noise else 'within seed noise'}")
        if top["val_macro_f1"] > self.best_f1:
            # read each value from its own column: a row of an all-numeric frame turns ints into floats
            values = {k: agg.at[top.name, k] for k in keys}
            self.best.update({k: v.item() if hasattr(v, "item") else v for k, v in values.items()})
            self.best_f1 = float(top["val_macro_f1"])
        print(f"BEST (mean val macro-F1 {self.best_f1:.4f}):", self.best)


def plot_sweep(agg, x, title, path, n_seeds=3, color="#4c72b0"):
    """Bar chart of mean val macro-F1 and RMSE with seed-std error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, m, s in zip(axes, ["val_macro_f1", "val_rmse"], ["f1_std", "rmse_std"]):
        pos = np.arange(len(agg))
        ax.bar(pos, agg[m], yerr=agg[s], capsize=4, color=color)
        ax.set_xticks(pos, agg[x].astype(str), rotation=15)
        ax.set_ylim(max(0, (agg[m] - agg[s]).min() - 0.03), (agg[m] + agg[s]).max() + 0.02)
        ax.set_title(f"{title}: {m} (mean ± std, {n_seeds} seeds)", fontsize=10)
        for i, v in enumerate(agg[m]):
            ax.text(i, ax.get_ylim()[0] + 0.005, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, color="white")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.show()


def history_table(hist):
    h = pd.DataFrame(hist)
    h.index = pd.RangeIndex(1, len(h) + 1, name="epoch")
    return h.round(4)
