"""
Learning curve for the MLP of train_cv_mlp.py: how validation performance changes with the
number of training pairs.

For every CV split (same grouped, stratified folds as train_cv_mlp.py for the same --seeds),
the training fold is subsampled to each fraction in --train-sizes (e.g. 25%, 50%, 75%, 100%),
the MLP is trained on that subset, and it is evaluated on the *full, fixed* validation fold.
So every point of the curve is measured on the same validation pairs; only the amount of
training data changes.

    * Subsets are nested (the 25% subset is part of the 50% subset, ...) and stratified on
      label x domain, and whole groups (same original text) are kept together. --repeats
      draws several different subsets per fold (and uses a different MLP initialisation
      for each draw), to average out which pairs happen to be picked.
    * The MLP is trained for a fixed --epochs on every subset, with no epoch selection on
      the validation folds, so the curve is not optimistically biased. With --eval-epochs
      (e.g. 50 100 200) it is also scored at earlier epochs, one curve per epoch, to show
      how sensitive the result is to training length. Note that with a fixed number of
      epochs a smaller subset also gets fewer gradient steps.
    * Training-set scores (on the non-augmented training subset) are shown as well: a large
      gap between training and validation means the model overfits; a validation curve that
      is still rising at 100% suggests more labelled pairs would help.
    * Metrics are pooled over the folds of one (seed, repeat), as out-of-fold scores, then
      averaged over seeds x repeats (band: +-1 std).

All embedding, feature and MLP options of train_cv_mlp.py are available (--hidden, --pca,
--swap-aug, --dropout, --weight-decay, --epochs, ...); --select-metric, --print-every and
--final-fit are ignored here.

Example
    python learning_curve_mlp.py --hidden 32 --swap-aug --epochs 100 --seeds 42 43 44 45 46 \\
        --repeats 3 --train-sizes 0.2 0.4 0.6 0.8 1.0 --output-dir results/lc_mlp_h32
    python learning_curve_mlp.py --plot-only --output-dir results/lc_mlp_h32 --per-domain

Outputs (--output-dir): learning_curve.csv (aggregated scores, the table behind the plot),
learning_curve_predictions.csv (every validation prediction), learning_curve_train.csv
(training-set scores per run) and learning_curve.png.

Requirements: as train_cv_mlp.py, plus matplotlib.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_cv_mlp import (BASE_DIR, DOMAIN_NAMES, METRICS, NUM_DOMAINS, build_parser,
                          compute_metrics, embed_pairs, fold_features, get_device, load_data,
                          make_folds, pair_features, prepare_texts, train_labels, train_mlp)

# Reference categorical palette (validated with the dataviz validator, light surface)
COLORS = {"val": "#2a78d6", "train": "#eb6834", "CoT": "#1baf7a", "Question": "#eda100"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ----------------------------------------------------------------------------
# Nested, stratified, group-aware subsets
# ----------------------------------------------------------------------------
def nested_order(df_tr, rng):
    """Order of the training rows such that every prefix is roughly stratified.

    Groups are shuffled within their stratum (label x domain of the group's first row), then
    interleaved by their relative rank in the stratum, so a prefix of any length holds about
    the same share of each stratum. Rows of a group stay together.
    """
    first = df_tr.groupby("group", sort=False).head(1)
    strata = (first["label"].astype(str) + "_" + first["domain"].astype(str)).values
    groups = first["group"].values
    keys = np.empty(len(groups))
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        perm = rng.permutation(idx)
        # relative rank in (0, 1), with a random offset so strata do not tie
        keys[perm] = (np.arange(len(perm)) + rng.uniform()) / len(perm)
    group_order = groups[np.argsort(keys, kind="stable")]
    pos = {g: i for i, g in enumerate(group_order)}
    return np.argsort(df_tr["group"].map(pos).values, kind="stable")


def subset_sizes(n_rows, fractions, min_rows):
    return {f: max(min(min_rows, n_rows), int(round(f * n_rows))) for f in fractions}


def take_prefix(df_tr, order, n_rows):
    """First n_rows rows in `order`, extended to the end of the last group."""
    groups = df_tr["group"].values[order]
    k = n_rows
    while k < len(order) and groups[k] == groups[k - 1]:
        k += 1
    return order[:k]


# ----------------------------------------------------------------------------
# Learning curve
# ----------------------------------------------------------------------------
def run_learning_curve(args, df, U, V, device):
    domain, labels = df["domain"].values, df["label"].values
    eval_epochs = set(args.eval_epochs)
    pred_rows, train_rows = [], []
    for seed in args.seeds:
        for fold, tr_idx, va_idx in make_folds(df, args.folds, seed):
            df_tr = df.iloc[tr_idx]
            sizes = subset_sizes(len(tr_idx), args.train_sizes, args.min_train)
            for rep in range(args.repeats):
                rng = np.random.default_rng([seed, fold, rep])
                order = nested_order(df_tr, rng)
                for frac in args.train_sizes:
                    sub = tr_idx[take_prefix(df_tr, order, sizes[frac])]
                    X_tr, X_va, pre = fold_features(U, V, domain, sub, va_idx, args, seed)
                    y_tr = train_labels(labels, sub, args)
                    # score the training subset too (original direction, no swap copies)
                    X_fit = pre.transform(pair_features(U[sub], V[sub]), domain[sub])
                    X_ev = np.concatenate([X_va, X_fit])
                    n_va = len(va_idx)

                    def on_epoch_end(epoch, loss, probs):
                        if epoch not in eval_epochs:
                            return
                        preds = probs.argmax(-1)
                        for j, i in enumerate(va_idx):
                            pred_rows.append({"seed": seed, "repeat": rep, "fold": fold,
                                              "train_frac": frac, "n_train": len(sub),
                                              "epoch": epoch, "row_id": int(df.index[i]),
                                              "domain": int(domain[i]), "label": int(labels[i]),
                                              "pred": int(preds[j])})
                        train_rows.append({"seed": seed, "repeat": rep, "fold": fold,
                                           "train_frac": frac, "n_train": len(sub),
                                           "epoch": epoch, "train_loss": loss,
                                           **compute_metrics(labels[sub], preds[n_va:])})

                    train_mlp(args, X_tr, y_tr, device, seed * 1000 + rep, X_ev=X_ev,
                              on_epoch_end=on_epoch_end)
            print(f"seed {seed} fold {fold}: training subsets of "
                  + ", ".join(f"{sizes[f]}" for f in args.train_sizes)
                  + f" / {len(tr_idx)} rows x {args.repeats} repeats | val {len(va_idx)}")
    return pd.DataFrame(pred_rows), pd.DataFrame(train_rows)


def aggregate(preds, train):
    """One row per (train_frac, epoch, split, subset) with mean/std of each metric."""
    rows = []
    keys = ["train_frac", "epoch"]
    for (frac, epoch), sub in preds.groupby(keys):
        n_train = sub.groupby(["seed", "repeat", "fold"])["n_train"].first().mean()
        for name, part in [("all", sub)] + [(DOMAIN_NAMES[d], sub[sub["domain"] == d])
                                            for d in range(NUM_DOMAINS)]:
            if not len(part):
                continue
            # out-of-fold metrics for each (seed, repeat), then mean/std over them
            per_run = pd.DataFrame([compute_metrics(r["label"], r["pred"])
                                    for _, r in part.groupby(["seed", "repeat"])])
            rows.append({"train_frac": frac, "n_train": n_train, "epoch": epoch,
                         "split": "validation", "subset": name, "runs": len(per_run),
                         **{f"{k}_mean": per_run[k].mean() for k in METRICS},
                         **{f"{k}_std": per_run[k].std(ddof=0) for k in METRICS}})
    for (frac, epoch), sub in train.groupby(keys):
        rows.append({"train_frac": frac, "n_train": sub["n_train"].mean(), "epoch": epoch,
                     "split": "train", "subset": "all", "runs": len(sub),
                     **{f"{k}_mean": sub[k].mean() for k in METRICS},
                     **{f"{k}_std": sub[k].std(ddof=0) for k in METRICS}})
    return pd.DataFrame(rows).sort_values(["split", "subset", "epoch", "train_frac"])


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
METRIC_LABELS = {"linear_kappa": "Linear-weighted kappa", "pearson": "Pearson r",
                 "accuracy": "Accuracy", "macro_f1": "Macro-F1", "mae": "MAE (lower is better)"}


def plot(curve, metrics, path, per_domain=False, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = sorted(curve["epoch"].unique())
    last = epochs[-1]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.2), squeeze=False,
                             facecolor=SURFACE)
    # several eval epochs: one validation line per epoch, from light to full strength
    val_alpha = dict(zip(epochs, np.linspace(0.35, 1.0, len(epochs)) if len(epochs) > 1 else [1.0]))

    def line(ax, d, metric, color, label, alpha=1.0, band=True, dashed=False, end_label=False):
        d = d.sort_values("n_train")
        x, y, s = d["n_train"].values, d[f"{metric}_mean"].values, d[f"{metric}_std"].values
        if band:
            ax.fill_between(x, y - s, y + s, color=color, alpha=0.12 * alpha, linewidth=0)
        ax.plot(x, y, color=color, alpha=alpha, linewidth=2, linestyle="--" if dashed else "-",
                marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=label)
        if end_label:  # selective direct label at the last point (main lines only)
            ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK_2)

    for ax, metric in zip(axes[0], metrics):
        ax.set_facecolor(SURFACE)
        tr = curve[(curve["split"] == "train") & (curve["epoch"] == last)]
        line(ax, tr, metric, COLORS["train"], f"Training subset (epoch {last})", band=False,
             end_label=True)
        for e in epochs:
            va = curve[(curve["split"] == "validation") & (curve["subset"] == "all")
                       & (curve["epoch"] == e)]
            label = "Validation" + (f" (epoch {e})" if len(epochs) > 1 else "")
            line(ax, va, metric, COLORS["val"], label, alpha=val_alpha[e], band=(e == last),
                 end_label=(e == last))
        if per_domain:
            for name in DOMAIN_NAMES.values():
                va = curve[(curve["split"] == "validation") & (curve["subset"] == name)
                           & (curve["epoch"] == last)]
                if len(va):
                    line(ax, va, metric, COLORS[name], f"Validation, {name}", band=False,
                         dashed=True)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11, color=INK, loc="left")
        ax.set_xlabel("Training pairs per fold", fontsize=9.5, color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8.5)
        ax.margins(x=0.08)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.01))
    if title:
        fig.suptitle(title, fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95 if title else 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def describe(args):
    hid = "x".join(map(str, args.hidden)) or "linear"
    return (f"MLP learning curve | hidden {hid} | {Path(args.embedding_model).name}"
            + (f" | PCA {args.pca}" if args.pca else "") + (" | swap-aug" if args.swap_aug else "")
            + f" | {len(args.seeds)} seeds x {args.repeats} repeats, {args.folds}-fold CV")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    p = build_parser(__doc__)
    g = p.add_argument_group("learning curve")
    g.add_argument("--train-sizes", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0],
                   help="fractions of each training fold to train on")
    g.add_argument("--repeats", type=int, default=3,
                   help="different subsets (and MLP initialisations) per fold and size")
    g.add_argument("--min-train", type=int, default=8,
                   help="smallest training subset, in pairs")
    g.add_argument("--eval-epochs", type=int, nargs="+", default=None,
                   help="epochs at which to score (default: only the last, --epochs)")
    g.add_argument("--plot-metrics", nargs="+", choices=METRICS,
                   default=["linear_kappa", "pearson"], help="one panel per metric")
    g.add_argument("--per-domain", action="store_true",
                   help="also plot the validation curve of each domain")
    g.add_argument("--plot-only", action="store_true",
                   help="redraw learning_curve.png from learning_curve.csv in --output-dir")
    args = p.parse_args()
    args.hidden = [h for h in args.hidden if h > 0]
    args.train_sizes = sorted(set(args.train_sizes))
    if not all(0 < f <= 1 for f in args.train_sizes):
        p.error("--train-sizes must be fractions in (0, 1]")
    args.eval_epochs = sorted(set(args.eval_epochs or [args.epochs]))
    if max(args.eval_epochs) > args.epochs or min(args.eval_epochs) < 1:
        p.error("--eval-epochs must lie between 1 and --epochs")
    if args.plot_only and not args.output_dir:
        p.error("--plot-only needs --output-dir")
    return args


def main():
    args = parse_args()
    if args.output_dir is None:
        hid = "x".join(map(str, args.hidden)) or "linear"
        args.output_dir = str(BASE_DIR / "results" / (
            f"lc_mlp_h{hid}_{Path(args.embedding_model).name}"
            + (f"_pca{args.pca}" if args.pca else "") + ("_swap" if args.swap_aug else "")
            + f"_ep{args.epochs}"))
    out = Path(args.output_dir)

    if args.plot_only:
        curve = pd.read_csv(out / "learning_curve.csv")
        saved = out / "args.json"
        title = describe(type("A", (), json.loads(saved.read_text()))) if saved.exists() else None
        plot(curve, args.plot_metrics, out / "learning_curve.png", args.per_domain, title)
        print(f"Redrew {out / 'learning_curve.png'}")
        return

    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    df = prepare_texts(load_data(args.data), args.keep_options)
    print(f"Device: {device} | {len(df)} rows, {df['group'].nunique()} groups")
    U, V = embed_pairs(df, args, device)
    print(f"{describe(args)}\nTraining fractions {args.train_sizes} | {args.epochs} epochs, "
          f"scored at epochs {args.eval_epochs}\n")

    preds, train = run_learning_curve(args, df, U, V, device)
    curve = aggregate(preds, train)
    preds.to_csv(out / "learning_curve_predictions.csv", index=False)
    train.to_csv(out / "learning_curve_train.csv", index=False)
    curve.to_csv(out / "learning_curve.csv", index=False)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    show = curve[(curve["subset"] == "all") & (curve["epoch"] == args.epochs)]
    cols = ["split", "train_frac", "n_train"] + [f"{m}_{s}" for m in ("linear_kappa", "pearson")
                                                 for s in ("mean", "std")]
    print(f"\nLearning curve at epoch {args.epochs} (validation = out-of-fold, mean±std over "
          "seeds x repeats):")
    print(show[cols].round(3).to_string(index=False))
    plot(curve, args.plot_metrics, out / "learning_curve.png", args.per_domain, describe(args))
    print(f"\nWrote {out / 'learning_curve.csv'} and {out / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
