"""
Cross-validated MLP on frozen sentence embeddings for perturbation-strength prediction.

Each text is embedded once with a frozen sentence-transformer (cached to disk). For a pair,
the input is [u, v, domain one-hot], where u = emb(text_a) (original) and v = emb(text_b)
(perturbed), and a small MLP predicts the strength 0-3:

    0: no/minimal change   1: small   2: moderately strong   3: strong

The input depends on the order of the two texts, so --swap-aug (also train on (v, u) with
the same label) is worth trying: it doubles the training rows of each fold.

--hidden sets the MLP layer sizes (e.g. --hidden 32, or --hidden 64 32 for two layers);
--hidden 0 gives a linear softmax layer with no hidden layer.

Question texts (domain 1) are cut before the word "Options", so only the question itself is
embedded (the answer options are identical in the original and the perturbed question). Pass
--keep-options to embed the full text. The setting is saved with the model and evaluate_mlp.py
applies the same rule.

Preprocessing is fitted on the training fold only: [u, v] is standardised and, with --pca N,
reduced to N components; the domain one-hot is appended afterwards (--no-domain drops it).

Cross-validation
    * All labelled pairs come from train.csv; each pair is used once, in its original
      direction (swap copies left in an older train.csv are dropped).
    * Folds come from StratifiedGroupKFold: rows sharing the same original text (`group`)
      never straddle train/validation, and folds are stratified on label x domain. A given
      seed always gives the same folds, so runs with the same --seeds are comparable.
    * The MLP is trained for a fixed number of epochs and evaluated after each one; the
      number of epochs is chosen from the fold-averaged curve (slightly optimistic, so
      final-epoch metrics are reported too). --seeds repeats the CV with other splits.
    * Metrics: linear-weighted kappa (primary), Pearson, accuracy, macro-F1, MAE; per fold,
      out-of-fold, and per domain.

Example
    python train_cv_mlp.py --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
        --hidden 32 --swap-aug --seeds 42 43 44 45 46 --final-fit
    python train_cv_mlp.py --pca 32 --hidden 16 --seeds 42 43 44 45 46

Requirements: torch, sentence-transformers, scikit-learn, pandas.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent


# ----------------------------------------------------------------------------
# Shared setup: constants, data loading, folds, metrics, summary
# ----------------------------------------------------------------------------
NUM_LABELS = 4
NUM_DOMAINS = 2
DOMAIN_NAMES = {0: "CoT", 1: "Question"}
METRICS = ["linear_kappa", "pearson", "accuracy", "macro_f1", "mae"]
HIGHER_IS_BETTER = {"linear_kappa": True, "pearson": True, "accuracy": True, "macro_f1": True,
                    "mae": False}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # kappa is undefined when only one class is present in both arrays
    kappa = (cohen_kappa_score(y_true, y_pred, weights="linear", labels=list(range(NUM_LABELS)))
             if len(set(y_true) | set(y_pred)) > 1 else float("nan"))
    # Pearson r is undefined when either side is constant (e.g. the model predicts one class)
    pearson = (float(np.corrcoef(y_true, y_pred)[0, 1])
               if y_true.std() > 0 and y_pred.std() > 0 else float("nan"))
    return {
        "linear_kappa": kappa,
        "pearson": pearson,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_LABELS)),
                             zero_division=0),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def load_data(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.dropna(subset=["text_a", "text_b", "label", "domain"])
    df["label"] = df["label"].astype(int)
    df["domain"] = df["domain"].astype(int)
    if "group" not in df.columns:
        df["group"] = df["domain"].astype(str) + "||" + df["text_a"]
    bad = ~df["label"].between(0, NUM_LABELS - 1) | ~df["domain"].between(0, NUM_DOMAINS - 1)
    if bad.any():
        raise ValueError(f"{bad.sum()} rows have out-of-range label/domain")
    return drop_swapped(df).reset_index(drop=True)


def drop_swapped(df):
    """Keep one row per unordered (text_a, text_b, domain) pair, in the original direction.

    xlsx_to_csv.py now saves original-direction pairs only, so this is normally a no-op. It
    guards against CSVs from older versions that had (text_b, text_a) swap copies included:
    left in, they would count each pair twice in validation and be swapped again by --swap-aug.
    The original direction is the row whose text_a is the group's original text.
    """
    df = df.copy()
    df["_key"] = [(d, *sorted((a, b))) for d, a, b in zip(df["domain"], df["text_a"], df["text_b"])]
    df["_orig"] = df["group"] == df["domain"].astype(str) + "||" + df["text_a"]
    df = df.sort_values("_orig", ascending=False, kind="stable")
    n0 = len(df)
    df = df.drop_duplicates("_key", keep="first").sort_index()
    if len(df) < n0:
        print(f"Dropped {n0 - len(df)} swapped duplicate rows ({n0} -> {len(df)} unique pairs); "
              "use --swap-aug to add swaps to the training folds only")
    return df.drop(columns=["_key", "_orig"])


OPTIONS_RE = re.compile(r"\bOptions\b")


def cut_options(text):
    """Question text before the word "Options" (unchanged if it does not contain it)."""
    return OPTIONS_RE.split(text, maxsplit=1)[0].strip()


def prepare_texts(df, keep_options=False):
    """Copy of df with the texts that are embedded: Question texts cut before "Options".

    Groups are left as they are (cutting creates no new duplicates in this data).
    """
    df = df.copy()
    if not keep_options:
        q = df["domain"] == 1
        n = int((df.loc[q, "text_a"].str.contains(OPTIONS_RE) |
                 df.loc[q, "text_b"].str.contains(OPTIONS_RE)).sum())
        for col in ("text_a", "text_b"):
            df.loc[q, col] = df.loc[q, col].map(cut_options)
        print(f"Question texts cut before 'Options' in {n}/{int(q.sum())} Question pairs")
    return df


def class_weights(labels, device):
    counts = np.bincount(labels, minlength=NUM_LABELS).astype(float)
    w = counts.sum() / (NUM_LABELS * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float, device=device)


def make_folds(df, n_splits, seed):
    """Yield (fold, train_idx, val_idx): grouped by `group`, stratified on label x domain."""
    strata = df["label"].astype(str) + "_" + df["domain"].astype(str)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, strata, groups=df["group"])):
        assert not set(df["group"].iloc[tr_idx]) & set(df["group"].iloc[va_idx]), \
            "group leakage between folds"
        yield fold, tr_idx, va_idx


def pooled_metrics(preds, epoch):
    """Out-of-fold metrics at one epoch: pool all folds per seed, then mean/std over seeds."""
    sub = preds[preds["epoch"] == epoch]
    per_seed = {"all": [], **{DOMAIN_NAMES[d]: [] for d in range(NUM_DOMAINS)}}
    for _, s in sub.groupby("seed"):
        per_seed["all"].append(compute_metrics(s["label"], s["pred"]))
        for d in range(NUM_DOMAINS):
            sd = s[s["domain"] == d]
            if len(sd):
                per_seed[DOMAIN_NAMES[d]].append(compute_metrics(sd["label"], sd["pred"]))
    out = {}
    for name, rows in per_seed.items():
        if rows:
            r = pd.DataFrame(rows)
            out[name] = {k: {"mean": float(r[k].mean()), "std": float(r[k].std(ddof=0))}
                         for k in METRICS}
    return out


def summarise(args, epochs_df, preds_df, print_every=1):
    curve = epochs_df.groupby("epoch")[METRICS + ["train_loss"]].agg(["mean", "std"])
    shown = curve.xs("mean", axis=1, level=1)
    shown = shown[(shown.index % print_every == 0) | (shown.index == shown.index.min())]
    print("\nFold-averaged validation curve (mean over seeds x folds):")
    print(shown.round(4).to_string())

    sel = curve[(args.select_metric, "mean")]
    if sel.isna().all():
        print(f"WARNING: {args.select_metric} is undefined at every epoch; using the final epoch")
        sel = pd.Series(0.0, index=sel.index).where(sel.index != args.epochs, 1.0)
    else:
        sel = sel.fillna(-np.inf if HIGHER_IS_BETTER[args.select_metric] else np.inf)
    best_epoch = int(sel.idxmax() if HIGHER_IS_BETTER[args.select_metric] else sel.idxmin())
    summary = {
        "selected_epoch": best_epoch,
        "select_metric": args.select_metric,
        "fold_mean_at_selected_epoch": {
            k: {"mean": float(curve.loc[best_epoch, (k, "mean")]),
                "std": float(curve.loc[best_epoch, (k, "std")])} for k in METRICS},
        "oof_at_selected_epoch": pooled_metrics(preds_df, best_epoch),
        "oof_at_final_epoch": pooled_metrics(preds_df, args.epochs),
        "args": vars(args),
    }

    def fmt(block):
        return " | ".join(f"{k} {v['mean']:.3f}±{v['std']:.3f}" for k, v in block.items())

    print(f"\nSelected epoch ({args.select_metric}): {best_epoch}")
    print(f"Per-fold mean±std @ epoch {best_epoch}: {fmt(summary['fold_mean_at_selected_epoch'])}")
    for title, key, ep in [("selected", "oof_at_selected_epoch", best_epoch),
                           ("final", "oof_at_final_epoch", args.epochs)]:
        print(f"Out-of-fold @ {title} epoch {ep} (mean±std over seeds):")
        for name, block in summary[key].items():
            print(f"  {name:9s} {fmt(block)}")
    return summary


# ----------------------------------------------------------------------------
# Embeddings (frozen, cached)
# ----------------------------------------------------------------------------
def embed_pairs(df, args, device):
    """Return (U, V): embeddings of text_a and text_b, shape (n, dim) each."""
    texts = sorted(set(df["text_a"]) | set(df["text_b"]))
    key = hashlib.sha1(json.dumps([args.embedding_model, args.max_seq_length, args.normalize,
                                   texts]).encode()).hexdigest()[:12]
    cache = Path(args.cache_dir) / f"{Path(args.embedding_model).name}_{key}.npz"
    if cache.exists():
        print(f"Loading cached embeddings from {cache}")
        data = np.load(cache, allow_pickle=True)
        lookup = dict(zip(data["texts"], data["emb"]))
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.embedding_model, device=str(device))
        if args.max_seq_length:
            model.max_seq_length = args.max_seq_length
        n_long = sum(len(model.tokenizer(t)["input_ids"]) > model.max_seq_length for t in texts)
        print(f"Embedding {len(texts)} unique texts with {args.embedding_model} "
              f"(max_seq_length {model.max_seq_length}; {n_long} texts will be truncated)")
        emb = model.encode(texts, batch_size=args.embed_batch_size, show_progress_bar=False,
                           normalize_embeddings=args.normalize, convert_to_numpy=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, texts=np.array(texts, dtype=object), emb=emb)
        lookup = dict(zip(texts, emb))
    U = np.stack([lookup[t] for t in df["text_a"]]).astype(np.float32)
    V = np.stack([lookup[t] for t in df["text_b"]]).astype(np.float32)
    return U, V


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def pair_features(U, V):
    """Pair input [u, v]: the two embeddings side by side, shape (n, 2 * dim)."""
    return np.concatenate([U, V], axis=1).astype(np.float32)


class Preprocessor:
    """Standardise (+ optional PCA) [u, v], then append the domain one-hot.

    Fitted on the training fold only.
    """

    def __init__(self, pca_components=0, use_domain=True, seed=0):
        self.pca_components, self.use_domain, self.seed = pca_components, use_domain, seed

    def fit(self, emb):
        self.emb_scaler = StandardScaler().fit(emb)
        self.pca = None
        if self.pca_components:
            n = min(self.pca_components, emb.shape[0] - 1, emb.shape[1])
            self.pca = PCA(n_components=n, random_state=self.seed).fit(self.emb_scaler.transform(emb))
        return self

    def transform(self, emb, domain):
        e = self.emb_scaler.transform(emb)
        parts = [self.pca.transform(e) if self.pca is not None else e]
        if self.use_domain:
            parts.append(np.eye(NUM_DOMAINS)[domain])
        return np.concatenate(parts, axis=1).astype(np.float32)


def fold_features(U, V, domain, tr_idx, ev_idx, args, seed):
    """Fit preprocessing on the training rows (optionally swap-augmented); transform both."""
    U_tr, V_tr, d_tr = U[tr_idx], V[tr_idx], domain[tr_idx]
    if args.swap_aug:
        U_tr, V_tr = np.concatenate([U_tr, V_tr]), np.concatenate([V_tr, U_tr])
        d_tr = np.concatenate([d_tr, d_tr])
    emb_tr = pair_features(U_tr, V_tr)
    pre = Preprocessor(args.pca, use_domain=not args.no_domain, seed=seed).fit(emb_tr)
    X_tr = pre.transform(emb_tr, d_tr)
    X_ev = None
    if ev_idx is not None:
        X_ev = pre.transform(pair_features(U[ev_idx], V[ev_idx]), domain[ev_idx])
    return X_tr, X_ev, pre


def train_labels(labels, idx, args):
    """Labels for training rows, repeated to match swap augmentation."""
    return np.concatenate([labels[idx]] * (2 if args.swap_aug else 1))


# ----------------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout, input_dropout):
        super().__init__()
        layers, d = [nn.Dropout(input_dropout)], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, NUM_LABELS))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(args, X_tr, y_tr, device, seed, X_ev=None, on_epoch_end=None):
    """Train for args.epochs; call on_epoch_end(epoch, train_loss, probs) after each."""
    set_seed(seed)
    model = MLP(X_tr.shape[1], args.hidden, args.dropout, args.input_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weight = class_weights(y_tr, device) if args.class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    X = torch.tensor(X_tr, device=device)
    y = torch.tensor(y_tr, dtype=torch.long, device=device)
    X_e = torch.tensor(X_ev, device=device) if X_ev is not None else None
    g = torch.Generator(device="cpu").manual_seed(seed)
    bs = args.batch_size or len(X)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(X), generator=g).to(device)
        total = 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            loss = loss_fn(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if on_epoch_end is not None:
            probs = None
            if X_e is not None:
                model.eval()
                with torch.no_grad():
                    probs = torch.softmax(model(X_e), dim=-1).cpu().numpy()
            on_epoch_end(epoch, total / len(X), probs)
    return model


# ----------------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------------
def run_cv(args, df, U, V, device):
    domain, labels = df["domain"].values, df["label"].values
    epoch_rows, pred_rows = [], []
    for seed in args.seeds:
        for fold, tr_idx, va_idx in make_folds(df, args.folds, seed):
            X_tr, X_va, _ = fold_features(U, V, domain, tr_idx, va_idx, args, seed)
            y_tr = train_labels(labels, tr_idx, args)
            y_va = labels[va_idx]

            def on_epoch_end(epoch, loss, probs):
                preds = probs.argmax(-1)
                epoch_rows.append({"seed": seed, "fold": fold, "epoch": epoch, "train_loss": loss,
                                   **compute_metrics(y_va, preds)})
                for i, idx in enumerate(va_idx):
                    pred_rows.append({"seed": seed, "fold": fold, "epoch": epoch,
                                      "row_id": int(df.index[idx]), "domain": int(domain[idx]),
                                      "label": int(y_va[i]), "pred": int(preds[i]),
                                      **{f"p{c}": float(probs[i, c]) for c in range(NUM_LABELS)}})

            train_mlp(args, X_tr, y_tr, device, seed, X_ev=X_va, on_epoch_end=on_epoch_end)
            last = epoch_rows[-1]
            print(f"seed {seed} fold {fold}: train {len(tr_idx)} / val {len(va_idx)} | "
                  f"input dim {X_tr.shape[1]} | final epoch: "
                  + " ".join(f"{k} {last[k]:.3f}" for k in METRICS))
    return pd.DataFrame(epoch_rows), pd.DataFrame(pred_rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=[str(BASE_DIR / "classifier_train_data/train.csv")],
                   help="CSV file(s) with all labelled pairs (several are pooled) for CV")
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                   help="sentence-transformers model name or local path")
    p.add_argument("--max-seq-length", type=int, default=None,
                   help="override the embedding model's max tokens (longer texts are truncated)")
    p.add_argument("--normalize", action="store_true", help="L2-normalise embeddings")
    p.add_argument("--keep-options", action="store_true",
                   help="embed full Question texts, including the answer options")
    p.add_argument("--embed-batch-size", type=int, default=32)
    p.add_argument("--cache-dir", default=str(BASE_DIR / "results" / "emb_cache"))
    p.add_argument("--no-domain", action="store_true", help="do not append the domain one-hot")
    p.add_argument("--pca", type=int, default=0,
                   help="reduce [u, v] to N components (0 = off)")
    p.add_argument("--hidden", type=int, nargs="+", default=[64],
                   help="hidden layer sizes, e.g. 32 or 64 32; 0 = no hidden layer (linear)")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--input-dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16, help="0 = full batch")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--class-weights", action="store_true", help="balanced class weights in the loss")
    p.add_argument("--swap-aug", action="store_true",
                   help="also train on (text_b, text_a) with the same label (training folds only)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--select-metric", choices=METRICS, default="linear_kappa",
                   help="metric used to pick the number of epochs")
    p.add_argument("--print-every", type=int, default=10, help="epochs between printed curve rows")
    p.add_argument("--final-fit", action="store_true",
                   help="retrain on all data for the selected epochs and save the model")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    args.hidden = [h for h in args.hidden if h > 0]
    return args


def main():
    args = parse_args()
    if args.output_dir is None:
        hid = "x".join(map(str, args.hidden)) or "linear"
        args.output_dir = str(BASE_DIR / "results" / (
            f"mlp_h{hid}_{Path(args.embedding_model).name}"
            + ("_nodomain" if args.no_domain else "") + (f"_pca{args.pca}" if args.pca else "")
            + ("_swap" if args.swap_aug else "")))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()

    df = prepare_texts(load_data(args.data), args.keep_options)
    print(f"Device: {device} | {len(df)} rows, {df['group'].nunique()} groups")
    U, V = embed_pairs(df, args, device)
    print(f"Embedding dim {U.shape[1]} | input [u, v] + domain | "
          f"MLP hidden {args.hidden or 'none (linear)'}\n")

    epochs_df, preds_df = run_cv(args, df, U, V, device)
    epochs_df.to_csv(out / "epoch_metrics.csv", index=False)
    preds_df.to_csv(out / "oof_predictions.csv", index=False)
    summary = summarise(args, epochs_df, preds_df, print_every=args.print_every)

    if args.final_fit:
        final_args = argparse.Namespace(**{**vars(args), "epochs": summary["selected_epoch"]})
        print(f"\nFinal fit on all {len(df)} rows for {final_args.epochs} epochs ...")
        all_idx = np.arange(len(df))
        X, _, pre = fold_features(U, V, df["domain"].values, all_idx, None, final_args,
                                  args.seeds[0])
        y = train_labels(df["label"].values, all_idx, args)
        model = train_mlp(final_args, X, y, device, args.seeds[0])
        joblib.dump({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                     "in_dim": X.shape[1], "preprocessor": pre, "args": vars(final_args)},
                    out / "final_model.joblib")
        summary["final_model"] = str(out / "final_model.joblib")
        print(f"Saved final model to {out / 'final_model.joblib'}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote results to {out}")


if __name__ == "__main__":
    # Run through the importable module so saved models reference train_cv_mlp.Preprocessor
    # (not __main__.Preprocessor) and can be loaded with joblib from other scripts/notebooks.
    import train_cv_mlp
    train_cv_mlp.main()
