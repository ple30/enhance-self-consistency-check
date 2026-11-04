"""
Cross-validated MLP on frozen sentence embeddings for perturbation-strength prediction.

Each text is embedded once with a frozen sentence-transformer (cached to disk). A pair
(u = emb(text_a), v = emb(text_b)) is turned into pair features, the domain is appended as
a one-hot vector, and a small MLP predicts the strength 0-3:

    0: no/minimal change   1: small   2: moderately strong   3: strong

Pair features (--features, comma-separated):
    diff : |u - v|          prod : u * v          cos : cosine similarity (1 number)
    u    : u                v    : v
The default "diff,prod,cos" is symmetric in (text_a, text_b) and keeps the input small,
which matters with ~60 labelled pairs. Adding "u,v" gives the classic [u, v, |u-v|, u*v]
input; only then does --swap-aug (also train on (v, u)) change anything.

--hidden sets the MLP layer sizes; --hidden 0 gives a linear softmax layer trained by
gradient descent with early stopping (close to, but not exactly, logistic regression).

--classifier logreg instead fits scikit-learn's multinomial LogisticRegression (L2 penalty,
solved to convergence) on the same features. Its inverse regularisation strength C is chosen
separately in every outer training fold by an inner grouped CV (--inner-folds, --C-grid),
with preprocessing refitted inside each inner fold, so the outer validation fold is never
used for any choice. There are no epochs: the reported "epoch" is always 1.

Preprocessing is fitted on the training fold only: features are standardised and, with
--pca N, the embedding block is reduced to N components.

Cross-validation is identical to train_cv.py: same data loading, same folds for a given
seed (grouped, stratified on label x domain), the number of epochs chosen from the
fold-averaged curve, and the same metrics (linear kappa, Pearson, accuracy, macro-F1, MAE)
reported per fold, out-of-fold, and per domain. Results can be compared directly.

Example
    python train_cv_mlp.py --embedding-model all-mpnet-base-v2 --seeds 42 43 44 45 46 --final-fit
    python train_cv_mlp.py --classifier logreg --seeds 42 43 44 45 46   # logistic regression
    python train_cv_mlp.py --features u,v,diff,prod,cos --pca 32 --swap-aug

Requirements: torch, sentence-transformers, scikit-learn, pandas.
"""
import argparse
import hashlib
import json
from pathlib import Path

import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from train_cv import (HIGHER_IS_BETTER, METRICS, NUM_DOMAINS, NUM_LABELS, class_weights,
                      compute_metrics, get_device, load_data, make_folds, set_seed, summarise)

BASE_DIR = Path(__file__).parent
FEATURE_KINDS = ["u", "v", "diff", "prod", "cos"]
EMB_KINDS = ["u", "v", "diff", "prod"]  # vector-valued features (PCA applies to these)


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
def pair_features(U, V, kinds):
    """Vector-valued block (for scaling/PCA) and scalar block, for the requested kinds."""
    vec = {"u": U, "v": V, "diff": np.abs(U - V), "prod": U * V}
    emb_block = [vec[k] for k in kinds if k in EMB_KINDS]
    emb_block = np.concatenate(emb_block, axis=1) if emb_block else np.zeros((len(U), 0))
    scalars = []
    if "cos" in kinds:
        nu, nv = np.linalg.norm(U, axis=1), np.linalg.norm(V, axis=1)
        scalars.append((U * V).sum(1) / np.maximum(nu * nv, 1e-12))
    scalar_block = np.stack(scalars, axis=1) if scalars else np.zeros((len(U), 0))
    return emb_block.astype(np.float32), scalar_block.astype(np.float32)


class Preprocessor:
    """Standardise (+ optional PCA) the embedding block, standardise scalars, add domain one-hot.

    Fitted on the training fold only.
    """

    def __init__(self, pca_components=0, use_domain=True, seed=0):
        self.pca_components, self.use_domain, self.seed = pca_components, use_domain, seed

    def fit(self, emb, scal):
        self.emb_scaler = StandardScaler().fit(emb) if emb.shape[1] else None
        self.pca = None
        if self.pca_components and emb.shape[1]:
            n = min(self.pca_components, emb.shape[0] - 1, emb.shape[1])
            self.pca = PCA(n_components=n, random_state=self.seed).fit(self.emb_scaler.transform(emb))
        self.scal_scaler = StandardScaler().fit(scal) if scal.shape[1] else None
        return self

    def transform(self, emb, scal, domain):
        parts = []
        if self.emb_scaler is not None:
            e = self.emb_scaler.transform(emb)
            parts.append(self.pca.transform(e) if self.pca is not None else e)
        if self.scal_scaler is not None:
            parts.append(self.scal_scaler.transform(scal))
        if self.use_domain:
            parts.append(np.eye(NUM_DOMAINS)[domain])
        return np.concatenate(parts, axis=1).astype(np.float32)


def fold_features(U, V, domain, tr_idx, ev_idx, args, seed):
    """Fit preprocessing on the training rows (optionally swap-augmented); transform both."""
    U_tr, V_tr, d_tr = U[tr_idx], V[tr_idx], domain[tr_idx]
    if args.swap_aug:
        U_tr, V_tr = np.concatenate([U_tr, V_tr]), np.concatenate([V_tr, U_tr])
        d_tr = np.concatenate([d_tr, d_tr])
    emb_tr, scal_tr = pair_features(U_tr, V_tr, args.features)
    pre = Preprocessor(args.pca, use_domain=not args.no_domain, seed=seed).fit(emb_tr, scal_tr)
    X_tr = pre.transform(emb_tr, scal_tr, d_tr)
    X_ev = None
    if ev_idx is not None:
        emb_ev, scal_ev = pair_features(U[ev_idx], V[ev_idx], args.features)
        X_ev = pre.transform(emb_ev, scal_ev, domain[ev_idx])
    return X_tr, X_ev, pre


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
# Logistic regression (scikit-learn), C chosen by inner grouped CV
# ----------------------------------------------------------------------------
def train_labels(labels, idx, args):
    """Labels for training rows, repeated to match swap augmentation."""
    return np.concatenate([labels[idx]] * (2 if args.swap_aug else 1))


def fit_logreg(X, y, C, args, seed):
    clf = LogisticRegression(C=C, max_iter=args.max_iter, random_state=seed,
                             class_weight="balanced" if args.class_weights else None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return clf.fit(X, y)


def logreg_proba(clf, X):
    """Class probabilities as an (n, NUM_LABELS) array, even if a class was absent in training."""
    probs = np.zeros((len(X), NUM_LABELS))
    probs[:, clf.classes_] = clf.predict_proba(X)
    return probs


def choose_C(df, U, V, idx, args, seed):
    """Pick C by inner grouped CV on rows `idx` (pooled inner out-of-fold predictions).

    Preprocessing is refitted inside each inner fold. Ties go to the smaller C
    (stronger regularisation). Returns (best C, {C: score}).
    """
    domain, labels = df["domain"].values, df["label"].values
    sub = df.iloc[idx]
    preds = {C: np.empty(len(idx), dtype=int) for C in args.C_grid}
    for _, in_tr, in_va in make_folds(sub, args.inner_folds, seed):
        tr, va = idx[in_tr], idx[in_va]
        X_tr, X_va, _ = fold_features(U, V, domain, tr, va, args, seed)
        y_tr = train_labels(labels, tr, args)
        for C in args.C_grid:
            preds[C][in_va] = logreg_proba(fit_logreg(X_tr, y_tr, C, args, seed), X_va).argmax(-1)
    worst = -np.inf if HIGHER_IS_BETTER[args.select_metric] else np.inf
    scores = {}
    for C in args.C_grid:
        s = compute_metrics(labels[idx], preds[C])[args.select_metric]
        scores[C] = worst if np.isnan(s) else s
    sign = 1 if HIGHER_IS_BETTER[args.select_metric] else -1
    best = max(sorted(args.C_grid), key=lambda C: sign * scores[C])  # first max = smallest C
    return best, scores


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
            extra = {}

            def on_epoch_end(epoch, loss, probs):
                preds = probs.argmax(-1)
                epoch_rows.append({"seed": seed, "fold": fold, "epoch": epoch, "train_loss": loss,
                                   **extra, **compute_metrics(y_va, preds)})
                for i, idx in enumerate(va_idx):
                    pred_rows.append({"seed": seed, "fold": fold, "epoch": epoch,
                                      "row_id": int(df.index[idx]), "domain": int(domain[idx]),
                                      "label": int(y_va[i]), "pred": int(preds[i]),
                                      **{f"p{c}": float(probs[i, c]) for c in range(NUM_LABELS)}})

            if args.classifier == "logreg":
                C, _ = choose_C(df, U, V, tr_idx, args, seed)
                clf = fit_logreg(X_tr, y_tr, C, args, seed)
                p_tr = logreg_proba(clf, X_tr)
                train_loss = float(-np.log(np.clip(p_tr[np.arange(len(y_tr)), y_tr], 1e-12, 1)).mean())
                extra["C"] = C
                on_epoch_end(1, train_loss, logreg_proba(clf, X_va))
            else:
                train_mlp(args, X_tr, y_tr, device, seed, X_ev=X_va, on_epoch_end=on_epoch_end)
            last = epoch_rows[-1]
            print(f"seed {seed} fold {fold}: train {len(tr_idx)} / val {len(va_idx)} | "
                  f"input dim {X_tr.shape[1]}"
                  + (f" | C {extra['C']:g}" if "C" in extra else " | final epoch") + ": "
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
    p.add_argument("--embed-batch-size", type=int, default=32)
    p.add_argument("--cache-dir", default=str(BASE_DIR / "results" / "emb_cache"))
    p.add_argument("--features", default="diff,prod,cos",
                   help=f"comma-separated subset of {FEATURE_KINDS}")
    p.add_argument("--no-domain", action="store_true", help="do not append the domain one-hot")
    p.add_argument("--pca", type=int, default=0,
                   help="reduce the embedding block to N components (0 = off)")
    p.add_argument("--classifier", choices=["mlp", "logreg"], default="mlp",
                   help="mlp: PyTorch MLP; logreg: scikit-learn LogisticRegression with C tuned "
                        "by inner grouped CV")
    p.add_argument("--C-grid", type=float, nargs="+", default=list(np.logspace(-4, 2, 13)),
                   help="logreg: candidate inverse regularisation strengths")
    p.add_argument("--inner-folds", type=int, default=4, help="logreg: inner CV folds for C")
    p.add_argument("--max-iter", type=int, default=5000, help="logreg: solver iterations")
    p.add_argument("--hidden", type=int, nargs="+", default=[64],
                   help="hidden layer sizes; 0 = linear classifier (logistic regression)")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--input-dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16, help="0 = full batch")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--class-weights", action="store_true", help="balanced class weights in the loss")
    p.add_argument("--swap-aug", action="store_true",
                   help="also train on (text_b, text_a); only matters if features include u or v")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--select-metric", choices=METRICS, default="linear_kappa",
                   help="metric used to pick the number of epochs")
    p.add_argument("--print-every", type=int, default=10, help="epochs between printed curve rows")
    p.add_argument("--final-fit", action="store_true",
                   help="retrain on all data for the selected epochs and save the model")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    args.features = [f.strip() for f in args.features.split(",") if f.strip()]
    unknown = set(args.features) - set(FEATURE_KINDS)
    if unknown or not args.features:
        p.error(f"--features must be a non-empty subset of {FEATURE_KINDS}; got {args.features}")
    args.hidden = [h for h in args.hidden if h > 0]
    if args.classifier == "logreg":
        args.epochs = 1  # no epochs; summarise() treats the single fit as epoch 1
        args.C_grid = sorted(float(c) for c in args.C_grid)
    if args.swap_aug and not {"u", "v"} & set(args.features):
        print("NOTE: --swap-aug has no effect with symmetric features only (diff, prod, cos)")
    return args


def main():
    args = parse_args()
    if args.output_dir is None:
        hid = "x".join(map(str, args.hidden)) or "linear"
        clf_name = "logreg" if args.classifier == "logreg" else f"mlp_h{hid}"
        args.output_dir = str(BASE_DIR / "results" / (
            f"{clf_name}_{Path(args.embedding_model).name}_{'-'.join(args.features)}"
            + ("_nodomain" if args.no_domain else "") + (f"_pca{args.pca}" if args.pca else "")))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()

    df = load_data(args.data)
    print(f"Device: {device} | {len(df)} rows, {df['group'].nunique()} groups")
    U, V = embed_pairs(df, args, device)
    model_desc = ("logreg (C by inner CV)" if args.classifier == "logreg"
                  else f"MLP hidden {args.hidden or 'linear'}")
    print(f"Embedding dim {U.shape[1]} | features {args.features} | {model_desc}\n")

    epochs_df, preds_df = run_cv(args, df, U, V, device)
    epochs_df.to_csv(out / "epoch_metrics.csv", index=False)
    preds_df.to_csv(out / "oof_predictions.csv", index=False)
    summary = summarise(args, epochs_df, preds_df, print_every=args.print_every)
    if args.classifier == "logreg":
        counts = epochs_df["C"].value_counts().sort_index()
        summary["chosen_C_counts"] = {f"{c:g}": int(n) for c, n in counts.items()}
        print("Chosen C per outer fold: "
              + ", ".join(f"{c:g} x{n}" for c, n in counts.items()))

    if args.final_fit and args.classifier == "logreg":
        all_idx = np.arange(len(df))
        C, scores = choose_C(df, U, V, all_idx, args, args.seeds[0])
        print(f"\nFinal fit on all {len(df)} rows with C={C:g} (chosen by inner CV on all data) ...")
        X, _, pre = fold_features(U, V, df["domain"].values, all_idx, None, args, args.seeds[0])
        clf = fit_logreg(X, train_labels(df["label"].values, all_idx, args), C, args, args.seeds[0])
        joblib.dump({"classifier": clf, "preprocessor": pre, "C": C, "args": vars(args)},
                    out / "final_model.joblib")
        summary.update(final_model=str(out / "final_model.joblib"), final_C=C)
        print(f"Saved final model to {out / 'final_model.joblib'}")
    elif args.final_fit:
        final_args = argparse.Namespace(**{**vars(args), "epochs": summary["selected_epoch"]})
        print(f"\nFinal fit on all {len(df)} rows for {final_args.epochs} epochs ...")
        all_idx = np.arange(len(df))
        X, _, pre = fold_features(U, V, df["domain"].values, all_idx, None, final_args,
                                  args.seeds[0])
        y = np.concatenate([df["label"].values] * (2 if args.swap_aug else 1))
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
