"""
Cross-validated ordinal logistic regression on frozen sentence embeddings for
perturbation-strength prediction (the linear, ordinal counterpart of train_cv_mlp.py).

Model: proportional-odds (cumulative-logit) logistic regression with an L2 penalty,

    P(y <= k | x) = sigmoid(theta_k - w . x),   k = 0, 1, 2,   theta_0 < theta_1 < theta_2,

fitted by L-BFGS on  sum_i -log P(y_i | x_i) + alpha/2 * ||w||^2  (thresholds are not
penalised). It uses one weight vector for all classes and respects the order 0 < 1 < 2 < 3,
so it has far fewer parameters than a softmax/MLP classifier.

Features for a pair (u = emb(text_a), the original; v = emb(text_b), the perturbed):
    --pair-mode diffprod  [|u - v|, u * v]          (default; symmetric in u and v)
                diff      |u - v|
                prod      u * v
                concat    [u, v]                    (as in train_cv_mlp.py; order-dependent)
                full      [u, v, |u - v|, u * v]
                none      no embedding features (with the cosine: a cosine-only baseline)
  plus cos(u, v) (drop with --no-cosine) and a 0/1 domain indicator (drop with --no-domain).
The embedding features are standardised and, with --pca N, reduced to N components; all
preprocessing is fitted on the training rows only. --swap-aug (also train on (v, u)) only
changes anything for the order-dependent modes (concat, full).

Embeddings, Question-text cutting before "Options" (--keep-options to embed the full text),
the embedding cache, the grouped and stratified folds and the metrics are the same as in
train_cv_mlp.py, so with the same --seeds both scripts use identical folds.

Choosing alpha (nested CV, so the reported scores are not optimistic)
    * For each outer fold, every alpha in --alphas is scored by an inner grouped, stratified
      CV on the outer training rows (--inner-folds), and the best one (--select-metric; ties
      go to the larger alpha) is refitted on the whole outer training fold and used to
      predict the outer validation fold. These out-of-fold predictions give the main result.
    * alpha_curve.csv also lists the outer-CV scores of every alpha. Picking the best row of
      that table is selection on the evaluation folds (optimistic): use it to inspect the
      regularisation path, not to report results.
    * --final-fit picks alpha by CV on all rows (pooled over --seeds), refits on all rows and
      saves final_model.joblib, which evaluate_mlp.py evaluates on the annotation files.

Predicted class: --decision argmax (most probable class, default), median (smallest k with
P(y <= k) >= 0.5) or expected (expected strength rounded to the nearest class).

Example
    python train_cv_ordinal.py --seeds 42 43 44 45 46 --final-fit
    python train_cv_ordinal.py --pair-mode diffprod --pca 16 --seeds 42 43 44 45 46
    python train_cv_ordinal.py --pair-mode none --seeds 42 43 44 45 46     # cosine only
    python evaluate_mlp.py --model-path results/ordinal_diffprod_all-MiniLM-L6-v2/final_model.joblib

Outputs (--output-dir): oof_predictions.csv (nested CV), alpha_curve.csv, selected_alphas.csv,
summary.json, and with --final-fit final_model.joblib.

Requirements: numpy, scipy, scikit-learn, pandas, sentence-transformers (for the embeddings).
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from train_cv_mlp import (DOMAIN_NAMES, HIGHER_IS_BETTER, METRICS, NUM_DOMAINS, NUM_LABELS,
                          compute_metrics, embed_pairs, get_device, load_data, make_folds,
                          prepare_texts)

BASE_DIR = Path(__file__).parent
PAIR_MODES = ["diffprod", "diff", "prod", "concat", "full", "none"]
DECISIONS = ["argmax", "median", "expected"]
SELECT_METRICS = METRICS + ["log_loss"]
SELECT_HIGHER_IS_BETTER = {**HIGHER_IS_BETTER, "log_loss": False}


# ----------------------------------------------------------------------------
# Ordinal logistic regression (proportional odds, L2 penalty)
# ----------------------------------------------------------------------------
class OrdinalLogit:
    """Cumulative-logit model P(y <= k) = sigmoid(theta_k - w.x) with an L2 penalty on w.

    The thresholds are parametrised as theta_0 = b, theta_k = theta_{k-1} + exp(delta_k), so
    they stay ordered during unconstrained optimisation.
    """

    def __init__(self, alpha=1.0, n_classes=NUM_LABELS, class_weight=None, max_iter=5000,
                 tol=1e-8):
        self.alpha, self.n_classes, self.class_weight = alpha, n_classes, class_weight
        self.max_iter, self.tol = max_iter, tol

    def _unpack(self, params, d):
        w, b, deltas = params[:d], params[d], params[d + 1:]
        theta = np.concatenate([[b], b + np.cumsum(np.exp(deltas))])
        return w, theta, deltas

    def _objective(self, params, X, y, sw):
        n, d = X.shape
        K = self.n_classes
        w, theta, deltas = self._unpack(params, d)
        s = X @ w
        th = np.concatenate([[-np.inf], theta, [np.inf]])  # class y lies in (th[y], th[y+1]]
        upper, lower = th[y + 1] - s, th[y] - s
        # P(y) = sigmoid(upper) - sigmoid(lower)
        #      = sigmoid(upper) * sigmoid(-lower) * (1 - exp(lower - upper)),
        # which gives log P(y) and its gradient without cancellation or underflow.
        gap = upper - lower  # > 0 (inf for the first and last class)
        with np.errstate(over="ignore"):
            inv = 1.0 / np.expm1(gap)  # 0 when gap = inf
        log_p = (-np.logaddexp(0, -upper) - np.logaddexp(0, lower)
                 + np.log(-np.expm1(-gap)))
        loss = -np.sum(sw * log_p) + 0.5 * self.alpha * w @ w
        g_upper = -sw * (expit(-upper) + inv)  # dLoss/d(upper)
        g_lower = sw * (expit(lower) + inv)    # dLoss/d(lower)
        grad_w = X.T @ (-(g_upper + g_lower)) + self.alpha * w
        g_th = np.zeros(K + 1)
        np.add.at(g_th, y + 1, g_upper)
        np.add.at(g_th, y, g_lower)
        g_theta = g_th[1:K]
        grad_b = g_theta.sum()
        grad_deltas = np.exp(deltas) * np.cumsum(g_theta[::-1])[::-1][1:]
        return loss, np.concatenate([grad_w, [grad_b], grad_deltas])

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        n, d = X.shape
        K = self.n_classes
        counts = np.bincount(y, minlength=K).astype(float)
        if self.class_weight == "balanced":
            sw = (n / (K * np.maximum(counts, 1)))[y]
        else:
            sw = np.ones(n)
        # start from the (smoothed) class frequencies with w = 0
        cum = np.cumsum(counts + 1)[:-1] / (n + K)
        theta0 = np.log(cum / (1 - cum))
        x0 = np.concatenate([np.zeros(d), [theta0[0]], np.log(np.diff(theta0))])
        res = minimize(self._objective, x0, args=(X, y, sw), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter, "gtol": self.tol, "ftol": 1e-12})
        self.converged_ = bool(res.success)
        self.coef_, self.thresholds_, _ = self._unpack(res.x, d)
        return self

    def predict_proba(self, X):
        s = np.asarray(X, dtype=float) @ self.coef_
        cdf = expit(self.thresholds_[None, :] - s[:, None])
        n = len(s)
        cdf = np.concatenate([np.zeros((n, 1)), cdf, np.ones((n, 1))], axis=1)
        return np.clip(np.diff(cdf, axis=1), 0, 1)

    @staticmethod
    def decide(probs, decision="argmax"):
        if decision == "argmax":
            return probs.argmax(-1)
        if decision == "median":
            return (np.cumsum(probs, axis=1) >= 0.5).argmax(-1)
        if decision == "expected":
            exp = probs @ np.arange(probs.shape[1])
            return np.clip(np.rint(exp), 0, probs.shape[1] - 1).astype(int)
        raise ValueError(f"unknown decision rule {decision!r}")

    def predict(self, X, decision="argmax"):
        return self.decide(self.predict_proba(X), decision)


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def pair_embedding(U, V, mode):
    """Embedding part of the pair features (n, ?) for --pair-mode, or None for 'none'."""
    parts = {"diffprod": [np.abs(U - V), U * V], "diff": [np.abs(U - V)], "prod": [U * V],
             "concat": [U, V], "full": [U, V, np.abs(U - V), U * V], "none": []}[mode]
    return np.concatenate(parts, axis=1) if parts else None


def cosine(U, V):
    num = np.sum(U * V, axis=1)
    den = np.linalg.norm(U, axis=1) * np.linalg.norm(V, axis=1)
    return (num / np.maximum(den, 1e-12))[:, None]


class PairPreprocessor:
    """Pair features -> standardise (+ optional PCA); append standardised cosine and domain.

    Fitted on the training rows only.
    """

    def __init__(self, pair_mode="diffprod", pca_components=0, use_cosine=True,
                 use_domain=True, seed=0):
        self.pair_mode, self.pca_components = pair_mode, pca_components
        self.use_cosine, self.use_domain, self.seed = use_cosine, use_domain, seed

    def fit(self, U, V, domain):
        E = pair_embedding(U, V, self.pair_mode)
        self.scaler, self.pca = None, None
        if E is not None:
            self.scaler = StandardScaler().fit(E)
            if self.pca_components:
                n = min(self.pca_components, E.shape[0] - 1, E.shape[1])
                self.pca = PCA(n_components=n, random_state=self.seed).fit(self.scaler.transform(E))
        if self.use_cosine:
            self.cos_scaler = StandardScaler().fit(cosine(U, V))
        return self

    def transform(self, U, V, domain):
        parts = []
        E = pair_embedding(U, V, self.pair_mode)
        if E is not None:
            e = self.scaler.transform(E)
            parts.append(self.pca.transform(e) if self.pca is not None else e)
        if self.use_cosine:
            parts.append(self.cos_scaler.transform(cosine(U, V)))
        if self.use_domain:
            parts.append((np.asarray(domain) == 1).astype(float)[:, None])
        if not parts:
            raise ValueError("no features: --pair-mode none needs the cosine or the domain")
        return np.concatenate(parts, axis=1)


def fit_preprocessor(U, V, domain, idx, args, seed):
    """Fit preprocessing on rows idx (swap-augmented with --swap-aug); return (pre, X, rows)."""
    U_tr, V_tr, d_tr = U[idx], V[idx], domain[idx]
    rows = np.asarray(idx)
    if args.swap_aug:
        U_tr, V_tr = np.concatenate([U_tr, V_tr]), np.concatenate([V_tr, U_tr])
        d_tr, rows = np.concatenate([d_tr, d_tr]), np.concatenate([rows, rows])
    pre = PairPreprocessor(args.pair_mode, args.pca, not args.no_cosine, not args.no_domain,
                           seed).fit(U_tr, V_tr, d_tr)
    return pre, pre.transform(U_tr, V_tr, d_tr), rows


def fit_predict_alphas(U, V, domain, labels, tr_idx, ev_idx, args, seed):
    """Fit one model per alpha on tr_idx; return {alpha: probs on ev_idx}."""
    pre, X_tr, rows = fit_preprocessor(U, V, domain, tr_idx, args, seed)
    X_ev = pre.transform(U[ev_idx], V[ev_idx], domain[ev_idx])
    out = {}
    for a in args.alphas:
        model = OrdinalLogit(a, class_weight=args.class_weight).fit(X_tr, labels[rows])
        out[a] = model.predict_proba(X_ev)
    return out


# ----------------------------------------------------------------------------
# Metrics and alpha selection
# ----------------------------------------------------------------------------
def log_loss(y, probs):
    return float(-np.mean(np.log(np.maximum(probs[np.arange(len(y)), y], 1e-15))))


def select_score(y, probs, args):
    if args.select_metric == "log_loss":
        return log_loss(y, probs)
    return compute_metrics(y, OrdinalLogit.decide(probs, args.decision))[args.select_metric]


def best_alpha(scores, metric):
    """Best alpha by score (NaN counts as worst); ties go to the larger alpha."""
    sign = 1 if SELECT_HIGHER_IS_BETTER[metric] else -1
    vals = {a: sign * (s if np.isfinite(s) else -np.inf * sign) for a, s in scores.items()}
    top = max(vals.values())
    return max(a for a, v in vals.items() if v >= top - 1e-9)


def inner_cv_scores(df, U, V, labels, idx, args, seed):
    """Score every alpha by grouped, stratified CV on the rows idx (pooled predictions)."""
    sub = df.iloc[idx].reset_index(drop=True)
    pooled = {a: np.zeros((len(idx), NUM_LABELS)) for a in args.alphas}
    for _, tr, va in make_folds(sub, args.inner_folds, seed):
        probs = fit_predict_alphas(U[idx], V[idx], sub["domain"].values, labels[idx], tr, va,
                                   args, seed)
        for a in args.alphas:
            pooled[a][va] = probs[a]
    y = labels[idx]
    return {a: select_score(y, pooled[a], args) for a in args.alphas}


def pooled_metrics(preds):
    """Out-of-fold metrics: pool all folds per seed, then mean/std over seeds."""
    per_seed = {"all": [], **{DOMAIN_NAMES[d]: [] for d in range(NUM_DOMAINS)}}
    for _, s in preds.groupby("seed"):
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


def fmt(block):
    return " | ".join(f"{k} {v['mean']:.3f}±{v['std']:.3f}" for k, v in block.items())


# ----------------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------------
def run_cv(args, df, U, V):
    domain, labels = df["domain"].values, df["label"].values
    pred_rows, curve_rows, sel_rows = [], [], []
    for seed in args.seeds:
        for fold, tr_idx, va_idx in make_folds(df, args.folds, seed):
            inner = inner_cv_scores(df, U, V, labels, tr_idx, args, seed * 1000 + fold)
            alpha = best_alpha(inner, args.select_metric)
            probs = fit_predict_alphas(U, V, domain, labels, tr_idx, va_idx, args, seed)
            for a, p in probs.items():  # outer scores of every alpha (for the curve only)
                curve_rows += [{"seed": seed, "fold": fold, "alpha": a, "row_id": int(df.index[i]),
                                "domain": int(domain[i]), "label": int(labels[i]),
                                "pred": int(OrdinalLogit.decide(p[j:j + 1], args.decision)[0]),
                                "p_true": float(p[j, labels[i]])}
                               for j, i in enumerate(va_idx)]
            p = probs[alpha]
            preds = OrdinalLogit.decide(p, args.decision)
            for j, i in enumerate(va_idx):
                pred_rows.append({"seed": seed, "fold": fold, "alpha": alpha,
                                  "row_id": int(df.index[i]), "domain": int(domain[i]),
                                  "label": int(labels[i]), "pred": int(preds[j]),
                                  **{f"p{c}": float(p[j, c]) for c in range(NUM_LABELS)}})
            sel_rows.append({"seed": seed, "fold": fold, "selected_alpha": alpha,
                             **{f"inner_{args.select_metric}@{a:g}": s for a, s in inner.items()}})
            m = compute_metrics(labels[va_idx], preds)
            print(f"seed {seed} fold {fold}: train {len(tr_idx)} / val {len(va_idx)} | "
                  f"alpha {alpha:g} | " + " ".join(f"{k} {m[k]:.3f}" for k in METRICS))
    return pd.DataFrame(pred_rows), pd.DataFrame(curve_rows), pd.DataFrame(sel_rows)


def alpha_curve(curve_df):
    """Outer-CV metrics of each alpha (pooled per seed, mean over seeds)."""
    rows = []
    for a, sub in curve_df.groupby("alpha"):
        per_seed = [{**compute_metrics(s["label"], s["pred"]),
                     "log_loss": float(-np.mean(np.log(np.maximum(s["p_true"], 1e-15))))}
                    for _, s in sub.groupby("seed")]
        r = pd.DataFrame(per_seed)
        rows.append({"alpha": a, **{f"{k}_mean": r[k].mean() for k in r.columns},
                     **{f"{k}_std": r[k].std(ddof=0) for k in r.columns}})
    return pd.DataFrame(rows).sort_values("alpha")


def final_alpha(args, df, U, V):
    """alpha for the final model: CV on all rows, scores averaged over --seeds."""
    labels = df["label"].values
    all_idx = np.arange(len(df))
    per_seed = [inner_cv_scores(df, U, V, labels, all_idx, args, seed) for seed in args.seeds]
    scores = {a: float(np.nanmean([s[a] for s in per_seed])) for a in args.alphas}
    return best_alpha(scores, args.select_metric), scores


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=[str(BASE_DIR / "classifier_train_data/train.csv")],
                   help="training CSV(s) from xlsx_to_csv.py")
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                   help="sentence-transformers model name or path")
    p.add_argument("--max-seq-length", type=int, default=None,
                   help="override the encoder's max sequence length")
    p.add_argument("--normalize", action="store_true", help="L2-normalise embeddings")
    p.add_argument("--keep-options", action="store_true",
                   help="embed full Question texts instead of cutting them before 'Options'")
    p.add_argument("--embed-batch-size", type=int, default=32)
    p.add_argument("--cache-dir", default=str(BASE_DIR / "results" / "emb_cache"))
    p.add_argument("--pair-mode", choices=PAIR_MODES, default="diffprod",
                   help="how u and v are combined (see the description above)")
    p.add_argument("--no-cosine", action="store_true", help="do not add cos(u, v)")
    p.add_argument("--no-domain", action="store_true", help="do not add the domain indicator")
    p.add_argument("--pca", type=int, default=0,
                   help="reduce the embedding features to N PCA components (0 = off)")
    p.add_argument("--swap-aug", action="store_true",
                   help="also train on (v, u); only matters for --pair-mode concat/full")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[float(a) for a in np.logspace(-2, 4, 13)],
                   help="L2 penalties to choose from (one value = no tuning)")
    p.add_argument("--class-weight", choices=["balanced"], default=None,
                   help="weight the loss by inverse class frequency")
    p.add_argument("--decision", choices=DECISIONS, default="argmax",
                   help="how the predicted class is read off the class probabilities")
    p.add_argument("--select-metric", choices=SELECT_METRICS, default="linear_kappa",
                   help="inner-CV metric used to choose alpha")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--final-fit", action="store_true",
                   help="choose alpha by CV on all rows, refit on all rows and save the model")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    args.alphas = sorted(set(args.alphas))
    if args.pair_mode == "none" and args.no_cosine and args.no_domain:
        p.error("--pair-mode none with --no-cosine and --no-domain leaves no features")
    return args


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = str(BASE_DIR / "results" / (
            f"ordinal_{args.pair_mode}_{Path(args.embedding_model).name}"
            + ("_nocos" if args.no_cosine else "") + ("_nodomain" if args.no_domain else "")
            + (f"_pca{args.pca}" if args.pca else "") + ("_swap" if args.swap_aug else "")
            + ("_balanced" if args.class_weight else "")
            + (f"_{args.decision}" if args.decision != "argmax" else "")))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()

    df = prepare_texts(load_data(args.data), args.keep_options)
    print(f"{len(df)} rows, {df['group'].nunique()} groups")
    U, V = embed_pairs(df, args, device)
    U, V = U.astype(np.float64), V.astype(np.float64)
    n_feat = PairPreprocessor(args.pair_mode, args.pca, not args.no_cosine,
                              not args.no_domain).fit(U, V, df["domain"].values) \
        .transform(U[:1], V[:1], df["domain"].values[:1]).shape[1]
    print(f"Embedding dim {U.shape[1]} | pair mode {args.pair_mode} | {n_feat} features"
          + (f" (PCA {args.pca})" if args.pca else "")
          + f" | alphas {', '.join(f'{a:g}' for a in args.alphas)} | "
          f"alpha chosen by inner {args.inner_folds}-fold CV on {args.select_metric}\n")

    preds_df, curve_df, sel_df = run_cv(args, df, U, V)
    preds_df.to_csv(out / "oof_predictions.csv", index=False)
    sel_df.to_csv(out / "selected_alphas.csv", index=False)
    curve = alpha_curve(curve_df)
    curve.to_csv(out / "alpha_curve.csv", index=False)

    summary = {"nested_cv_oof": pooled_metrics(preds_df),
               "selected_alphas": sel_df["selected_alpha"].tolist(),
               "args": vars(args)}
    cols = ["alpha"] + [f"{k}_mean" for k in METRICS + ["log_loss"]]
    print("\nOuter-CV score of each alpha (inspection only; choosing from this table is "
          "optimistic):")
    print(curve[cols].rename(columns=lambda c: c.replace("_mean", "")).round(4).to_string(index=False))
    counts = sel_df["selected_alpha"].value_counts().sort_index()
    print("\nalpha chosen by the inner CV: "
          + ", ".join(f"{a:g} ({n}x)" for a, n in counts.items()))
    print("Nested-CV out-of-fold results (mean±std over seeds):")
    for name, block in summary["nested_cv_oof"].items():
        print(f"  {name:9s} {fmt(block)}")

    if args.final_fit:
        alpha, scores = final_alpha(args, df, U, V)
        print(f"\nFinal fit on all {len(df)} rows with alpha {alpha:g} "
              f"(CV {args.select_metric} on all rows: {scores[alpha]:.3f})")
        all_idx = np.arange(len(df))
        pre, X, rows = fit_preprocessor(U, V, df["domain"].values, all_idx, args, args.seeds[0])
        model = OrdinalLogit(alpha, class_weight=args.class_weight).fit(X, df["label"].values[rows])
        if not model.converged_:
            print("WARNING: the optimiser did not fully converge")
        joblib.dump({"model_type": "ordinal", "model": model, "preprocessor": pre,
                     "alpha": alpha, "args": vars(args)}, out / "final_model.joblib")
        summary.update(final_model=str(out / "final_model.joblib"), final_alpha=alpha,
                       final_alpha_cv_scores={f"{a:g}": s for a, s in scores.items()})
        print(f"Saved final model to {out / 'final_model.joblib'}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote results to {out}")


if __name__ == "__main__":
    # Run through the importable module so saved models reference train_cv_ordinal.OrdinalLogit
    # (not __main__.OrdinalLogit) and can be loaded with joblib from evaluate_mlp.py.
    import train_cv_ordinal
    train_cv_ordinal.main()
