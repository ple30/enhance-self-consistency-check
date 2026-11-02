"""
Cross-validated fine-tuning of a cross-encoder that predicts perturbation strength.

Given an (original, perturbed) text pair, the model predicts how strongly the
perturbation changes the meaning / reasoning:

    0: no/minimal change   1: small   2: moderately strong   3: strong

Domain information (0 = CoT, 1 = Question; see xlsx_to_csv.py) is used in two ways,
selectable with --domain-mode:
    prefix : a short domain tag is prepended to text_a, so the encoder sees it
    embed  : a learned domain embedding is concatenated to the pooled [CLS] vector
    both   : prefix + embed (default)
    none   : ignore domain (ablation)

Cross-validation
    * train.csv and val.csv are pooled (the fixed val split is too small to be useful).
    * Folds come from StratifiedGroupKFold: rows sharing the same original text
      (`group`) never straddle train/validation, and folds are stratified on
      label x domain.
    * Every fold is trained for a fixed number of epochs and evaluated after each one.
      The number of epochs is chosen afterwards from the fold-averaged curve
      (no per-fold early stopping). Metrics at the selected epoch are therefore slightly
      optimistic; the metrics at the final epoch (fixed recipe) are reported too.
    * --seeds repeats the whole CV with different splits and initialisations.
    * --final-fit retrains on all data for the selected number of epochs and saves it.

Example
    python train_cv.py --model microsoft/deberta-v3-small --epochs 10 --seeds 42 43 44 \
        --swap-aug --final-fit

Requirements: torch, transformers, sentencepiece (for DeBERTa), scikit-learn, pandas.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from xlsx_to_csv import add_swapped

BASE_DIR = Path(__file__).parent
NUM_LABELS = 4
NUM_DOMAINS = 2
DOMAIN_NAMES = {0: "CoT", 1: "Question"}
DOMAIN_TAGS = {0: "Domain: chain-of-thought.", 1: "Domain: question."}
METRICS = ["linear_kappa", "pearson", "accuracy", "macro_f1", "mae"]
HIGHER_IS_BETTER = {"linear_kappa": True, "pearson": True, "accuracy": True, "macro_f1": True,
                    "mae": False}


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
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
    return df.reset_index(drop=True)


class Collator:
    def __init__(self, tokenizer, max_length, use_prefix):
        self.tok, self.max_length, self.use_prefix = tokenizer, max_length, use_prefix

    def __call__(self, rows):
        a = [r["text_a"] for r in rows]
        if self.use_prefix:
            a = [f"{DOMAIN_TAGS[r['domain']]} {t}" for r, t in zip(rows, a)]
        b = [r["text_b"] for r in rows]
        enc = self.tok(a, b, truncation=True, max_length=self.max_length,
                       padding=True, return_tensors="pt")
        enc["domain"] = torch.tensor([r["domain"] for r in rows], dtype=torch.long)
        enc["labels"] = torch.tensor([r["label"] for r in rows], dtype=torch.long)
        return enc


def make_loader(df, collator, batch_size, shuffle, seed=0):
    records = df[["text_a", "text_b", "domain", "label"]].to_dict("records")
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(records, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collator, generator=g)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class PairClassifier(nn.Module):
    """Cross-encoder: [CLS] text_a [SEP] text_b -> pooled vector (+ domain emb) -> 4 classes."""

    def __init__(self, model_name, use_domain_emb=True, domain_dim=16, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.domain_emb = nn.Embedding(NUM_DOMAINS, domain_dim) if use_domain_emb else None
        in_dim = hidden + (domain_dim if use_domain_emb else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, NUM_LABELS))

    def forward(self, domain, **enc):
        pooled = self.encoder(**enc).last_hidden_state[:, 0]
        if self.domain_emb is not None:
            pooled = torch.cat([pooled, self.domain_emb(domain)], dim=-1)
        return self.head(pooled)

    def head_parameters(self):
        params = list(self.head.parameters())
        if self.domain_emb is not None:
            params += list(self.domain_emb.parameters())
        return params


def build_model(args, device):
    use_emb = args.domain_mode in ("embed", "both")
    model = PairClassifier(args.model, use_domain_emb=use_emb, domain_dim=args.domain_dim,
                           dropout=args.dropout).to(device)
    head_ids = {id(p) for p in model.head_parameters()}
    enc_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr},
        {"params": model.head_parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    return model, optimizer


def class_weights(labels, device):
    counts = np.bincount(labels, minlength=NUM_LABELS).astype(float)
    w = counts.sum() / (NUM_LABELS * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float, device=device)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, max_grad_norm):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        labels = batch.pop("labels")
        logits = model(**batch)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(labels)
        n += len(labels)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch = to_device(batch, device)
        batch.pop("labels")
        probs.append(torch.softmax(model(**batch).float(), dim=-1).cpu().numpy())
    return np.concatenate(probs)


def fit(args, train_df, device, seed, eval_df=None, on_epoch_end=None):
    """Train for args.epochs epochs; call on_epoch_end(epoch, train_loss, probs) after each."""
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collator = Collator(tokenizer, args.max_length, args.domain_mode in ("prefix", "both"))
    if args.swap_aug:
        train_df = add_swapped(train_df)
    train_loader = make_loader(train_df, collator, args.batch_size, shuffle=True, seed=seed)
    eval_loader = (make_loader(eval_df, collator, args.eval_batch_size, shuffle=False)
                   if eval_df is not None else None)

    model, optimizer = build_model(args, device)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.ceil(args.warmup_ratio * total_steps), total_steps)
    weight = class_weights(train_df["label"].values, device) if args.class_weights else None
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device,
                               args.max_grad_norm)
        if on_epoch_end is not None:
            probs = predict(model, eval_loader, device) if eval_loader is not None else None
            on_epoch_end(epoch, loss, probs)
    return model, tokenizer


# ----------------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------------
def run_cv(args, df, device):
    epoch_rows, pred_rows = [], []
    strata = df["label"].astype(str) + "_" + df["domain"].astype(str)
    for seed in args.seeds:
        skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(df, strata, groups=df["group"])):
            tr, va = df.iloc[tr_idx], df.iloc[va_idx]
            assert not set(tr["group"]) & set(va["group"]), "group leakage between folds"
            t0 = time.time()

            def on_epoch_end(epoch, loss, probs):
                preds = probs.argmax(-1)
                m = compute_metrics(va["label"], preds)
                epoch_rows.append({"seed": seed, "fold": fold, "epoch": epoch,
                                   "train_loss": loss, **m})
                for i, (idx, p) in enumerate(zip(va.index, preds)):
                    pred_rows.append({"seed": seed, "fold": fold, "epoch": epoch, "row_id": idx,
                                      "domain": va.at[idx, "domain"], "label": va.at[idx, "label"],
                                      "pred": int(p),
                                      **{f"p{c}": float(probs[i, c]) for c in range(NUM_LABELS)}})
                print(f"  seed {seed} fold {fold} epoch {epoch:2d} | loss {loss:.4f} | "
                      + " ".join(f"{k} {m[k]:.3f}" for k in METRICS))

            print(f"seed {seed} fold {fold}: train {len(tr)} / val {len(va)} rows "
                  f"(val labels {np.bincount(va['label'], minlength=NUM_LABELS).tolist()})")
            fit(args, tr, device, seed, eval_df=va, on_epoch_end=on_epoch_end)
            print(f"  fold done in {time.time() - t0:.0f}s")
    return pd.DataFrame(epoch_rows), pd.DataFrame(pred_rows)


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


def summarise(args, epochs_df, preds_df):
    curve = epochs_df.groupby("epoch")[METRICS + ["train_loss"]].agg(["mean", "std"])
    print("\nFold-averaged validation curve (mean over seeds x folds):")
    print(curve.xs("mean", axis=1, level=1).round(4).to_string())

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
# Main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=[str(BASE_DIR / "classifier_train_data/train.csv"),
                                                 str(BASE_DIR / "classifier_train_data/val.csv")],
                   help="CSV files to pool for CV (columns: text_a, text_b, label, domain, group)")
    p.add_argument("--model", default="microsoft/deberta-v3-small",
                   help="Hugging Face model name or local path")
    p.add_argument("--output-dir", default=None, help="default: results/cv_<model>_<domain-mode>")
    p.add_argument("--domain-mode", choices=["both", "embed", "prefix", "none"], default="both")
    p.add_argument("--domain-dim", type=int, default=16)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5, help="encoder learning rate")
    p.add_argument("--head-lr", type=float, default=1e-4, help="classifier head + domain emb LR")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--class-weights", action="store_true", help="balanced class weights in the loss")
    p.add_argument("--swap-aug", action="store_true",
                   help="add (text_b, text_a) copies of training rows (train folds only)")
    p.add_argument("--select-metric", choices=METRICS, default="linear_kappa",
                   help="metric used to pick the number of epochs")
    p.add_argument("--final-fit", action="store_true",
                   help="retrain on all data for the selected epochs and save the model")
    return p.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = str(BASE_DIR / "results" /
                              f"cv_{Path(args.model).name}_{args.domain_mode}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()

    df = load_data(args.data)
    print(f"Device: {device} | {len(df)} rows, {df['group'].nunique()} groups")
    print(pd.crosstab(df["domain"].map(DOMAIN_NAMES), df["label"]).to_string(), "\n")

    epochs_df, preds_df = run_cv(args, df, device)
    epochs_df.to_csv(out / "epoch_metrics.csv", index=False)
    preds_df.to_csv(out / "oof_predictions.csv", index=False)
    summary = summarise(args, epochs_df, preds_df)

    if args.final_fit:
        final_args = argparse.Namespace(**{**vars(args), "epochs": summary["selected_epoch"]})
        print(f"\nFinal fit on all {len(df)} rows for {final_args.epochs} epochs ...")
        model, tokenizer = fit(final_args, df, device, args.seeds[0])
        model_dir = out / "final_model"
        model_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_dir / "model.pt")
        tokenizer.save_pretrained(model_dir)
        (model_dir / "train_args.json").write_text(json.dumps(vars(final_args), indent=2))
        summary["final_model_dir"] = str(model_dir)
        print(f"Saved final model to {model_dir}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote results to {out}")


if __name__ == "__main__":
    main()
