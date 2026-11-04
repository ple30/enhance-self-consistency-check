"""
Evaluate a final model from train_cv_mlp.py (--final-fit) on held-out test data.

Loads final_model.joblib (the MLP with its fitted preprocessing),
embeds the test pairs with the same frozen sentence-transformer and settings used in
training, predicts perturbation strength 0-3, and reports linear-weighted kappa and Pearson
correlation (plus accuracy, macro-F1, MAE): overall, per domain and per source file.

Test pairs whose original text (group) also occurs in the training data are flagged, and
metrics are also reported without them, since they are not fully unseen.

Example
    python train_cv_mlp.py --embedding-model sentence-transformers/all-MiniLM-L6-v2 --hidden 32 \
        --swap-aug --seeds 42 43 44 45 46 --final-fit --output-dir results/mlp_minilm
    python evaluate_mlp.py --model-path results/mlp_minilm/final_model.joblib

Outputs (default: <model dir>/test/): test_predictions.csv, test_metrics.json.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

import train_cv_mlp
from train_cv_mlp import (DOMAIN_NAMES, METRICS, MLP, NUM_LABELS, compute_metrics, embed_pairs,
                          get_device, load_data, pair_features)

BASE_DIR = Path(__file__).parent


def load_bundle(path):
    # Models saved before train_cv_mlp.py ran through its module reference __main__.Preprocessor
    import __main__
    if not hasattr(__main__, "Preprocessor"):
        __main__.Preprocessor = train_cv_mlp.Preprocessor
    bundle = joblib.load(path)
    # Models from earlier versions (logistic regression, or features other than [u, v]) are
    # not compatible with the current preprocessing.
    features = bundle["args"].get("features", ["u", "v"])
    if "classifier" in bundle or list(features) != ["u", "v"]:
        raise SystemExit(f"{path} was trained by an earlier version of train_cv_mlp.py "
                         f"(features {features}); retrain it with the current version.")
    return bundle


def predict_proba(bundle, X, device):
    a = bundle["args"]
    model = MLP(bundle["in_dim"], a["hidden"], a["dropout"], a["input_dropout"]).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X, device=device)), dim=-1).cpu().numpy()


def fmt(m):
    return " | ".join(f"{k} {m[k]:.3f}" for k in METRICS)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, help="final_model.joblib from train_cv_mlp.py --final-fit")
    p.add_argument("--test-data", nargs="+", default=[str(BASE_DIR / "annotation_data/test.csv")])
    p.add_argument("--train-data", nargs="+", default=None,
                   help="training CSV(s), to flag test pairs sharing an original text "
                        "(default: the files the model was trained on)")
    p.add_argument("--cache-dir", default=None, help="embedding cache (default: as in training)")
    p.add_argument("--output-dir", default=None, help="default: <model dir>/test")
    args = p.parse_args()

    bundle = load_bundle(args.model_path)
    train_args = argparse.Namespace(**bundle["args"])
    if args.cache_dir:
        train_args.cache_dir = args.cache_dir
    out = Path(args.output_dir or Path(args.model_path).parent / "test")
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print(f"Model: MLP hidden {train_args.hidden or 'none (linear)'} | input [u, v] + domain | "
          f"embeddings {train_args.embedding_model}")

    df = load_data(args.test_data)
    train_files = args.train_data or [f for f in train_args.data if Path(f).exists()]
    if not train_files:  # trained elsewhere (e.g. another Colab path): fall back to the repo copy
        train_files = [str(BASE_DIR / "classifier_train_data/train.csv")]
    train_groups = set(load_data(train_files)["group"])
    df["seen_original"] = df["group"].isin(train_groups)
    print(f"Test: {len(df)} pairs | {int(df['seen_original'].sum())} share an original text "
          f"with the training data")

    U, V = embed_pairs(df, train_args, device)
    X = bundle["preprocessor"].transform(pair_features(U, V), df["domain"].values)
    probs = predict_proba(bundle, X, device)
    df["pred"] = probs.argmax(-1)
    for c in range(NUM_LABELS):
        df[f"p{c}"] = probs[:, c]

    unseen = df[~df["seen_original"]]
    results = {"overall": compute_metrics(df["label"], df["pred"]),
               "excluding_seen_originals": (compute_metrics(unseen["label"], unseen["pred"])
                                            if len(unseen) else None),
               "per_domain": {}, "per_source_file": {}}
    for d, name in DOMAIN_NAMES.items():
        sub = df[df["domain"] == d]
        if len(sub):
            results["per_domain"][name] = compute_metrics(sub["label"], sub["pred"])
    for f, sub in df.groupby("source_file"):
        results["per_source_file"][f] = {"n": len(sub), **compute_metrics(sub["label"], sub["pred"])}
    cm = confusion_matrix(df["label"], df["pred"], labels=list(range(NUM_LABELS)))
    results["confusion_matrix"] = cm.tolist()
    results.update(n=len(df), model_path=str(args.model_path), test_data=args.test_data)

    print(f"\nLinear kappa: {results['overall']['linear_kappa']:.3f}   "
          f"Pearson r: {results['overall']['pearson']:.3f}")
    print(f"\n{'overall':28s} {fmt(results['overall'])}")
    if results["excluding_seen_originals"] is not None:
        print(f"{'excluding seen originals':28s} {fmt(results['excluding_seen_originals'])}")
    else:
        print(f"{'excluding seen originals':28s} (none: every test pair shares an original text)")
    for name, m in results["per_domain"].items():
        print(f"{name:28s} {fmt(m)}")
    print("\nPer source file:")
    for f, m in results["per_source_file"].items():
        print(f"  {f:36s} n={m['n']:3d}  linear_kappa {m['linear_kappa']:.3f}  pearson {m['pearson']:.3f}")
    print("\nConfusion matrix (rows = gold 0-3, columns = predicted 0-3):")
    print(pd.DataFrame(cm, index=[f"gold {i}" for i in range(NUM_LABELS)],
                       columns=[f"pred {i}" for i in range(NUM_LABELS)]).to_string())

    df.to_csv(out / "test_predictions.csv", index=False)
    (out / "test_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out / 'test_predictions.csv'} and {out / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
