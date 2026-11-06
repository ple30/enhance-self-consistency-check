"""
Evaluate a final model from train_cv_mlp.py or train_cv_ordinal.py (--final-fit) on the
annotation Excel files.

Reads every Excel file in annotation_data (--excel-dir) and every sheet, and predicts the
perturbation strength (0-3) of every text pair that has both texts (CoT files: "Full
original CoT" -> "Full edited CoT"; input files: "Question" -> "Edited Question"), using
the model and its fitted preprocessing from final_model.joblib and the same frozen
sentence-transformer and settings as in training. Question texts are cut before "Options"
exactly as in training (models trained with --keep-options, or by versions before this
option existed, embed the full text). The Excel files and output CSVs keep the full texts.

Metrics are computed on the pairs that have a human score, after the same cleaning as the
training data (xlsx_to_csv.clean_train: whole-number scores 0-3, duplicates dropped,
conflicting duplicates keep the first): linear-weighted kappa and Pearson correlation (plus
accuracy, macro-F1, MAE), overall, per domain and per source file, with a confusion matrix.
Scored pairs whose original text (group) also occurs in the training data are flagged, and
metrics are also reported without them, since they are not fully unseen.

Every predicted pair, scored or not, is then written to a column "mlp" for MLP models and
"ordinal" for ordinal-regression models (--excel-column sets another name):
reused if it already exists (old values are cleared first), otherwise added after the last
column. Files are saved in place, like the "gemini" column of prompt_perturbation_strength.py;
use --excel-out-dir to write copies instead, or --no-write to only compute metrics. Close the
files in Excel first. In a git clone, `git checkout -- annotation_data` restores the originals.

Example
    python train_cv_mlp.py --embedding-model sentence-transformers/all-MiniLM-L6-v2 --hidden 32 \
        --swap-aug --seeds 42 43 44 45 46 --final-fit --output-dir results/mlp_minilm
    python evaluate_mlp.py --model-path results/mlp_minilm/final_model.joblib
    python train_cv_ordinal.py --seeds 42 43 44 45 46 --final-fit --output-dir results/ordinal
    python evaluate_mlp.py --model-path results/ordinal/final_model.joblib

Outputs (default: <model dir>/test/): test_predictions.csv (scored pairs used for the
metrics), test_metrics.json, excel_predictions.csv (every pair predicted), and the "mlp"
(or "ordinal") column in the Excel files.
"""
import argparse
import glob
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from openpyxl import load_workbook
from sklearn.metrics import confusion_matrix

import train_cv_mlp
from train_cv_mlp import (DOMAIN_NAMES, METRICS, MLP, NUM_LABELS, compute_metrics, embed_pairs,
                          drop_swapped, get_device, load_data, pair_features, prepare_texts)
from xlsx_to_csv import COLUMNS, clean_train, load_domain

BASE_DIR = Path(__file__).parent


def load_bundle(path):
    # Models saved before train_cv_mlp.py ran through its module reference __main__.Preprocessor
    import __main__
    if not hasattr(__main__, "Preprocessor"):
        __main__.Preprocessor = train_cv_mlp.Preprocessor
    bundle = joblib.load(path)
    if bundle.get("model_type") == "ordinal":  # from train_cv_ordinal.py
        return bundle
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


def predict(df, bundle, train_args, device):
    """Predicted strength (0-3) and class probabilities for every pair in df."""
    U, V = embed_pairs(df, train_args, device)
    if bundle.get("model_type") == "ordinal":
        U, V = U.astype(np.float64), V.astype(np.float64)
        X = bundle["preprocessor"].transform(U, V, df["domain"].values)
        probs = bundle["model"].predict_proba(X)
        return bundle["model"].decide(probs, train_args.decision), probs
    X = bundle["preprocessor"].transform(pair_features(U, V), df["domain"].values)
    probs = predict_proba(bundle, X, device)
    return probs.argmax(-1), probs


# ----------------------------------------------------------------------------
# Excel: predict every text pair in every sheet, write a prediction column
# ----------------------------------------------------------------------------
EXCEL_PATTERNS = {0: "*_cot_*.xlsx", 1: "*_input_*.xlsx"}  # domain -> file pattern


def read_excel_pairs(excel_dir):
    """All text pairs in all sheets (scored or not), with file path, sheet and Excel row."""
    frames = []
    for dom, pattern in EXCEL_PATTERNS.items():
        files = sorted(glob.glob(str(Path(excel_dir) / pattern)))
        if files:
            d = load_domain(files, COLUMNS[dom]["col_a"], COLUMNS[dom]["col_b"], dom,
                            require_score=False)
            d["path"] = d["source_file"].map({Path(f).name: f for f in files})
            d["col_a_name"] = COLUMNS[dom]["col_a"]
            frames.append(d)
    if not frames:
        raise SystemExit(f"No *_cot_*.xlsx or *_input_*.xlsx files found in {excel_dir}")
    df = pd.concat(frames, ignore_index=True)
    is_text = lambda s: s.notna() & (s.astype(str).str.strip() != "")
    df = df[is_text(df["text_a"]) & is_text(df["text_b"])].copy()
    # same text normalisation as xlsx_to_csv.clean_train
    df["text_a"] = df["text_a"].astype(str).str.strip()
    df["text_b"] = df["text_b"].astype(str).str.strip()
    return df.reset_index(drop=True)


def find_column(ws, name):
    """1-based index of the header cell equal to `name` (case/space-insensitive), or None."""
    for cell in ws[1]:
        if cell.value is not None and str(cell.value).strip().lower() == name.strip().lower():
            return cell.column
    return None


def write_excel_predictions(df, column, out_dir=None):
    """Write df["pred"] into `column` of each sheet at each pair's Excel row."""
    written = {}
    for path, file_rows in df.groupby("path"):
        wb = load_workbook(path)
        for sheet, rows in file_rows.groupby("sheet"):
            ws = wb[sheet]
            a_col = find_column(ws, rows["col_a_name"].iloc[0])
            # safety check: the Excel row must hold the text that was predicted
            for r, text in zip(rows["row"], rows["text_a"]):
                cell = ws.cell(row=int(r), column=a_col).value
                if str(cell).strip() != text:
                    raise SystemExit(f"{Path(path).name} [{sheet}] row {r}: text does not match "
                                     "the row read by pandas; nothing was written to this file.")
            col = find_column(ws, column)
            if col is None:
                col = ws.max_column + 1
                ws.cell(row=1, column=col, value=column)
            for r in range(2, ws.max_row + 1):  # clear predictions from an earlier run
                ws.cell(row=r, column=col).value = None
            for r, pred in zip(rows["row"], rows["pred"]):
                ws.cell(row=int(r), column=col, value=int(pred))
            written[(Path(path).name, sheet)] = len(rows)
        target = Path(out_dir) / Path(path).name if out_dir else Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        wb.save(target)
    return written


def scored_pairs(xl):
    """The pairs used for metrics: scored rows after the same cleaning as the training data."""
    print("Cleaning the scored pairs (same rules as the training data):")
    return drop_swapped(clean_train(xl)).reset_index(drop=True)


def fmt(m):
    return " | ".join(f"{k} {m[k]:.3f}" for k in METRICS)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True,
                   help="final_model.joblib from train_cv_mlp.py or train_cv_ordinal.py --final-fit")
    p.add_argument("--train-data", nargs="+", default=None,
                   help="training CSV(s), to flag test pairs sharing an original text "
                        "(default: the files the model was trained on)")
    p.add_argument("--cache-dir", default=None, help="embedding cache (default: as in training)")
    p.add_argument("--output-dir", default=None, help="default: <model dir>/test")
    p.add_argument("--excel-dir", default=str(BASE_DIR / "annotation_data"),
                   help="folder with the *_cot_*.xlsx / *_input_*.xlsx annotation files")
    p.add_argument("--excel-column", default=None,
                   help='name of the prediction column (default: "mlp", or "ordinal" for '
                        'ordinal-regression models)')
    p.add_argument("--excel-out-dir", default=None,
                   help="save the updated Excel files here instead of overwriting them")
    p.add_argument("--no-write", action="store_true",
                   help="only compute metrics; do not write predictions to the Excel files")
    args = p.parse_args()

    bundle = load_bundle(args.model_path)
    train_args = argparse.Namespace(**bundle["args"])
    if args.cache_dir:
        train_args.cache_dir = args.cache_dir
    out = Path(args.output_dir or Path(args.model_path).parent / "test")
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()

    ordinal = bundle.get("model_type") == "ordinal"
    if args.excel_column is None:
        args.excel_column = "ordinal" if ordinal else "mlp"
    model_desc = (f"ordinal logistic regression (alpha {bundle['alpha']:g}) | pair mode "
                  f"{train_args.pair_mode}" if ordinal else
                  f"MLP hidden {train_args.hidden or 'none (linear)'} | input [u, v] + domain")
    print(f"Model: {model_desc} | "
          f"embeddings {train_args.embedding_model} | Question texts "
          + ("in full" if getattr(train_args, "keep_options", True) else "cut before 'Options'"))

    xl = read_excel_pairs(args.excel_dir)
    print(f"Excel: {len(xl)} text pairs in {xl['path'].nunique()} files "
          f"({int(xl['label'].notna().sum())} have a human score)")
    # models saved before the option existed were trained on full Question texts
    keep_options = getattr(train_args, "keep_options", True)
    xl["pred"], xl_probs = predict(prepare_texts(xl, keep_options), bundle, train_args, device)
    for c in range(NUM_LABELS):
        xl[f"p{c}"] = xl_probs[:, c]

    df = scored_pairs(xl)
    train_files = args.train_data or [f for f in train_args.data if Path(f).exists()]
    if not train_files:  # trained elsewhere (e.g. another Colab path): fall back to the repo copy
        train_files = [str(BASE_DIR / "classifier_train_data/train.csv")]
    train_groups = set(load_data(train_files)["group"])
    df["seen_original"] = df["group"].isin(train_groups)
    print(f"Test: {len(df)} scored pairs | {int(df['seen_original'].sum())} share an original "
          f"text with the training data")

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
    results.update(n=len(df), model_path=str(args.model_path), excel_dir=str(args.excel_dir))

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

    helper_cols = ["path", "col_a_name"]
    df.drop(columns=helper_cols).to_csv(out / "test_predictions.csv", index=False)
    (out / "test_metrics.json").write_text(json.dumps(results, indent=2))
    xl.drop(columns=helper_cols).to_csv(out / "excel_predictions.csv", index=False)
    print(f"\nWrote {out / 'test_predictions.csv'}, {out / 'test_metrics.json'} and "
          f"{out / 'excel_predictions.csv'}")

    if not args.no_write:
        written = write_excel_predictions(xl, args.excel_column, args.excel_out_dir)
        print(f"\nWriting column '{args.excel_column}':")
        for (f, sheet), n in written.items():
            print(f"  {f:36s} [{sheet}] {n} predictions")
        print(f"Wrote column '{args.excel_column}' to the Excel files in "
              f"{args.excel_out_dir or args.excel_dir}")


if __name__ == "__main__":
    main()
