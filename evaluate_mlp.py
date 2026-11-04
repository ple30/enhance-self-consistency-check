"""
Evaluate a final model from train_cv_mlp.py (--final-fit) on held-out test data.

Loads final_model.joblib (the MLP with its fitted preprocessing),
embeds the test pairs with the same frozen sentence-transformer and settings used in
training, predicts perturbation strength 0-3, and reports linear-weighted kappa and Pearson
correlation (plus accuracy, macro-F1, MAE): overall, per domain and per source file.

Test pairs whose original text (group) also occurs in the training data are flagged, and
metrics are also reported without them, since they are not fully unseen.

It then goes through every Excel file in annotation_data (--excel-dir), every sheet, and
predicts the strength of every text pair that has both texts (CoT files: "Full original CoT"
-> "Full edited CoT"; input files: "Question" -> "Edited Question"), including rows without
a human score. The prediction is written to a column "mlp" (--excel-column): reused if it
already exists (old values are cleared first), otherwise added after the last column. Files
are saved in place, like the "gemini" column of prompt_perturbation_strength.py; use
--excel-out-dir to write copies instead, or --no-excel to skip this step. Close the files in
Excel first. In a git clone, `git checkout -- annotation_data` restores the originals.

Example
    python train_cv_mlp.py --embedding-model sentence-transformers/all-MiniLM-L6-v2 --hidden 32 \
        --swap-aug --seeds 42 43 44 45 46 --final-fit --output-dir results/mlp_minilm
    python evaluate_mlp.py --model-path results/mlp_minilm/final_model.joblib

Outputs (default: <model dir>/test/): test_predictions.csv, test_metrics.json,
excel_predictions.csv (every Excel row predicted), and the "mlp" column in the Excel files.
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
                          get_device, load_data, pair_features)
from xlsx_to_csv import COLUMNS, load_domain

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


def predict(df, bundle, train_args, device):
    """Predicted strength (0-3) and class probabilities for every pair in df."""
    U, V = embed_pairs(df, train_args, device)
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
    # same text normalisation as xlsx_to_csv.clean_train, so predictions match test.csv
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
    p.add_argument("--excel-dir", default=str(BASE_DIR / "annotation_data"),
                   help="folder with the *_cot_*.xlsx / *_input_*.xlsx files to add predictions to")
    p.add_argument("--excel-column", default="mlp", help="name of the prediction column")
    p.add_argument("--excel-out-dir", default=None,
                   help="save the updated Excel files here instead of overwriting them")
    p.add_argument("--no-excel", action="store_true", help="skip writing predictions to Excel")
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

    df["pred"], probs = predict(df, bundle, train_args, device)
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

    if not args.no_excel:
        xl = read_excel_pairs(args.excel_dir)
        print(f"\nExcel: predicting {len(xl)} text pairs in {xl['path'].nunique()} files "
              f"({int(xl['label'].notna().sum())} have a human score)")
        xl["pred"], xl_probs = predict(xl, bundle, train_args, device)
        for c in range(NUM_LABELS):
            xl[f"p{c}"] = xl_probs[:, c]
        written = write_excel_predictions(xl, args.excel_column, args.excel_out_dir)
        for (f, sheet), n in written.items():
            print(f"  {f:36s} [{sheet}] {n} predictions")
        where = args.excel_out_dir or args.excel_dir
        print(f"Wrote column '{args.excel_column}' to the Excel files in {where}")
        xl.drop(columns=["path", "col_a_name"]).to_csv(out / "excel_predictions.csv", index=False)
        print(f"Wrote {out / 'excel_predictions.csv'}")


if __name__ == "__main__":
    main()
