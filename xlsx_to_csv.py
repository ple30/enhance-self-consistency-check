"""
Prepare text-pair data for fine-tuning the cross-encoder.

Train and test Excel files live in separate folders. This script prepares the
TRAIN data (train/val DataFrames with columns text_a, text_b, domain (0 (CoT) / 1 (Question)),
perturbation strength label (0-3), source_file, sheet, row, group). 
"""
import glob
import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

base_dir = Path(__file__).parent

COLUMNS = {
    0: {"col_a": "Full original CoT", "col_b": "Full edited CoT"},  # domain 1
    1: {"col_a": "Question", "col_b": "Edited Question"},           # domain 2
}
FILES = {
    "train": {
        0: sorted(glob.glob(str(base_dir / "classifier_train_data/cot/*.xlsx"))),
        1: sorted(glob.glob(str(base_dir / "classifier_train_data/input/*.xlsx"))),
    },
    "test": {
        0: sorted(glob.glob("data/test/domain1/*.xlsx")),
        1: sorted(glob.glob("data/test/domain2/*.xlsx")),
    },
}
SCORE_COL = "score"
NUM_LEVELS = 4  # labels 0..3
SEED = 42


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _find_col(df, wanted, required=True):
    """Case/whitespace-insensitive column lookup."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    key = wanted.strip().lower()
    if key in lookup:
        return lookup[key]
    if required:
        raise KeyError(f"Column '{wanted}' not found. Available: {list(df.columns)}")
    return None


def load_domain(files, col_a, col_b, domain_id, require_score=True):
    """Read every sheet of every file and return one tidy DataFrame."""
    frames = []
    for path in files:
        sheets = pd.read_excel(path, sheet_name=None)  # dict: sheet name -> DataFrame
        for sheet_name, df in sheets.items():
            where = f"{os.path.basename(path)} / sheet '{sheet_name}'"
            try:
                a, b = _find_col(df, col_a), _find_col(df, col_b)
                s = _find_col(df, SCORE_COL, required=require_score)
            except KeyError as e:
                raise KeyError(f"{where}: {e}")
            sub = pd.DataFrame({
                "text_a": df[a],
                "text_b": df[b],
                "label": df[s] if s is not None else float("nan"),
            })
            sub["domain"] = domain_id
            sub["source_file"] = os.path.basename(path)
            sub["sheet"] = sheet_name
            sub["row"] = df.index + 2  # Excel row number (row 1 is the header)
            frames.append(sub)
    if not frames:
        raise FileNotFoundError(f"No files found for domain {domain_id}. Check the glob pattern.")
    return pd.concat(frames, ignore_index=True)


def load_split(split, require_score=True):
    parts = [
        load_domain(FILES[split][dom], COLUMNS[dom]["col_a"], COLUMNS[dom]["col_b"],
                    dom, require_score=require_score)
        for dom in COLUMNS
    ]
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def clean_train(df):
    """Training data: drop bad rows, coerce types, remove duplicates."""
    n0 = len(df)
    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["text_a", "text_b", "label"])
    df["text_a"] = df["text_a"].astype(str).str.strip()
    df["text_b"] = df["text_b"].astype(str).str.strip()
    df = df[(df["text_a"] != "") & (df["text_b"] != "")]

    # keep only integer labels inside the valid range
    df = df[df["label"] == df["label"].round()]
    df["label"] = df["label"].astype(int)
    df = df[df["label"].between(0, NUM_LEVELS - 1)]

    # exact duplicates
    df = df.drop_duplicates(subset=["text_a", "text_b", "domain", "label"])

    # same pair with conflicting labels -> warn (keep first)
    conflicts = df.duplicated(subset=["text_a", "text_b", "domain"], keep=False).sum()
    if conflicts:
        print(f"WARNING: {conflicts} train rows share a pair but have different labels; keeping first.")
        df = df.drop_duplicates(subset=["text_a", "text_b", "domain"], keep="first")

    print(f"clean_train(): {n0} -> {len(df)} rows")
    # rows sharing the same original text (within a domain) must stay in the same
    # split, otherwise the same original appears in train and validation (leakage)
    df["group"] = df["domain"].astype(str) + "||" + df["text_a"]
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Train / validation split (grouped, per domain)
# ----------------------------------------------------------------------------
def split_train_val(df, val_frac=0.1, seed=SEED):
    train, val = [], []
    for _, d in df.groupby("domain"):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        i_tr, i_va = next(gss.split(d, groups=d["group"]))
        train.append(d.iloc[i_tr]), val.append(d.iloc[i_va])
    shuffle = lambda xs: pd.concat(xs).sample(frac=1, random_state=seed).reset_index(drop=True)
    return shuffle(train), shuffle(val)


def add_swapped(df):
    """Swap augmentation: add (B, A) copies. Apply to training data only (used by train_cv.py)."""
    swapped = df.copy()
    swapped["text_a"], swapped["text_b"] = df["text_b"].values, df["text_a"].values
    return pd.concat([df, swapped], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    train_all = clean_train(load_split("train"))

    print("\nTrain label distribution per domain:")
    print(pd.crosstab(train_all["domain"], train_all["label"]))

    train, val = split_train_val(train_all, val_frac=0.1)
    print(f"\ntrain={len(train)}, val={len(val)}")

    # Save original-direction pairs only. Swap augmentation is applied at training time
    # (train_cv.py --swap-aug) so it only ever touches the training folds.
    train.to_csv(base_dir / "classifier_train_data/train.csv", index=False)
    val.to_csv(base_dir / "classifier_train_data/val.csv", index=False)