from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import os

annotation_folder = Path(__file__).parent / 'annotation_data'
input_annotation = Path(__file__).parent / 'input_annotation'
cot_annotation = Path(__file__).parent / 'cot_annotation_1'

def is_valid_score(series):
    """Check if all values in a series are numbers between 0 and 3."""
    try:
        numeric = pd.to_numeric(series, errors='coerce')
        return numeric.dropna().between(0, 3).all() and numeric.notna().any()
    except Exception:
        return False
'''
def compute_kappa_for_folder(folder):

    folder_s1, folder_s2 = [], []

    for filepath in sorted(folder.glob('*.xlsx')):
        print(f"Opening: {filepath}")
        xl = pd.ExcelFile(filepath)

        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            df.columns = df.columns.str.strip().str.lower()

            score_col = 'score'
            annotator_col = 'annotator 2 score'

            if score_col not in df.columns or annotator_col not in df.columns:
                print(f"  [{filepath.name}] Sheet '{sheet_name}': Skipped (missing required columns)")
                continue

            scores1 = pd.to_numeric(df[score_col], errors='coerce')
            scores2 = pd.to_numeric(df[annotator_col], errors='coerce')

            mask = (
                scores1.notna() & scores2.notna() &
                scores1.between(0, 3) & scores2.between(0, 3)
            )
            folder_s1.extend(scores1[mask].astype(int).tolist())
            folder_s2.extend(scores2[mask].astype(int).tolist())

    if len(folder_s1) < 2:
        print(f"  Skipped (not enough valid rows across all files)")
        return

    kappa = cohen_kappa_score(folder_s1, folder_s2, weights='linear', labels=[0, 1, 2, 3])
    print(f"  Linear Weighted Kappa = {kappa:.4f}  (n={len(folder_s1)})")
'''

def compute_kappa_for_folder(folder):

    folder_s1, folder_s2 = [], []
    folder_s1_low, folder_s2_low = [], []   # rows where score col is 0 or 3
    folder_s1_mid, folder_s2_mid = [], []   # rows where score col is 1 or 2

    for filepath in sorted(folder.glob('*.xlsx')):
        print(f"Opening: {filepath}")
        xl = pd.ExcelFile(filepath)

        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            df.columns = df.columns.str.strip().str.lower()

            score_col = 'score'
            annotator_col = 'annotator 2 score'

            if score_col not in df.columns or annotator_col not in df.columns:
                print(f"  [{filepath.name}] Sheet '{sheet_name}': Skipped (missing required columns)")
                continue

            scores1 = pd.to_numeric(df[score_col], errors='coerce')
            scores2 = pd.to_numeric(df[annotator_col], errors='coerce')

            base_mask = (
                scores1.notna() & scores2.notna() &
                scores1.between(0, 3) & scores2.between(0, 3)
            )

            folder_s1.extend(scores1[base_mask].astype(int).tolist())
            folder_s2.extend(scores2[base_mask].astype(int).tolist())

            # Filter by score_col only
            low_mask = base_mask & scores1.isin([0, 3])
            folder_s1_low.extend(scores1[low_mask].astype(int).tolist())
            folder_s2_low.extend(scores2[low_mask].astype(int).tolist())

            mid_mask = base_mask & scores1.isin([1, 2])
            folder_s1_mid.extend(scores1[mid_mask].astype(int).tolist())
            folder_s2_mid.extend(scores2[mid_mask].astype(int).tolist())

    if len(folder_s1) < 2:
        print(f"  Skipped (not enough valid rows across all files)")
        return

    kappa = cohen_kappa_score(folder_s1, folder_s2, weights='linear', labels=[0, 1, 2, 3])
    print(f"  [All]     Linear Weighted Kappa = {kappa:.4f}  (n={len(folder_s1)})")

    if len(folder_s1_low) >= 2:
        kappa_low = cohen_kappa_score(folder_s1_low, folder_s2_low, weights='linear', labels=[0, 1, 2, 3])
        print(f"  [score=0|3]  Linear Weighted Kappa = {kappa_low:.4f}  (n={len(folder_s1_low)})")
    else:
        print(f"  [score=0|3]  Skipped (not enough valid rows)")

    if len(folder_s1_mid) >= 2:
        kappa_mid = cohen_kappa_score(folder_s1_mid, folder_s2_mid, weights='linear', labels=[0, 1, 2, 3])
        print(f"  [score=1|2]  Linear Weighted Kappa = {kappa_mid:.4f}  (n={len(folder_s1_mid)})")
    else:
        print(f"  [score=1|2]  Skipped (not enough valid rows)")

compute_kappa_for_folder(input_annotation)
compute_kappa_for_folder(cot_annotation)