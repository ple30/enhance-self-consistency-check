from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import numpy as np

def get_folder_scores(folder_path, suffix, machine_score='cosine distance'):

    folder_s1, folder_s2 = [], []

    # Iterate through all excel files in the folder
    for filepath in sorted(folder_path.glob(suffix)):
        try:
            xl = pd.ExcelFile(filepath)
        except Exception as e:
            print(f"  Could not read {filepath.name}: {e}")
            continue

        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)

            # Normalize column names
            df.columns = df.columns.str.strip().str.lower()

            human_score = 'score'

            if human_score not in df.columns or machine_score not in df.columns:
                continue

            # Convert to numeric, forcing errors to NaN
            human_scores = pd.to_numeric(df[human_score], errors='coerce')
            machine_scores = pd.to_numeric(df[machine_score], errors='coerce')

            # --- MODIFICATION 1: Remove 0-1 filter for cosine distance ---
            # Now only filtering for scores 0-3 and ensuring both are numbers
            mask = (
                human_scores.notna() & machine_scores.notna() &
                human_scores.between(0, 3)
            )
            
            folder_s1.extend(human_scores[mask].tolist())
            folder_s2.extend(machine_scores[mask].tolist())

    return folder_s1, folder_s2
def compute_folder_correlation(folder_path, suffix, machine_score='cosine distance'):
    folder_s1, folder_s2 = get_folder_scores(folder_path, suffix, machine_score)

    # --- MODIFICATION 2: Compute correlation for ALL files combined ---
    if len(folder_s1) < 2:
        print(f"  Not enough valid rows in folder {folder_path.name}")
        return

    pearson_r, _ = pearsonr(folder_s1, folder_s2)
    spearman_r, _ = spearmanr(folder_s1, folder_s2)

    print(f"  Total Data Points (n): {len(folder_s1)}")
    plt.figure(figsize=(8, 5))  
    plt.scatter(folder_s1, folder_s2)
    plt.xlabel('True Perturbation Strengths')
    plt.ylabel('Change in Surprisals')
    plt.title(f'Change in Surprisals vs True Perturbation Strengths for CoT perturbations')
    plt.show()
    return pearson_r, spearman_r

annotation_folder = Path(__file__).parent / 'annotation_data'
correlation_results = {}
# Run for both folders
for suffix in ['*input_annotation.xlsx', '*cot_annotation.xlsx']:
    for score_method in ['surprisal']:
        print(f"\n--- Correlation for {score_method} in {suffix} ---")
        pearson_r, spearman_r = compute_folder_correlation(annotation_folder, suffix, machine_score=score_method)
        
        # Store results
        if suffix not in correlation_results:
            correlation_results[suffix] = {}

        correlation_results[suffix][score_method] = {
            'pearson': pearson_r,
            'spearman': spearman_r
        }

# --- NEW: Create Table and Save to Excel ---
all_tables = []

for folder_name, methods in correlation_results.items():
    # Convert nested dict to DataFrame
    df_result = pd.DataFrame(methods).T # Transpose to get methods as rows
    df_result.index.name = 'method'
    df_result = df_result.reset_index()
    df_result['folder'] = folder_name
    all_tables.append(df_result)

# Combine all folders into one large DataFrame
final_df = pd.concat(all_tables)

# Save to CSV
output_file = Path(__file__).parent / 'correlation_results(1).csv'
final_df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")