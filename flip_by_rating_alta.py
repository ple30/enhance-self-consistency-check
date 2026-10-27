import zipfile
import json
from pathlib import Path

base_dir = Path(__file__).parent

from collections import defaultdict
def flip_by_rating_cot(dataset, model):
    # load ratings file into a dict keyed by (question_id, step_idx)
    ratings = {}
    with open(base_dir / 'perturbation_strength_result' / f'{dataset}_{model}_ratings.jsonl') as f:
        for line in f:
            entry = json.loads(line)
            key = (entry["question_id"], entry["step_idx"])
            ratings[key] = entry["rating"]

    # load predictions file and join
    flip_by_rating_result = defaultdict(lambda: {"total": 0, "flipped": 0, "percentage": 0})

    with zipfile.ZipFile(base_dir / "fur_add_mistake_stats.zip") as z:
        # see all contents and their paths
        target_files = [f for f in z.namelist() if f.startswith(f"mistake_stats/{dataset}/{model}/") and f.endswith(".jsonl")]
        
        # read a specific file by its full path inside the zip
        
        for filepath in target_files:
            with z.open(filepath) as f:
                for line in f:
                    entry = json.loads(line)
                    key = (entry["id"], entry["step_idx"])
                    if key not in ratings:
                        continue

                    rating = ratings[key]
                    if rating is None:
                        continue
                    flip_by_rating_result[rating]["total"] += 1
                    if entry["mistake_flipped"]:
                        flip_by_rating_result[rating]["flipped"] += 1

    for rating in sorted(flip_by_rating_result):
        total = flip_by_rating_result[rating]["total"]
        flipped = flip_by_rating_result[rating]["flipped"]
        pct = flipped / total * 100
        flip_by_rating_result[rating]["percentage"] = pct

    return dict(flip_by_rating_result)

def flip_by_rating_input(dataset, model, pred_dir):
    # load ratings file into a dict keyed by question_id
    ratings = {}
    with open(base_dir / 'perturbation_strength_result' / 'input_perturbation' / f'{dataset}_{model}_ratings.jsonl') as f:
        for line in f:
            entry = json.loads(line)
            key = entry["question_id"]
            ratings[key] = entry["rating"]

    # load predictions file and join
    flip_by_rating_result = defaultdict(lambda: {"total": 0, "flipped": 0, "percentage": 0})

    flip_results_path = base_dir / pred_dir / 'flip_results' / f'{dataset}_{model}_flip_results.jsonl'

    with open(flip_results_path) as f:
        for line in f:
            entry = json.loads(line)
            key = entry["id"]

            if key not in ratings:
                continue

            rating = ratings[key]
            if rating is None:
                continue

            flip_by_rating_result[rating]["total"] += 1
            if entry["mistake_flipped"]:
                flip_by_rating_result[rating]["flipped"] += 1

    for rating in sorted(flip_by_rating_result):
        total = flip_by_rating_result[rating]["total"]
        flipped = flip_by_rating_result[rating]["flipped"]
        pct = flipped / total * 100
        flip_by_rating_result[rating]["percentage"] = pct

    return dict(flip_by_rating_result)

import matplotlib.pyplot as plt
import numpy as np

def plot_flip_rates(cot_percentage, input_cot_percentage, input_direct_percentage, dataset_title):
    strength = ['0', '1', '2', '3']
    x = np.arange(len(strength))
    width = 0.35

    data = [
        {"title": "Phi-3 (4B)", "CoT Perturbation": cot_percentage['Phi-3'], "Input CoT prompting": input_cot_percentage['Phi-3'], "Input Direct Prompting": input_direct_percentage['Phi-3']},
        {"title": "LLaMA-3 (8B)", "CoT Perturbation": cot_percentage['LLaMA-3'], "Input CoT prompting": input_cot_percentage['LLaMA-3'], "Input Direct Prompting": input_direct_percentage['LLaMA-3']},
        {"title": "Mistral-2 (7B)", "CoT Perturbation": cot_percentage['Mistral-2'], "Input CoT prompting": input_cot_percentage['Mistral-2'], "Input Direct Prompting": input_direct_percentage['Mistral-2']},
        {"title": "LLaMA-3-3B (3B)", "CoT Perturbation": cot_percentage['LLaMA-3-3B'], "Input CoT prompting": input_cot_percentage['LLaMA-3-3B'], "Input Direct Prompting": input_direct_percentage['LLaMA-3-3B']},
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        d = data[i]
        bars1 = ax.bar(x - width/2, d["CoT Perturbation"], width, label='CoT', color='steelblue')
        bars3 = ax.bar(x + width/2, d["Input Direct Prompting"], width, label='Input', color='coral')

        ax.set_title(d["title"], fontsize=17, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(strength, fontsize=13)
        if i == 0:
            ax.set_ylabel('Flip Rate (%)', fontsize=15)
        ax.set_xlabel('Perturbation Strength', fontsize=16)
        ax.legend(fontsize=11)

        # Value labels on bars
        for bars in [bars1, bars3]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(int(bar.get_height())), ha='center', va='bottom', fontsize=10)

    plt.suptitle(f'{dataset_title} Flip Rates', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{dataset_title.replace(" ", "_")}_flip_rates.pdf')
    print('Flip rates plot saved to', f'{dataset_title.replace(" ", "_")}_flip_rates.pdf')
    #plt.show()


def mistake_result_per_id(dataset, model):
    """Load first file, keeping only the first occurrence of each id."""
    first_occurrences = {}
    with zipfile.ZipFile(base_dir / "fur_add_mistake_stats.zip") as z:
        # see all contents and their paths
        target_files = [f for f in z.namelist() if f.startswith(f"mistake_stats/{dataset}/{model}/") and f.endswith(".jsonl")]
        
        # read a specific file by its full path inside the zip
        
        for filepath in target_files:
            with z.open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    id_ = record["id"]
                    if id_ not in first_occurrences:
                        first_occurrences[id_] = record
    return first_occurrences

def get_flip_result(dataset, model, prediction_file, output_path):
    mistake_result = mistake_result_per_id(dataset, model)
    results = []
    with open(prediction_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            id_ = record["id"]
            question = record["question"]
            options = record["options"]
            edited_question = record["edited_question"]
            segmented_cot = record["segmented_cot"]
            mistake_prediction = record["mistake_prediction"]

            if id_ not in mistake_result:
                print(f"Warning: id '{id_}' not found in mistake result file, skipping.")
                continue

            cot_prediction = mistake_result[id_]["cot_prediction"]
            mistake_flipped = (cot_prediction != mistake_prediction)

            results.append({"id": id_, "question": question, "options": options, "edited_question": edited_question, "segmented_cot": segmented_cot, "mistake_flipped": mistake_flipped, "mistake_prediction": mistake_prediction})

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Done. {len(results)} records written to {output_path}")

datasets = ['arc-challenge', 'openbook', 'sports', 'sqa']
models = ['LLaMA-3-3B', 'Mistral-2', 'Phi-3', 'LLaMA-3']

cot_pred_dir = "input_perturbation_cot_predictions"
direct_pred_dir = "input_perturbation_predictions"

for pred_dir, filename_suffix in [(cot_pred_dir, "cot_predictions"), (direct_pred_dir, "predictions")]:
    result_dir = base_dir / pred_dir / "flip_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        for model in models:
            prediction_file = base_dir / pred_dir / f"{dataset}_{model}_{filename_suffix}.jsonl"
            output_path = result_dir / f"{dataset}_{model}_flip_results.jsonl"
            get_flip_result(dataset, model, prediction_file, output_path)


title_for_dataset = {
    'arc-challenge': 'ARC-Challenge',
    'openbook': 'OpenBookQA',
    'sports': 'Sports',
    'sqa': 'StrategyQA',
}

for dataset in datasets:
    cot_percentage = {'LLaMA-3-3B': [], 'Mistral-2': [], 'Phi-3': [], 'LLaMA-3': []}
    input_cot_percentage = {'LLaMA-3-3B': [], 'Mistral-2': [], 'Phi-3': [], 'LLaMA-3': []}
    input_direct_percentage = {'LLaMA-3-3B': [], 'Mistral-2': [], 'Phi-3': [], 'LLaMA-3': []}
    for model in models:
        cot_result = flip_by_rating_cot(dataset, model)
        input_cot_result = flip_by_rating_input(dataset, model, cot_pred_dir)
        input_direct_result = flip_by_rating_input(dataset, model, direct_pred_dir)
        for rating in sorted(cot_result):
            cot_percentage[model].append(cot_result[rating]["percentage"])
        for rating in sorted(input_cot_result):
            input_cot_percentage[model].append(input_cot_result[rating]["percentage"])
        for rating in sorted(input_direct_result):
            input_direct_percentage[model].append(input_direct_result[rating]["percentage"])

    plot_flip_rates(cot_percentage, input_cot_percentage, input_direct_percentage, dataset_title=title_for_dataset[dataset])

'''print("GAP COT")
print(gap_cot)
print("GAP INPUT")
print(gap_input)

flip_result = base_dir / "input_perturbation_predictions" / "flip_results" / "arc-challenge_LLaMA-3_flip_results.jsonl"
ratings_file = base_dir / "perturbation_strength_result" / "input_perturbation" / "arc-challenge_LLaMA-3_ratings.jsonl"  # has rating

# Load ratings keyed by id
ratings = {}
with open(ratings_file) as f:
    for line in f:
        entry = json.loads(line)
        ratings[entry["question_id"]] = entry["rating"]

# Check conditions and print
with open(flip_result) as f:
    for line in f:
        entry = json.loads(line)
        id_ = entry["id"]

        if ratings.get(id_) == 0 and entry["mistake_flipped"] == True:
            print(f"id: {entry['id']}")
            print(f"Question:        {entry['question']}")
            print(f"Options:         {entry['options']}")
            print(f"Edited Question: {entry['edited_question']}")
            print(f"Segmented CoT:   {entry['segmented_cot']}")
            print(f"Mistake prediction: {entry['mistake_prediction']}")
            print("-" * 60)


ratings_file = base_dir / "perturbation_strength_result" / "arc-challenge_LLaMA-3_ratings.jsonl"  # has rating

# Load ratings keyed by id
ratings = {}
with open(ratings_file) as f:
    for line in f:
        entry = json.loads(line)
        key = (entry["question_id"], entry["step_idx"])
        ratings[key] = entry["rating"]

with zipfile.ZipFile(base_dir / "fur_add_mistake_stats.zip") as z:
        # see all contents and their paths
        target_files = [f for f in z.namelist() if f.startswith(f"mistake_stats/arc-challenge/LLaMA-3/") and f.endswith(".jsonl")]
        
        # read a specific file by its full path inside the zip
        
        for filepath in target_files:
            with z.open(filepath) as f:
                for line in f:
                    entry = json.loads(line)
                    id_ = entry["id"]
                    step_idx = entry["step_idx"]

                    if ratings.get((id_, step_idx)) == 3 and entry["mistake_flipped"] == False:
                        print(f"id: {entry['id']}")
                        print(f"step: {entry['step_idx']}")
                        print("-" * 60)
'''
