from collections import defaultdict
from pathlib import Path
import json
from random import random
from google import genai
from google.genai import types
from prompt_perturbation_strength import client, perturbation_instruction, create_prompt_cot, extract_rating
import re


base_dir = Path(__file__).parent
result_dir = base_dir / 'perturbation_strength_result'
batch_requests_dir = base_dir / 'batch_requests'
result_dir.mkdir(parents=True, exist_ok=True)
batch_requests_dir.mkdir(parents=True, exist_ok=True)

# The main folders to iterate through
main_folders = ['sports', 'sqa']
cot_models = ['LLaMA-3-3B', 'LLaMA-3', 'Mistral-2', 'Phi-3']
for main_folder in main_folders:
    main_path = base_dir / main_folder
    with open(Path(__file__).parent / 'examples_for_llm' / f'{main_folder}_cot_example.txt', "r") as f:
        examples = f.read()
    
    subfolders = [main_path / model for model in cot_models if (main_path / model).is_dir()]
    for subfolder in subfolders:
        # Get the file in this subfolder
        files = list(subfolder.iterdir())
        
        file_path = files[0]  # Get the first (and only) file
        print(f"Processing: {file_path}")

        # Load the .out file (each line is a separate JSON object)
        with open(file_path, 'r') as f:
            instances = [json.loads(line) for line in f if line.strip()]
        
        # Group instances by question id
        grouped_data = defaultdict(list)
        for instance in instances:
            grouped_data[instance['id']].append(instance)
        # Process each group
        batch_request_file = batch_requests_dir / f'{main_folder}_{subfolder.name}_batch_requests.jsonl'
        with open(batch_request_file, 'w', encoding='utf-8') as f:
            for question_id, instances in grouped_data.items():
                first = instances[0]  # All instances share the same question/segmented_cot

                # 1. Question + Options
                question_options = first['question'] + '\nOptions\n' + '\n'.join(first['options'])

                # 2. Full correct COT
                full_cot = '\n'.join(first['segmented_cot'])

                # 3. Full mistake COT for each step
                for instance in instances:
                    segmented_cot = list(first['segmented_cot'])  # fresh copy each time
                    segmented_cot[instance['step_idx']] = instance['mistake_cot_step']
                    full_mistake_cot = '\n'.join(segmented_cot)
                    # Create prompt for LLM
                    prompt = create_prompt_cot(question_options, full_cot, full_mistake_cot, examples)
                    final_prompt = f"{perturbation_instruction}\n\n{prompt}"
                    # Construct the Gemini Batch API structure
                    batch_key = f"{question_id}_step_{instance['step_idx']}"
                    batch_entry = {"key": batch_key,
                        "request": {
                            "model": "gemini-3-flash-preview",
                            "contents": [
                                {
                                    "role": "user",
                                    "parts": [{"text": final_prompt}]
                                }
                            ],
                        }
                    }
                    f.write(json.dumps(batch_entry) + '\n')
'''                    
# The file you already made
original_file = batch_requests_dir / "openbook_Phi-3_batch_requests.jsonl"
# The new mini-version
test_file = batch_requests_dir / "mini_test.jsonl"

with open(original_file, 'r') as f_in:
    # Read the first 5 lines
    first_five = [next(f_in) for _ in range(5)]

with open(test_file, 'w') as f_out:
    # Write them to the new file
    f_out.writelines(first_five)


# 1. Upload the JSONL file
uploaded_file = client.files.upload(file=test_file, config=types.UploadFileConfig(display_name='my-batch-requests', mime_type='jsonl'))

# 2. Create the batch job
batch_job = client.batches.create(
    model="gemini-3-flash-preview",
    src=uploaded_file.name,
    config={'display_name': 'mini_test_batch_job'}
)

print(f"Batch job created: {batch_job.name}")

# 3. Check status (can take up to 24 hours)
# status = client.batches.get(name=batch_job.name)
print(f"Batch job status: {batch_job.state}")
'''
for main_folder in main_folders:
    for model in cot_models:
        batch_file = batch_requests_dir / f'{main_folder}_{model}_batch_requests.jsonl'
        if not batch_file.exists():
            continue

        print(f"Uploading: {batch_file.name}")
        uploaded_file = client.files.upload(
            file=batch_file,
            config=types.UploadFileConfig(
                display_name=batch_file.stem,
                mime_type='jsonl'
            )
        )

        print(f"Submitting: {batch_file.name}")
        batch_job = client.batches.create(
            model="gemini-3-flash-preview",
            src=uploaded_file.name,
            config={'display_name': batch_file.stem}
)
        
        print(f" Batch:{batch_file.stem} Job name: {batch_job.name}, State: {batch_job.state}")