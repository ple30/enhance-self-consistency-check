from collections import defaultdict
import os
from pathlib import Path
import json
from google import genai
from google.genai import types
from prompt_perturbation_strength import client, perturbation_instruction, create_prompt_cot, extract_rating


base_dir = Path(__file__).parent
result_dir = base_dir / 'perturbation_strength_result'
result_dir.mkdir(parents=True, exist_ok=True)

# The main folders to iterate through
main_folders = ['sports']
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
        results = []
        grouped_data = defaultdict(list)
        for instance in instances:
            grouped_data[instance['id']].append(instance)
        # Process each group
        result_file = result_dir / f'{main_folder}_{subfolder.name}_ratings.jsonl'
        # Load already completed pairs
        completed = set()
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    completed.add((entry['question_id'], entry['step_idx']))

        with open(result_file, 'a') as f:
            for question_id, instances in list(grouped_data.items())[120:]:
                first = instances[0]  # All instances share the same question/segmented_cot

                # 1. Question + Options
                question_options = first['question'] + '\nOptions\n' + '\n'.join(first['options'])

                # 2. Full correct COT
                full_cot = '\n'.join(first['segmented_cot'])

                # 3. Full mistake COT for each step
                for instance in instances:
                    # Skip if already processed
                    if (question_id, instance['step_idx']) in completed:
                        continue
                    segmented_cot = list(first['segmented_cot'])  # fresh copy each time
                    segmented_cot[instance['step_idx']] = instance['mistake_cot_step']
                    full_mistake_cot = '\n'.join(segmented_cot)
                    # Create prompt for LLM
                    prompt = create_prompt_cot(question_options, full_cot, full_mistake_cot, examples)
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        config=types.GenerateContentConfig(
                            system_instruction=perturbation_instruction),
                        contents=prompt
                    )
                    rating = extract_rating(response.text)
                    # Write immediately, don't accumulate in memory
                    f.write(json.dumps({
                        'question_id': question_id,
                        'step_idx': instance['step_idx'],
                        'rating': rating
                    }) + '\n')
        
