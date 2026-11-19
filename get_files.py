import json
from pathlib import Path
from constant import dataset_folders, models
from sample_cot import sample_ids_from_file

# Get the current directory (where the script is located)
base_dir = Path(__file__).parent

for folder in dataset_folders:
    dataset_path = base_dir / folder
    
    # Get the first subfolder only
    subfolders = [f for f in dataset_path.iterdir() if f.is_dir()]
    
    if subfolders:
        first_subfolder = subfolders[0]
        
        # Get the single file in this subfolder
        files = list(first_subfolder.iterdir())
        
        if files:
            file_path = files[0]  # Get the first (and only) file

            # Sample 80 IDs from the file
            sampled_ids = sample_ids_from_file(file_path, num_samples=80)
            # Save the sampled IDs in the dataset folder
            output_file = dataset_path / 'sampled_ids.json'
            with open(output_file, 'w') as f:
                json.dump(sampled_ids, f, indent=2)


for folder in dataset_folders:
    dataset_path = base_dir / folder

    # Read the sampled_ids.json file
    sampled_ids_path = dataset_path / 'sampled_ids.json'
    with open(sampled_ids_path, 'r') as f:
        sampled_ids = json.load(f)
    
    # Iterate through each model (subfolder)
    for model_name, id_list in sampled_ids.items():
        dataset_model_path = dataset_path / model_name
        
        # Get the single file in this subfolder
        files = list(dataset_model_path.iterdir())
        
        file_path = files[0]
        
        # Read the file
        with open(file_path, 'r') as f:
            all_instances = [json.loads(line) for line in f if line.strip()]
        
        # Sample instances with matching IDs
        sampled_instances = [
            instance for instance in all_instances 
            if instance.get('id') in id_list
        ]
        
        # Save the sampled instances back to the dataset folder
        output_file = dataset_path / f'{model_name}_sampled.json'
        with open(output_file, 'w') as f:
            json.dump(sampled_instances, f, indent=2)
        
        print(f"  Saved to: {output_file}")

'''
def sample_all(dataset, model, id_list):
    # Get the current directory (where the script is located)
    base_dir = Path(__file__).parent
    model_path = base_dir / dataset / model
    # Get the single file in this subfolder
    files = list(model_path.iterdir())
    file_path = files[0]  # Get the first (and only) file
    with open(file_path, 'r') as f:
        all_instances = [json.loads(line) for line in f if line.strip()]
    
    sampled_instances = [
        instance for instance in all_instances 
        if instance.get('id') in id_list
    ]
    # Save the sampled instances back to the main folder
    output_file = base_dir / dataset / f'{model}_sampled_full.json'
    with open(output_file, 'w') as f:
        json.dump(sampled_instances, f, indent=2)
    print(f"Sampled {len(sampled_instances)} instances for {dataset} - {model}")
'''