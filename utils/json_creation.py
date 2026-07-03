import os
import json

import random
from collections import defaultdict

def create_split_datasets(json_paths, val_split=0.1):

    # 1. Group data by patient/sample
    patient_data = defaultdict(list)
    sample_names = [json_path.split("/")[-2] for json_path in json_paths]
    
    for i, json_path in enumerate(json_paths):        
        # Use your parser function (from earlier) for this specific sample
        sample_dicts = create_doserad_dataset(json_path)
        patient_data[sample_names[i]].extend(sample_dicts)
        
    # 2. Shuffle the patients, NOT the individual beamlets
    random.seed(42) # Exact reproducibility
    random.shuffle(sample_names)
    
    # 3. Calculate split index based on the number of patients
    split_idx = int(len(sample_names) * (1 - val_split))
    train_patients = sample_names[:split_idx]
    val_patients = sample_names[split_idx:]
    
    # 4. Flatten back into MONAI-ready lists
    train_list = []
    for p in train_patients:
        train_list.extend(patient_data[p])
        
    val_list = []
    for p in val_patients:
        val_list.extend(patient_data[p])
        
    print(f"Training Samples: {len(train_list)} (from {len(train_patients)} patients)")
    print(f"Validation Samples: {len(val_list)} (from {len(val_patients)} patients)")
    
    return train_list, val_list


def create_doserad_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    dataset_list = []
    base_dir = os.path.dirname(json_path)
    ct_path = os.path.join(base_dir, "image", "ct.mha")
    
    for beam in data.get('beams', []):
        beam_idx = beam['beam_idx']
        
        for ray in beam.get('rays', []):
            ray_idx = ray['ray_idx']
            ray_source = ray['ray_source'] # Expected format: [x, y, z] in mm
            ray_target = ray['ray_target'] # Expected format: [x, y, z] in mm
            
            for beamlet in ray.get('beamlets', []):
                beamlet_idx = beamlet['beamlet_idx']
                energy = beamlet['energy']
                
                # Reconstruct the specific dose filename
                dose_filename = f"Dose_B{beam_idx}_R{ray_idx}_L{beamlet_idx}.mha"
                dose_path = os.path.join(base_dir, "dose", dose_filename)
                
                if os.path.exists(dose_path) and os.path.exists(ct_path):
                    dataset_list.append({
                        "ct": ct_path,
                        "gt_dose": dose_path,
                        "ray_source_phys": ray_source,
                        "ray_target_phys": ray_target,
                        "energy": energy
                    })
                    
    return dataset_list


if __name__ == "__main__":
    import glob
    import numpy as np
<<<<<<< HEAD
    jsonPaths = glob.glob('data/LMUK-RADONC-PHYS-RES__DoseRAD2026/proton/training/*/*.json')
    def format_dict(item):
        src = np.array(item["ray_source_phys"])
        tgt = np.array(item["ray_target_phys"])
        
        # Calculate the direction vector (Target - Source)
        delta_x = tgt[0] - src[0]
        delta_y = tgt[1] - src[1]
        
        # Normalize to a unit vector (length 1)
        magnitude = np.sqrt(delta_x**2 + delta_y**2)
        dir_x = delta_x / magnitude if magnitude > 0 else 0.0
        dir_y = delta_y / magnitude if magnitude > 0 else 0.0
        
        return {
            "ct": item["ct"],
            "gt_dose": item["gt_dose"],
            "ray_source": item["ray_source_phys"],
            "ray_target": item["ray_target_phys"],
            # Sleek, 3-element physics vector: [Angle X, Angle Y, Energy]
            "condition": [float(dir_x), float(dir_y), float(item["energy"])]
        }
=======
    jsonPaths = glob.glob('/home/nr_dodb/nr_dose_scratch/training/*/*.json')
    print(f"Found {len(jsonPaths)} JSON files for dataset creation.")
>>>>>>> 7a4f25f (works on komondor)
    
    dataset_list = [create_doserad_dataset(json_path) for json_path in jsonPaths]
    dataset_list = [item for sublist in dataset_list for item in sublist]
    
    # Format and save Full Dataset
    formatted_dataset_list = [format_dict(item) for item in dataset_list]
    json.dump(formatted_dataset_list, open("data/dataset_list.json", "w"), indent=4)

    # Generate Train/Val Splits (Using your existing function)
    train_list, val_list = create_split_datasets(jsonPaths)
    
    # Format and save Splits
    formatted_train_list = [format_dict(item) for item in train_list]
    formatted_val_list = [format_dict(item) for item in val_list]
    
    json.dump(formatted_train_list, open("data/train.json", "w"), indent=4)
    json.dump(formatted_val_list, open("data/val.json", "w"), indent=4)