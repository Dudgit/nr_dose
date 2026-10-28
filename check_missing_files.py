import json
import os

def check_json_manifest(json_path, label="Dataset"):
    if not os.path.exists(json_path):
        print(f"[!] Error: JSON file not found at '{json_path}'")
        return []

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Standardize MONAI JSON structures (list of dicts or dict of lists)
    items_to_check = []
    if isinstance(data, list):
        items_to_check = data
    elif isinstance(data, dict):
        for key in ["training", "validation", "test", "data"]:
            if key in data and isinstance(data[key], list):
                items_to_check.extend(data[key])
        if not items_to_check:
            items_to_check = [data]

    missing_paths = []
    total_files_checked = 0

    for entry in items_to_check:
        if isinstance(entry, dict):
            for key, path in entry.items():
                # Check string entries that represent file paths
                if isinstance(path, str) and ("/" in path or "\\" in path):
                    total_files_checked += 1
                    if not os.path.exists(path):
                        missing_paths.append(path)
        elif isinstance(entry, str):
            total_files_checked += 1
            if not os.path.exists(entry):
                missing_paths.append(entry)

    print(f"=== {label} ({json_path}) ===")
    print(f"Total file paths checked : {total_files_checked}")
    print(f"Missing files found      : {len(missing_paths)}")
    print("-" * 40)

    return missing_paths

# Run checks on both manifests
train_missing = check_json_manifest("data/train.json", "Train Manifest")
val_missing = check_json_manifest("data/val.json", "Validation Manifest")

all_missing = sorted(list(set(train_missing + val_missing)))

if all_missing:
    print(f"\n[!] RESULT: {len(all_missing)} unique missing files detected.")
    
    # Save missing file list to disk
    log_file = "missing_files_log.txt"
    with open(log_file, "w") as f:
        for p in all_missing:
            f.write(f"{p}\n")
            
    print(f"A detailed list of missing paths has been saved to '{log_file}'.")
else:
    print("\n[✓] SUCCESS: All files listed in train.json and val.json exist on disk!")