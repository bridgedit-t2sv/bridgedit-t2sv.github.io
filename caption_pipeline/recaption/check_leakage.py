import json
import os
import itertools
from typing import Dict, Set, Optional, Any

def load_basename_set(filepath: str) -> Optional[Set[str]]:
    """
    Loads a JSON file, extracts all keys, and returns a set of all key basenames.

    Args:
        filepath: The path to the JSON file.

    Returns:
        A set of all basename strings, or None if loading fails.
    """
    print(f"  > Loading: {filepath}...")
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"  [Error] File not found: {filepath}. Skipping.")
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure the loaded data is a dictionary
        if not isinstance(data, dict):
            print(f"  [Error] {filepath} is not a JSON dictionary. Skipping.")
            return None
        
        # Extract basenames
        basenames = set(os.path.basename(key) for key in data.keys())
        
        print(f"  > Loaded successfully: {filepath} ({len(basenames)} unique files)")
        return basenames

    except json.JSONDecodeError:
        print(f"  [Error] Could not parse JSON: {filepath}. File might be corrupt. Skipping.")
        return None
    except Exception as e:
        print(f"  [Error] An unknown error occurred while loading {filepath}: {e}. Skipping.")
        return None

def generate_and_print_leakage_table(loaded_keys: Dict[str, Dict[str, Set[str]]]):
    """
    Generates and prints a matrix-style table comparing Test Sets (rows)
    against Train Sets (columns), showing the percentage of overlap.

    Args:
        loaded_keys: A nested dictionary: {base_name: {"train": {set}, "test": {set}}}
    """
    print("\n--- 2. Data Leakage Matrix (Testset vs. Trainset) ---")
    
    # Get sorted base names of datasets that have both train and test sets
    base_names = sorted(loaded_keys.keys())
    if not base_names:
        print("  No complete dataset pairs (train/test) were loaded. Cannot generate matrix.")
        return
    
    # Create the header row
    headers = ["Testset"] + [f"{name} (Train)" for name in base_names]
    
    # Prepare all rows of data
    table_data = [headers]
    for test_name in base_names:
        row = [f"{test_name} (Test)"]
        test_set = loaded_keys[test_name]["test"]
        test_size = len(test_set)
        
        for train_name in base_names:
            if test_name == train_name:
                # Diagonal is non-applicable, as in the user's image
                row.append(" - ")
            else:
                train_set = loaded_keys[train_name]["train"]
                
                if test_size == 0:
                    # Handle division by zero if test set is empty
                    row.append("0.0%")
                else:
                    overlap = len(test_set & train_set)
                    percentage = (overlap / test_size) * 100
                    row.append(f"{percentage:.1f}%")
        table_data.append(row)

    # --- Print the formatted table ---
    
    # Calculate column widths
    col_widths = [0] * len(headers)
    for row in table_data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Print Header
    header_line = ""
    for i, cell in enumerate(headers):
        header_line += cell.ljust(col_widths[i] + 2) # +2 for padding
    print(header_line)

    # Print separator
    separator_line = ""
    for width in col_widths:
        separator_line += "-" * width + "  "
    print(separator_line)

    # Print data rows
    for row in table_data[1:]:
        row_line = ""
        # First column (row header) is left-justified
        row_line += row[0].ljust(col_widths[0] + 2)
        
        # Data columns are right-justified for number alignment
        for i, cell in enumerate(row[1:], 1):
            row_line += cell.rjust(col_widths[i]) + "  "
        print(row_line)

def check_detailed_leakage(dataset_keys: Dict[str, Set[str]]):
    """
    Compares all loaded dataset keys and prints a detailed overlap report.
    This is the original function from the previous script.
    """
    
    # Get all unique pairs for comparison
    all_pairs = list(itertools.combinations(dataset_keys.keys(), 2))
    
    # For categorized reporting
    report = {
        "train_vs_test": [],
        "test_vs_test": [],
        "train_vs_train": []
    }

    print("\n--- 3. Detailed Pair-wise Overlap Report ---")
    
    for name1, name2 in all_pairs:
        keys1 = dataset_keys[name1]
        keys2 = dataset_keys[name2]
        
        # Calculate overlap
        overlap = keys1 & keys2
        overlap_count = len(overlap)
        
        # Prepare result line
        result_line = f"{name1:<18} <-> {name2:<18} : {overlap_count:>5} overlapping files"
        
        # Categorize the overlap based on naming conventions ('train' or 'test')
        is_name1_train = "train" in name1
        is_name2_train = "train" in name2
        is_name1_test = "test" in name1
        is_name2_test = "test" in name2

        category = None
        if (is_name1_train and is_name2_test) or (is_name1_test and is_name2_train):
            category = "train_vs_test"
        elif is_name1_test and is_name2_test:
            category = "test_vs_test"
        elif is_name1_train and is_name2_train:
            category = "train_vs_train"
        
        if category:
            if overlap_count > 0:
                report[category].append(f"  [!!] {result_line}")
            else:
                report[category].append(f"  [OK] {result_line}")

    # --- Print the final report ---
    print("\n### Check 1: Train vs. Test (High-Risk Leakage) ###")
    if report["train_vs_test"]:
        # Sort warnings (!!) to the top
        for line in sorted(report["train_vs_test"], reverse=True):
            print(line)
    else:
        print("  No configured [Train vs. Test] pairs found.")

    print("\n### Check 2: Test vs. Test (Evaluation Contamination) ###")
    if report["test_vs_test"]:
        for line in sorted(report["test_vs_test"], reverse=True):
            print(line)
    else:
        print("  No configured [Test vs. Test] pairs found (or only one test set).")

    print("\n### Check 3: Train vs. Train (Data Redundancy) ###")
    if report["train_vs_train"]:
        for line in sorted(report["train_vs_train"], reverse=True):
            print(line)
    else:
        print("  No configured [Train vs. Train] pairs found (or only one train set).")

def main():
    # --- 1. Configure Datasets ---
    # Define the datasets you want to compare.
    # For the table, provide both "train" and "test" paths.
    # If a file is missing, set its path to None.
    
    DATASETS_TO_COMPARE = {
        # "BaseName": {"train": "path/to/train.json", "test": "path/to/test.json"}
        
        "Avsync": {
            "train": "avsync_train-72B-captions.json",
            "test": "avsync-test-72B-captions.json"  # <-- !! UPDATE THIS PATH !! (e.g., "avsync_test.json")
        },
        "VGGSound-SS": {
            "train": "vgg-ss-train-72B-caption.json",
            "test": "vgg-ss-test-72B-caption.json"
        },
        "Landscape": {
            "train": "landscape-captions-train.json",
            "test": "landscape-captions-test.json"
        }
    }
    
    print("--- 1. Loading Dataset Keys ---")
    
    # loaded_keys is for the table
    loaded_keys: Dict[str, Dict[str, Set[str]]] = {}
    
    # flat_keys is for the detailed list report
    flat_keys: Dict[str, Set[str]] = {}

    for base_name, paths in DATASETS_TO_COMPARE.items():
        train_path = paths.get("train")
        test_path = paths.get("test")
        
        train_set = load_basename_set(train_path) if train_path else None
        test_set = load_basename_set(test_path) if test_path else None
        
        # Add to flat_keys for the list report
        if train_set is not None:
            flat_keys[f"{base_name}_train"] = train_set
        if test_set is not None:
            flat_keys[f"{base_name}_test"] = test_set
            
        # Add to loaded_keys for the table (only if both are present)
        if train_set is not None and test_set is not None:
            loaded_keys[base_name] = {"train": train_set, "test": test_set}
        else:
            print(f"  [Info] '{base_name}' will be skipped in the matrix table "
                  "because its train or test set is missing or failed to load.")

    if not flat_keys:
        print("\nError: No datasets were successfully loaded. Exiting.")
        return

    # --- 2. Generate Leakage Matrix Table ---
    generate_and_print_leakage_table(loaded_keys)

    # --- 3. Generate Detailed List Report ---
    check_detailed_leakage(flat_keys)

if __name__ == "__main__":
    main()
