"""
Create synthetic preference datasets from mix-instruct data based on different criteria.

This script creates:
1. Single-criteria datasets: preference based on one criterion (rougeL, bleu, bertscore, bleurt, bartscore)
2. Mixed-criteria datasets: preference randomly from different criteria (without knowing which criterion)

The input CSV format should have columns:
- prompt, response_0, response_1
- rougeL_0, rougeL_1, bleu_0, bleu_1, bertscore_0, bertscore_1, bleurt_0, bleurt_1, bartscore_0, bartscore_1

For each criterion, preference is determined by comparing metric_0 vs metric_1.
Higher score is better (including for negative-valued metrics like bartscore/bleurt).
"""

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import os
import argparse
from typing import List, Dict

PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'
EOS_TOKEN = '<|endoftext|>'

METRICS = ["rougeL", "bleu", "bertscore", "bleurt", "bartscore"]


def create_single_criteria_dataset(df: pd.DataFrame, criterion: str, split: str = "train") -> Dataset:
    """
    Create a preference dataset based on a single criterion.
    
    Args:
        df: DataFrame with preference data (must have '{criterion}_0' and '{criterion}_1' columns)
        criterion: The criterion name (e.g., 'rougeL', 'bleu', 'bartscore')
        split: Dataset split name
    
    Returns:
        Dataset with preference based on the specified criterion
    """
    score_0_col = f"{criterion}_0"
    score_1_col = f"{criterion}_1"
    
    # Check if required columns exist
    if score_0_col not in df.columns or score_1_col not in df.columns:
        print(f"Warning: Columns '{score_0_col}' or '{score_1_col}' not found in data")
        return Dataset.from_dict({"prompt": [], "y_w": [], "y_l": []})
    
    # Create dataset
    data = {
        "prompt": [],
        "y_w": [],
        "y_l": [],
    }
    
    for idx, row in df.iterrows():
        score_0 = row[score_0_col]
        score_1 = row[score_1_col]
        
        # Skip if scores are NaN
        if pd.isna(score_0) or pd.isna(score_1):
            continue
        
        # Higher score is better (including for negative-valued metrics)
        if score_0 > score_1:
            # response_0 is preferred
            data["prompt"].append(row["prompt"])
            data["y_w"].append(f"{ASSISTANT_TOKEN} {row['response_0']}")
            data["y_l"].append(f"{ASSISTANT_TOKEN} {row['response_1']}")
        elif score_1 > score_0:
            # response_1 is preferred
            data["prompt"].append(row["prompt"])
            data["y_w"].append(f"{ASSISTANT_TOKEN} {row['response_1']}")
            data["y_l"].append(f"{ASSISTANT_TOKEN} {row['response_0']}")
        # If scores are equal, skip this example
    
    if len(data["prompt"]) == 0:
        print(f"Warning: No valid preference pairs found for criterion '{criterion}'")
        return Dataset.from_dict({"prompt": [], "y_w": [], "y_l": []})
    
    dataset = Dataset.from_dict(data)
    return dataset


def create_mixed_criteria_dataset(df: pd.DataFrame, split: str = "train", seed: int = 42) -> Dataset:
    """
    Create a preference dataset where each preference comes from a random criterion.
    The model doesn't know which criterion was used.
    
    Args:
        df: DataFrame with preference data (must have metric score columns)
        split: Dataset split name
        seed: Random seed
    
    Returns:
        Dataset with mixed criteria preferences
    """
    np.random.seed(seed)
    
    # Create dataset
    data = {
        "prompt": [],
        "y_w": [],
        "y_l": [],
        "criterion_used": [],  # For analysis only, not used in training
    }
    
    for idx, row in df.iterrows():
        # Randomly select a criterion
        criterion = np.random.choice(METRICS)
        score_0_col = f"{criterion}_0"
        score_1_col = f"{criterion}_1"
        
        # Check if columns exist
        if score_0_col not in df.columns or score_1_col not in df.columns:
            continue
        
        score_0 = row[score_0_col]
        score_1 = row[score_1_col]
        
        # Skip if scores are NaN
        if pd.isna(score_0) or pd.isna(score_1):
            continue
        
        # Higher score is better
        if score_0 > score_1:
            # response_0 is preferred
            data["prompt"].append(row["prompt"])
            data["y_w"].append(f"{ASSISTANT_TOKEN} {row['response_0']}")
            data["y_l"].append(f"{ASSISTANT_TOKEN} {row['response_1']}")
            data["criterion_used"].append(criterion)
        elif score_1 > score_0:
            # response_1 is preferred
            data["prompt"].append(row["prompt"])
            data["y_w"].append(f"{ASSISTANT_TOKEN} {row['response_1']}")
            data["y_l"].append(f"{ASSISTANT_TOKEN} {row['response_0']}")
            data["criterion_used"].append(criterion)
        # If scores are equal, skip this example
    
    dataset = Dataset.from_dict(data)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Create synthetic preference datasets from mix-instruct")
    parser.add_argument("--data_dir", type=str, default="data_out", help="Directory containing CSV files")
    parser.add_argument("--output_dir", type=str, default="mixinstruct_synthetic_datasets", help="Output directory for datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--downsample_ratio", type=float, default=1.0, help="Downsample ratio for train/val/test (default: 1.0)")
    parser.add_argument("--ds_suffix", type=str, default="ds01", help="Downsample suffix in filename (e.g., 'ds01' for _ds01.csv files)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Construct file paths
    train_path = os.path.join(args.data_dir, f"mix_instruct_pref_train_{args.ds_suffix}.csv")
    val_path = os.path.join(args.data_dir, f"mix_instruct_pref_validation_{args.ds_suffix}.csv")
    test_path = os.path.join(args.data_dir, f"mix_instruct_pref_test_{args.ds_suffix}.csv")
    
    # Check if files exist, fallback to files without suffix
    if not os.path.exists(train_path):
        train_path = os.path.join(args.data_dir, "mix_instruct_pref_train.csv")
    if not os.path.exists(val_path):
        val_path = os.path.join(args.data_dir, "mix_instruct_pref_validation.csv")
    if not os.path.exists(test_path):
        test_path = os.path.join(args.data_dir, "mix_instruct_pref_test.csv")
    
    # Load data
    print(f"Loading data from:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Test: {test_path}")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"\nLoaded data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Check available metrics
    available_metrics = []
    for metric in METRICS:
        if f"{metric}_0" in train_df.columns and f"{metric}_1" in train_df.columns:
            available_metrics.append(metric)
    
    print(f"\nAvailable metrics: {available_metrics}")
    
    # Downsample if specified
    if args.downsample_ratio < 1.0:
        np.random.seed(args.seed)
        train_size = int(len(train_df) * args.downsample_ratio)
        val_size = int(len(val_df) * args.downsample_ratio)
        test_size = int(len(test_df) * args.downsample_ratio)
        
        train_df = train_df.sample(n=train_size, random_state=args.seed).reset_index(drop=True)
        val_df = val_df.sample(n=val_size, random_state=args.seed + 1).reset_index(drop=True)
        test_df = test_df.sample(n=test_size, random_state=args.seed + 2).reset_index(drop=True)
        
        print(f"After downsample ({args.downsample_ratio}): train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Create single-criteria datasets
    for criterion in available_metrics:
        print(f"\nCreating single-criteria dataset for: {criterion}")
        
        train_ds = create_single_criteria_dataset(train_df, criterion, "train")
        val_ds = create_single_criteria_dataset(val_df, criterion, "validation")
        test_ds = create_single_criteria_dataset(test_df, criterion, "test")
        
        if len(train_ds) == 0:
            print(f"  Skipping {criterion} - no data in train set")
            continue
        
        dataset_dict = DatasetDict({
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds
        })
        
        # Sanitize criterion name for filesystem
        criterion_safe = criterion.replace(" ", "_").replace("/", "_")
        output_path = os.path.join(args.output_dir, f"single_{criterion_safe}")
        dataset_dict.save_to_disk(output_path)
        print(f"Saved to {output_path}")
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Create mixed-criteria dataset
    print(f"\nCreating mixed-criteria dataset")
    train_ds = create_mixed_criteria_dataset(train_df, "train", args.seed)
    val_ds = create_mixed_criteria_dataset(val_df, "validation", args.seed + 1)
    test_ds = create_mixed_criteria_dataset(test_df, "test", args.seed + 2)
    
    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })
    
    output_path = os.path.join(args.output_dir, "mixed_criteria")
    dataset_dict.save_to_disk(output_path)
    print(f"Saved to {output_path}")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Print criterion distribution for mixed dataset
    if len(train_ds) > 0:
        print("\nCriterion distribution in mixed dataset (train):")
        criterion_counts = {}
        for crit in train_ds["criterion_used"]:
            criterion_counts[crit] = criterion_counts.get(crit, 0) + 1
        for crit, count in sorted(criterion_counts.items()):
            print(f"  {crit}: {count} ({count/len(train_ds)*100:.1f}%)")


if __name__ == "__main__":
    main()
