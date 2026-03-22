"""
Create synthetic preference datasets from reward-bench-2 data based on different criteria.

This script creates:
1. Single-criteria datasets: preference based on one criterion (Factuality, Safety, etc.)
2. Mixed-criteria datasets: preference randomly from different criteria (without knowing which criterion)
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


def create_single_criteria_dataset(df: pd.DataFrame, criterion: str, split: str = "train") -> Dataset:
    """
    Create a preference dataset based on a single criterion.
    
    Args:
        df: DataFrame with preference data (must have 'criterion' column)
        criterion: The criterion name to filter (e.g., 'Factuality', 'Safety')
        split: Dataset split name
    
    Returns:
        Dataset with preference based on the specified criterion
    """
    # Filter data for this criterion
    criterion_df = df[df["criterion"] == criterion].copy()
    
    if len(criterion_df) == 0:
        print(f"Warning: No data found for criterion '{criterion}'")
        return Dataset.from_dict({"prompt": [], "y_w": [], "y_l": []})
    
    # Create dataset
    data = {
        "prompt": criterion_df["prompt"].tolist(),
        "y_w": [],
        "y_l": [],
    }
    
    for idx, row in criterion_df.iterrows():
        # chosen=0 means response_0 is preferred, chosen=1 means response_1 is preferred
        if row["chosen"] == 0:
            # response_0 is preferred
            data["y_w"].append(f"{ASSISTANT_TOKEN} {row['response_0']}")
            data["y_l"].append(f"{ASSISTANT_TOKEN} {row['response_1']}")
        else:
            # response_1 is preferred
            data["y_w"].append(f"{ASSISTANT_TOKEN} {row['response_1']}")
            data["y_l"].append(f"{ASSISTANT_TOKEN} {row['response_0']}")
    
    dataset = Dataset.from_dict(data)
    return dataset


def create_mixed_criteria_dataset(df: pd.DataFrame, split: str = "train", seed: int = 42) -> Dataset:
    """
    Create a preference dataset where each preference comes from a random criterion.
    The model doesn't know which criterion was used.
    
    Args:
        df: DataFrame with preference data (must have 'criterion' column)
        split: Dataset split name
        seed: Random seed
    
    Returns:
        Dataset with mixed criteria preferences
    """
    np.random.seed(seed)
    
    # Get all unique criteria
    criteria_list = df["criterion"].unique().tolist()
    
    # Count examples per criterion
    criterion_counts = df["criterion"].value_counts().to_dict()
    total_examples = len(df)
    
    # Calculate weights based on frequency (or use equal weights)
    # Using equal weights for now, but can be adjusted
    criterion_weights = {crit: 1.0 / len(criteria_list) for crit in criteria_list}
    
    # Create probability array matching criteria_list order
    p = [criterion_weights[crit] for crit in criteria_list]
    
    # Sample from each criterion proportionally
    sampled_rows = []
    for criterion in criteria_list:
        criterion_df = df[df["criterion"] == criterion]
        # Sample proportionally to maintain criterion distribution
        n_samples = int(len(criterion_df) * (total_examples / len(df)))
        if n_samples > len(criterion_df):
            n_samples = len(criterion_df)
        sampled = criterion_df.sample(n=n_samples, random_state=seed + hash(criterion) % 1000)
        sampled_rows.append(sampled)
    
    # Combine and shuffle
    mixed_df = pd.concat(sampled_rows, ignore_index=True)
    mixed_df = mixed_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Limit to original size if needed
    if len(mixed_df) > total_examples:
        mixed_df = mixed_df.head(total_examples)
    
    data = {
        "prompt": mixed_df["prompt"].tolist(),
        "y_w": [],
        "y_l": [],
        "criterion_used": [],  # For analysis only, not used in training
    }
    
    for idx, row in mixed_df.iterrows():
        # Determine preference based on chosen field
        if row["chosen"] == 0:
            # response_0 is preferred
            data["y_w"].append(f"{ASSISTANT_TOKEN} {row['response_0']}")
            data["y_l"].append(f"{ASSISTANT_TOKEN} {row['response_1']}")
        else:
            # response_1 is preferred
            data["y_w"].append(f"{ASSISTANT_TOKEN} {row['response_1']}")
            data["y_l"].append(f"{ASSISTANT_TOKEN} {row['response_0']}")
        
        data["criterion_used"].append(row["criterion"])
    
    dataset = Dataset.from_dict(data)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Create synthetic preference datasets from reward-bench-2")
    parser.add_argument("--data_dir", type=str, default="data_out", help="Directory containing CSV files")
    parser.add_argument("--output_dir", type=str, default="reward_bench2_synthetic_datasets", help="Output directory for datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--downsample_ratio", type=float, default=1.0, help="Downsample ratio for train/val/test (default: 1.0)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    train_df = pd.read_csv(os.path.join(args.data_dir, "reward_bench2_preference_train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "reward_bench2_preference_validation.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "reward_bench2_preference_test.csv"))
    
    print(f"Loaded data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Print criterion distribution
    print("\nCriterion distribution in train set:")
    train_criterion_counts = train_df["criterion"].value_counts()
    for criterion, count in train_criterion_counts.items():
        print(f"  {criterion}: {count} ({count/len(train_df)*100:.1f}%)")
    
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
    
    # Get all unique criteria
    all_criteria = sorted(train_df["criterion"].unique().tolist())
    print(f"\nFound {len(all_criteria)} criteria: {all_criteria}")
    
    # Create single-criteria datasets
    for criterion in all_criteria:
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
    print("\nCriterion distribution in mixed dataset (train):")
    criterion_counts = {}
    for crit in train_ds["criterion_used"]:
        criterion_counts[crit] = criterion_counts.get(crit, 0) + 1
    for crit, count in sorted(criterion_counts.items()):
        print(f"  {crit}: {count} ({count/len(train_ds)*100:.1f}%)")


if __name__ == "__main__":
    main()

