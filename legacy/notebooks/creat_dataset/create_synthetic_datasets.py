"""
Create synthetic preference datasets based on different criteria.

This script creates:
1. Single-criteria datasets: preference based on one criterion (sentiment, conciseness, lexical_richness, semantic_intensity)
2. Mixed-criteria datasets: preference randomly from different criteria (without knowing which criterion)
"""

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import os
import argparse
from typing import List, Dict

CRITERIA = {
    "sentiment": "rm_sentiment_reward",
    "conciseness": "conciseness_reward", 
    "lexical_richness": "lexical_richness",
    "semantic_intensity": "semantic_intensity"
}

PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'
EOS_TOKEN = '<|endoftext|>'


def create_single_criteria_dataset(df: pd.DataFrame, criterion: str, split: str = "train") -> Dataset:
    """
    Create a preference dataset based on a single criterion.
    
    Args:
        df: DataFrame with preference data
        criterion: One of 'sentiment', 'conciseness', 'lexical_richness', 'semantic_intensity'
        split: Dataset split name
    
    Returns:
        Dataset with preference based on the specified criterion
    """
    criterion_col = CRITERIA[criterion]
    
    # Determine preference based on criterion
    # For all criteria: higher value is better
    # (conciseness_reward is negative of length, so higher = shorter = better)
    preference = (df[f"{criterion_col}_0"] < df[f"{criterion_col}_1"]).astype(int)
    
    # Create dataset
    data = {
        "prompt": df["prompt"].tolist(),
        "y_w": [],
        "y_l": [],
    }
    
    for idx, pref in enumerate(preference):
        if pref == 0:
            # response_0 is preferred
            data["y_w"].append(f"{ASSISTANT_TOKEN} {df.iloc[idx]['response_0']}")
            data["y_l"].append(f"{ASSISTANT_TOKEN} {df.iloc[idx]['response_1']}")
        else:
            # response_1 is preferred
            data["y_w"].append(f"{ASSISTANT_TOKEN} {df.iloc[idx]['response_1']}")
            data["y_l"].append(f"{ASSISTANT_TOKEN} {df.iloc[idx]['response_0']}")
    
    dataset = Dataset.from_dict(data)
    return dataset


def create_mixed_criteria_dataset(df: pd.DataFrame, split: str = "train", seed: int = 42) -> Dataset:
    """
    Create a preference dataset where each preference is based on a random criterion.
    The model doesn't know which criterion was used.
    
    Args:
        df: DataFrame with preference data
        split: Dataset split name
        seed: Random seed
    
    Returns:
        Dataset with mixed criteria preferences
    """
    np.random.seed(seed)
    
    data = {
        "prompt": df["prompt"].tolist(),
        "y_w": [],
        "y_l": [],
        "criterion_used": [],  # For analysis only, not used in training
    }
    
    criteria_list = list(CRITERIA.keys())
    
    # Weighted sampling: reduce sampling for conciseness and lexical_richness
    # to lower overall accuracy to ~50%
    # Based on accuracies: conciseness=0.9062, lexical_richness=0.8676,
    # semantic_intensity=0.4167, sentiment=0.5172
    criterion_weights = {
        "conciseness": 0.05,           # 5% (was 25%)
        "lexical_richness": 0.05,      # 5% (was 25%)
        "semantic_intensity": 0.55,    # 55% (was 25%)
        "sentiment": 0.35              # 35% (was 25%)
    }
    # Create probability array matching criteria_list order
    p = [criterion_weights[crit] for crit in criteria_list]
    
    for idx in range(len(df)):
        # Weighted random selection of criterion for this example
        criterion = np.random.choice(criteria_list, p=p)
        criterion_col = CRITERIA[criterion]
        
        # Determine preference based on selected criterion
        if df.iloc[idx][f"{criterion_col}_0"] < df.iloc[idx][f"{criterion_col}_1"]:
            # response_1 is preferred
            data["y_w"].append(f"{ASSISTANT_TOKEN} {df.iloc[idx]['response_1']}")
            data["y_l"].append(f"{ASSISTANT_TOKEN} {df.iloc[idx]['response_0']}")
        else:
            # response_0 is preferred
            data["y_w"].append(f"{ASSISTANT_TOKEN} {df.iloc[idx]['response_0']}")
            data["y_l"].append(f"{ASSISTANT_TOKEN} {df.iloc[idx]['response_1']}")
        
        data["criterion_used"].append(criterion)
    
    dataset = Dataset.from_dict(data)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Create synthetic preference datasets")
    parser.add_argument("--data_dir", type=str, default="data_out", help="Directory containing CSV files")
    parser.add_argument("--output_dir", type=str, default="synthetic_datasets", help="Output directory for datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--downsample_ratio", type=float, default=0.25, help="Downsample ratio for train/val/test (default: 0.25)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    train_df = pd.read_csv(os.path.join(args.data_dir, "imdb_preference_train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "imdb_preference_validation.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "imdb_preference_test.csv"))
    
    print(f"Loaded data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
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
    for criterion in CRITERIA.keys():
        print(f"\nCreating single-criteria dataset for: {criterion}")
        
        train_ds = create_single_criteria_dataset(train_df, criterion, "train")
        val_ds = create_single_criteria_dataset(val_df, criterion, "validation")
        test_ds = create_single_criteria_dataset(test_df, criterion, "test")
        
        dataset_dict = DatasetDict({
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds
        })
        
        output_path = os.path.join(args.output_dir, f"single_{criterion}")
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

