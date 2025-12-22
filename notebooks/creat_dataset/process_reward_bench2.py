"""
Process reward-bench-2 dataset for DPO training.

This script:
1. Loads reward-bench-2 dataset (only has test split)
2. Extracts data with different criteria (subset field)
3. Splits into train/validation/test with 8:1:1 ratio
4. Saves as CSV files similar to ultrafeedback format
"""

import os
import random
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def load_reward_bench2(split: str = "test"):
    """Load reward-bench-2 dataset from HuggingFace."""
    ds = load_dataset("allenai/reward-bench-2", split=split)
    return ds


def process_reward_bench2_data(
    split: str = "test",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Process reward-bench-2 dataset into preference format.
    
    Each row contains:
      - prompt: instruction
      - response_0, response_1: chosen and rejected responses
      - chosen: 0 or 1 (0 means response_0 is chosen, 1 means response_1 is chosen)
      - criterion: the subset/criterion name (e.g., "Factuality", "Safety", etc.)
    """
    random.seed(seed)
    np.random.seed(seed)

    ds = load_reward_bench2(split=split)
    
    all_rows: List[Dict[str, Any]] = []

    for idx in tqdm(range(len(ds)), desc=f"Processing reward-bench-2 ({split})"):
        ex = ds[idx]
        
        prompt = ex.get("prompt", "")
        chosen_list = ex.get("chosen", [])
        rejected_list = ex.get("rejected", [])
        subset = ex.get("subset", "unknown")
        
        # Skip if missing required fields
        if not prompt or not chosen_list or not rejected_list:
            continue
        
        # Get first chosen and rejected response
        # chosen and rejected are lists, take the first element
        chosen_text = chosen_list[0] if isinstance(chosen_list, list) and len(chosen_list) > 0 else str(chosen_list)
        rejected_text = rejected_list[0] if isinstance(rejected_list, list) and len(rejected_list) > 0 else str(rejected_list)
        
        # Skip if responses are empty
        if not chosen_text or not rejected_text:
            continue
        
        # Create preference pair
        # response_0 is chosen, response_1 is rejected
        row: Dict[str, Any] = {
            "prompt": prompt,
            "response_0": chosen_text,
            "response_1": rejected_text,
            "chosen": 0,  # response_0 is the chosen one
            "criterion": subset,
        }
        
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    return df


def split_data_8_1_1(
    df: pd.DataFrame,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test with 8:1:1 ratio.
    """
    np.random.seed(seed)
    shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(shuffled_df)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_df = shuffled_df[:n_train].reset_index(drop=True)
    val_df = shuffled_df[n_train:n_train + n_val].reset_index(drop=True)
    test_df = shuffled_df[n_train + n_val:].reset_index(drop=True)

    return train_df, val_df, test_df


if __name__ == "__main__":
    os.makedirs("data_out", exist_ok=True)

    print("Loading reward-bench-2 dataset...")
    df_all = process_reward_bench2_data(
        split="test",
        seed=42,
    )

    print(f"\nTotal examples: {len(df_all)}")
    
    # Print criterion distribution
    print("\nCriterion distribution:")
    criterion_counts = df_all["criterion"].value_counts()
    for criterion, count in criterion_counts.items():
        print(f"  {criterion}: {count} ({count/len(df_all)*100:.1f}%)")
    
    print("\nSplitting into train:validation:test = 8:1:1...")
    train_df, val_df, test_df = split_data_8_1_1(df_all, seed=42)

    print(f"Train: {len(train_df)} ({len(train_df) / len(df_all) * 100:.1f}%)")
    print(f"Validation: {len(val_df)} ({len(val_df) / len(df_all) * 100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df) / len(df_all) * 100:.1f}%)")

    train_path = "data_out/reward_bench2_preference_train.csv"
    val_path = "data_out/reward_bench2_preference_validation.csv"
    test_path = "data_out/reward_bench2_preference_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSaved:")
    print(f"  Train: {train_path} ({len(train_df)} rows)")
    print(f"  Validation: {val_path} ({len(val_df)} rows)")
    print(f"  Test: {test_path} ({len(test_df)} rows)")

