import os
import random
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


ASPECTS = ["helpfulness", "truthfulness", "instruction_following", "honesty"]


def load_ultrafeedback(split: str = "train"):
    ds = load_dataset("openbmb/UltraFeedback", split=split)
    return ds


def extract_aspect_scores(
    completion: Dict[str, Any],
    aspects: List[str] = ASPECTS,
) -> Optional[Dict[str, float]]:
    """
    Extract four criteria ratings from a single completion, convert to float.
    Returns None if missing or 'N/A'.
    """
    ann = completion.get("annotations", {})
    scores: Dict[str, float] = {}

    for aspect in aspects:
        crit = ann.get(aspect, None)
        if crit is None:
            return None
        rating_str = crit.get("Rating", "N/A")
        if rating_str == "N/A":
            return None
        try:
            rating = float(rating_str)
        except ValueError:
            return None
        scores[aspect] = rating

    return scores


def build_ultrafeedback_preference_table(
    split: str = "train",
    max_examples: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build preference table from UltraFeedback using only the four original criteria ratings.

    Each row contains:
      - prompt: instruction
      - response_0, response_1: two randomly selected responses
      - score_0, score_1: average of four criteria ratings
      - chosen: 0 or 1 (which has higher average score); skip if tied
      - rm_helpfulness_reward_0/1 etc.: ratings for each of the four criteria
    """
    random.seed(seed)
    np.random.seed(seed)

    ds = load_ultrafeedback(split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    all_rows: List[Dict[str, Any]] = []

    for idx in tqdm(range(len(ds)), desc=f"Processing UltraFeedback ({split})"):
        ex = ds[idx]
        instruction = ex.get("instruction", "")
        completions = ex.get("completions", [])

        if not completions or len(completions) < 2:
            continue

        # Collect all completions with complete four-dimensional ratings
        eligible: List[Tuple[str, Dict[str, float]]] = []
        for comp in completions:
            scores = extract_aspect_scores(comp, ASPECTS)
            if scores is None:
                continue
            response_text = comp.get("response", "")
            eligible.append((response_text, scores))

        if len(eligible) < 2:
            continue

        # Randomly sample two different completions
        idx0, idx1 = np.random.choice(len(eligible), size=2, replace=False)
        response_0, scores_0 = eligible[idx0]
        response_1, scores_1 = eligible[idx1]

        # Overall score = average of four criteria ratings
        overall_0 = float(np.mean([scores_0[a] for a in ASPECTS]))
        overall_1 = float(np.mean([scores_1[a] for a in ASPECTS]))

        # Determine chosen (skip if tied)
        if overall_0 > overall_1:
            chosen = 0
        elif overall_1 > overall_0:
            chosen = 1
        else:
            continue

        row: Dict[str, Any] = {
            "prompt": instruction,
            "response_0": response_0,
            "response_1": response_1,
            "score_0": overall_0,
            "score_1": overall_1,
            "chosen": chosen,
        }

        # Four criteria ratings, used as rewards
        for aspect in ASPECTS:
            row[f"rm_{aspect}_reward_0"] = float(scores_0[aspect])
            row[f"rm_{aspect}_reward_1"] = float(scores_1[aspect])

        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    return df


def split_data_6_2_2(
    df: pd.DataFrame,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test with 6:2:2 ratio.
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

    print("Loading UltraFeedback dataset...")
    df_all = build_ultrafeedback_preference_table(
        split="train",
        max_examples=None,
        seed=42,
    )

    print(f"\nTotal examples: {len(df_all)}")
    
    # Downsample to 20% of the data
    print("\nDownsampling to 20% of the data...")
    df_all = df_all.sample(frac=0.2, random_state=42).reset_index(drop=True)
    print(f"After downsampling: {len(df_all)} examples")
    
    print("\nSplitting into train:validation:test = 6:2:2...")
    train_df, val_df, test_df = split_data_6_2_2(df_all, seed=42)

    print(f"Train: {len(train_df)} ({len(train_df) / len(df_all) * 100:.1f}%)")
    print(f"Validation: {len(val_df)} ({len(val_df) / len(df_all) * 100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df) / len(df_all) * 100:.1f}%)")

    train_path = "data_out/ultrafeedback_preference_train.csv"
    val_path = "data_out/ultrafeedback_preference_validation.csv"
    test_path = "data_out/ultrafeedback_preference_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSaved:")
    print(f"  Train: {train_path} ({len(train_df)} rows)")
    print(f"  Validation: {val_path} ({len(val_df)} rows)")
    print(f"  Test: {test_path} ({len(test_df)} rows)")
