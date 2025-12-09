import os
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def load_imdb_preference(split: str = "train"):
    """
    Load imdb_preference dataset from Hugging Face.
    Splits: train, validation, test
    """
    ds = load_dataset("ZHZisZZ/imdb_preference", split=split)
    return ds


def build_sentiment_reward_model(
    model_name: str = "lvwerra/distilbert-imdb",
    device: str = "cuda",
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, int]:
    """
    Build pretrained sentiment classifier as reward model and move it to device.
    Returns:
        tokenizer
        model
        pos_label_idx: index of the positive class in model outputs
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Find which index corresponds to the positive class
    id2label = model.config.id2label
    pos_label_idx = None
    for idx, label in id2label.items():
        if "POS" in label.upper():
            pos_label_idx = int(idx)
            break
    if pos_label_idx is None:
        # Fall back to assuming index 1 is positive for binary classification
        pos_label_idx = 1

    return tokenizer, model, pos_label_idx


def compute_rewards_for_batch(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    pos_label_idx: int,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Compute two rewards:
      - sentiment_reward: probability of positive sentiment
      - conciseness_reward: negative token length
    """
    # Tokenize for model forward
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape: (batch_size, num_labels)
        probs = torch.softmax(logits, dim=-1)
        pos_probs = probs[:, pos_label_idx]  # shape: (batch_size,)

    sentiment_reward = pos_probs.cpu().numpy().astype(np.float32)

    # Binary label from reward model: 1 if positive prob >= 0.5
    sentiment_label = (pos_probs >= 0.5).long().cpu().numpy().astype(np.int64)

    # Conciseness reward based on token length (without padding)
    enc_no_pad = tokenizer(
        list(texts),
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )
    lengths = np.array([len(ids) for ids in enc_no_pad["input_ids"]], dtype=np.float32)
    conciseness_reward = -lengths

    return {
        "sentiment_reward": sentiment_reward,
        "sentiment_label_from_rm": sentiment_label,
        "token_length": lengths,
        "conciseness_reward": conciseness_reward,
    }


def build_imdb_preference_table(
    split: str = "train",
    max_examples: int = None,
    model_name: str = "lvwerra/distilbert-imdb",
    batch_size: int = 32,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Build table on imdb_preference with rewards for both responses.

    For each row in imdb_preference:
      prompt: str
      responses: list of 2 strings
      scores: list of 2 floats (from original dataset)
      chosen: int, index of preferred response

    We compute for each of the 2 responses:
      - rm_sentiment_reward_0/1
      - rm_sentiment_label_0/1
      - token_length_0/1
      - conciseness_reward_0/1
    """
    # Load data
    ds = load_imdb_preference(split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    # Build reward model
    tokenizer, model, pos_label_idx = build_sentiment_reward_model(
        model_name=model_name,
        device=device,
    )

    all_rows = []
    for start in tqdm(range(0, len(ds), batch_size), desc=f"Processing {split}"):
        end = min(start + batch_size, len(ds))
        batch = ds[start:end]

        prompts = batch["prompt"]               # list of str
        responses = batch["responses"]          # list of list[str], length 2
        scores = batch["scores"]                # list of list[float], length 2
        chosen_list = batch["chosen"]           # list[int]

        # Flatten responses for reward computation
        flat_texts = [r for pair in responses for r in pair]  # length = 2 * batch_size_current

        rewards = compute_rewards_for_batch(
            texts=flat_texts,
            tokenizer=tokenizer,
            model=model,
            pos_label_idx=pos_label_idx,
            device=device,
        )

        # reshape to (batch_size_current, 2)
        n = end - start
        sr = rewards["sentiment_reward"].reshape(n, 2)
        sl = rewards["sentiment_label_from_rm"].reshape(n, 2)
        tl = rewards["token_length"].reshape(n, 2)
        cr = rewards["conciseness_reward"].reshape(n, 2)

        for i in range(n):
            row = {
                "prompt": prompts[i],
                "response_0": responses[i][0],
                "response_1": responses[i][1],
                "score_0": float(scores[i][0]),
                "score_1": float(scores[i][1]),
                "chosen": int(chosen_list[i]),
                "rm_sentiment_reward_0": float(sr[i, 0]),
                "rm_sentiment_reward_1": float(sr[i, 1]),
                "rm_sentiment_label_0": int(sl[i, 0]),
                "rm_sentiment_label_1": int(sl[i, 1]),
                "token_length_0": int(tl[i, 0]),
                "token_length_1": int(tl[i, 1]),
                "conciseness_reward_0": float(cr[i, 0]),
                "conciseness_reward_1": float(cr[i, 1]),
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    return df


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs("data_out", exist_ok=True)

    for split in ["train", "validation", "test"]:
        df = build_imdb_preference_table(
            split=split,
            max_examples=None,
            model_name="lvwerra/distilbert-imdb",
            batch_size=128,
            device=device,
        )
        out_path = f"data_out/imdb_preference_{split}_with_two_preferences.csv"
        df.to_csv(out_path, index=False)
        print("saved to", out_path)
