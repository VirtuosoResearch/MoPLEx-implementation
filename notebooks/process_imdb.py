import os
from typing import Dict, Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from tqdm import tqdm

def load_imdb(split: str = "train"):
    """
    Load IMDB dataset.
    """
    ds = load_dataset("imdb", split=split)
    return ds


def build_sentiment_reward_model(model_name: str = "lvwerra/distilbert-imdb"):
    """
    Build pretrained sentiment classifier as reward model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        truncation=True,
        padding=True,
        device=-1,
    )
    return pipe, tokenizer


def compute_rewards_for_batch(
    texts,
    sentiment_pipe: TextClassificationPipeline,
    tokenizer,
) -> Dict[str, Any]:
    """
    Compute two rewards:
      - sentiment_reward: probability of positive sentiment
      - conciseness_reward: negative token length
    """
    # Sentiment reward
    outputs = sentiment_pipe(texts)

    sentiment_reward = []
    sentiment_label = []
    for out in outputs:
        # out is a list of dicts: [{"label": "POSITIVE", "score": ...}, ...]
        if out[0]["label"].upper().startswith("POS"):
            pos_score = out[0]["score"]
            neg_score = out[1]["score"]
        else:
            pos_score = out[1]["score"]
            neg_score = out[0]["score"]

        sentiment_reward.append(pos_score)
        sentiment_label.append(1 if pos_score >= neg_score else 0)

    sentiment_reward = np.array(sentiment_reward, dtype=np.float32)

    # Conciseness reward
    enc = tokenizer(
        texts,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )
    lengths = np.array([len(ids) for ids in enc["input_ids"]], dtype=np.float32)
    conciseness_reward = -lengths

    return {
        "sentiment_reward": sentiment_reward,
        "sentiment_label_from_rm": np.array(sentiment_label, dtype=np.int64),
        "token_length": lengths,
        "conciseness_reward": conciseness_reward,
    }


def build_imdb_preference_table(
    split: str = "train",
    max_examples: int = 5000,
    model_name: str = "lvwerra/distilbert-imdb",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Build IMDB table with sentiment_reward and conciseness_reward.
    """
    ds = load_imdb(split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    sentiment_pipe, tokenizer = build_sentiment_reward_model(model_name)

    all_rows = []
    for start in tqdm(range(0, len(ds), batch_size)):
        end = min(start + batch_size, len(ds))
        batch = ds[start:end]
        texts = batch["text"]

        rewards = compute_rewards_for_batch(
            texts=texts,
            sentiment_pipe=sentiment_pipe,
            tokenizer=tokenizer,
        )

        for i in range(end - start):
            row = {
                "text": texts[i],
                "imdb_label": int(batch["label"][i]),
                "sentiment_reward": float(rewards["sentiment_reward"][i]),
                "sentiment_label_from_rm": int(rewards["sentiment_label_from_rm"][i]),
                "token_length": int(rewards["token_length"][i]),
                "conciseness_reward": float(rewards["conciseness_reward"][i]),
            }
            all_rows.append(row)

        # print(f"processed {end}/{len(ds)}")

    df = pd.DataFrame(all_rows)
    return df


if __name__ == "__main__":
    df = build_imdb_preference_table(
        split="train",
        max_examples=25000,
        model_name="lvwerra/distilbert-imdb",
        batch_size=32,
    )

    os.makedirs("data_out", exist_ok=True)
    out_path = "data_out/imdb_with_two_preferences.csv"
    df.to_csv(out_path, index=False)
    print("saved to", out_path)
