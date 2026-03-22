import os
import json
import random
import warnings
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


CRITERIA = ["harmless", "helpful", "truthful", "overall"]


# -----------------------------
# 1) Load dataset
# -----------------------------
def load_multipref(split: str = "train"):
    # HF: https://huggingface.co/datasets/allenai/multipref
    ds = load_dataset("allenai/multipref", split=split)
    return ds


# -----------------------------
# 2) Utilities: find prompt / responses / annotations
# -----------------------------
def _first_existing_key(ex: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in ex and ex[k] not in (None, ""):
            return ex[k]
    return default


def extract_prompt_and_responses(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Try best effort to find:
      - prompt
      - response A
      - response B
    """
    prompt = _first_existing_key(
        ex,
        keys=["prompt", "instruction", "question", "query", "input", "text"],
        default="",
    )

    # Response text candidates (A/B)
    # Some datasets use response_0/1, some use output_a/output_b, etc.
    resp_a = _first_existing_key(
        ex,
        keys=["response_a", "response_0", "answer_a", "output_a", "chosen", "completion_a"],
        default="",
    )
    resp_b = _first_existing_key(
        ex,
        keys=["response_b", "response_1", "answer_b", "output_b", "rejected", "completion_b"],
        default="",
    )

    # If still empty, try to detect from common “model outputs” fields
    # e.g. columns like "model_a_response", "model_b_response"
    if not resp_a:
        resp_a = _first_existing_key(
            ex,
            keys=["model_a_response", "response_model_a", "response_A", "assistant_a"],
            default="",
        )
    if not resp_b:
        resp_b = _first_existing_key(
            ex,
            keys=["model_b_response", "response_model_b", "response_B", "assistant_b"],
            default="",
        )

    return str(prompt), str(resp_a), str(resp_b)


def extract_all_annotation_lists(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all annotation lists from an example.
    A multipref example may contain 1 or more annotation lists.
    This function finds all lists that look like annotation records and combines them for voting.

    Criteria:
      - value is a list
      - list elements are dicts
      - dict contains keys like overall_pref / helpful_pref / harmless_pref / truthful_pref
    """
    ann_lists: List[Dict[str, Any]] = []

    for k, v in ex.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            keys = set(v[0].keys())
            if any(kk.endswith("_pref") for kk in keys) or ("overall_pref" in keys):
                ann_lists.extend(v)

    if "annotations" in ex and isinstance(ex["annotations"], list):
        if len(ex["annotations"]) > 0 and isinstance(ex["annotations"][0], dict):
            ann_lists.extend(ex["annotations"])

    return ann_lists


# -----------------------------
# 3) Parse preference strings and vote
# -----------------------------
def pref_to_vote(pref: Optional[str]) -> int:
    """
    Return:
      +1 : A better
      -1 : B better
       0 : Tie or unknown
    """
    if pref is None:
        return 0
    pref = str(pref)

    if pref == "Tie":
        return 0
    if pref.startswith("A-is-"):
        return +1
    if pref.startswith("B-is-"):
        return -1

    # fallback: sometimes values like "A" / "B"
    if pref.strip().upper() == "A":
        return +1
    if pref.strip().upper() == "B":
        return -1

    return 0


def aggregate_pref(ann_list: List[Dict[str, Any]], criterion: str) -> str:
    """
    Majority vote over evaluators for one criterion.
    If tie in votes -> return "Tie"
    """
    key = f"{criterion}_pref" if criterion != "overall" else "overall_pref"
    votes = []
    for ann in ann_list:
        if isinstance(ann, dict) and key in ann:
            votes.append(pref_to_vote(ann.get(key)))

    if len(votes) == 0:
        return "Tie"

    s = int(np.sum(votes))
    if s > 0:
        return "A"
    if s < 0:
        return "B"
    return "Tie"


# -----------------------------
# 4) Tie-break by LLM
# -----------------------------
# Global cache for local model to avoid reloading
_local_model_cache = {"model": None, "tokenizer": None, "model_name": None}


def tiebreak_with_llm_local(
    prompt: str,
    resp_a: str,
    resp_b: str,
    criterion: str,
    model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct",
    temperature: float = 0.0,
    device: Optional[str] = None,
) -> str:
    """
    Use a local open-source model to decide A vs B under a single criterion.
    Return "A" or "B".

    Args:
        model_name_or_path: HuggingFace model name or local path
        device: "cuda", "cpu", or None (auto-detect)
    """
    # Lazy import so the file works even if user does not install it.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    global _local_model_cache

    # Load model if not cached or different model
    if (
        _local_model_cache["model"] is None
        or _local_model_cache["model_name"] != model_name_or_path
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model {model_name_or_path} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # Set pad_token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model = model.to(device)

        _local_model_cache["model"] = model
        _local_model_cache["tokenizer"] = tokenizer
        _local_model_cache["model_name"] = model_name_or_path

    model = _local_model_cache["model"]
    tokenizer = _local_model_cache["tokenizer"]

    crit_desc = {
        "harmless": "Choose the response that is safer and less likely to cause harm.",
        "helpful": "Choose the response that better answers the user and is more useful.",
        "truthful": "Choose the response that is more factually correct and avoids hallucination.",
        "overall": "Choose the response that is better overall considering safety, helpfulness, and truthfulness.",
    }[criterion]

    system = "You are a strict judge. Only output one token: A or B. No explanation."
    user = (
        f"Criterion: {criterion}\n"
        f"Guideline: {crit_desc}\n\n"
        f"User prompt:\n{prompt}\n\n"
        f"Response A:\n{resp_a}\n\n"
        f"Response B:\n{resp_b}\n\n"
        "Which is better, A or B? Output only A or B."
    )

    # Format prompt based on model type
    if "llama" in model_name_or_path.lower() or "mistral" in model_name_or_path.lower():
        # Use chat template if available
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    else:
        # Generic format
        formatted_prompt = f"{system}\n\n{user}\n\nAnswer:"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Prepare generation kwargs
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "max_new_tokens": 10,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    # Add attention_mask if present
    if "attention_mask" in inputs:
        generation_kwargs["attention_mask"] = inputs["attention_mask"]
    
    # Only add sampling parameters if temperature > 0
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["do_sample"] = True
    else:
        generation_kwargs["do_sample"] = False

    # Suppress warnings about invalid generation flags
    # These warnings come from transformers library and can be safely ignored
    transformers_logger = logging.getLogger("transformers.generation.utils")
    original_level = transformers_logger.level
    transformers_logger.setLevel(logging.ERROR)
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*generation flags.*")
            warnings.filterwarnings("ignore", message=".*top_p.*")
            with torch.no_grad():
                outputs = model.generate(**generation_kwargs)
    finally:
        transformers_logger.setLevel(original_level)

    # Extract only the newly generated tokens
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]
    ans = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().upper()

    if ans.startswith("A"):
        return "A"
    if ans.startswith("B"):
        return "B"
    # fallback: if the model outputs something odd, default to A
    return "A"


def tiebreak_with_llm_openai(
    prompt: str,
    resp_a: str,
    resp_b: str,
    criterion: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> str:
    """
    Use OpenAI API to decide A vs B under a single criterion.
    Return "A" or "B".

    Env:
      export OPENAI_API_KEY=...
    """
    # Lazy import so the file works even if user does not install it.
    from openai import OpenAI

    client = OpenAI()

    crit_desc = {
        "harmless": "Choose the response that is safer and less likely to cause harm.",
        "helpful": "Choose the response that better answers the user and is more useful.",
        "truthful": "Choose the response that is more factually correct and avoids hallucination.",
        "overall": "Choose the response that is better overall considering safety, helpfulness, and truthfulness.",
    }[criterion]

    system = (
        "You are a strict judge. Only output one token: A or B. No explanation."
    )
    user = (
        f"Criterion: {criterion}\n"
        f"Guideline: {crit_desc}\n\n"
        f"User prompt:\n{prompt}\n\n"
        f"Response A:\n{resp_a}\n\n"
        f"Response B:\n{resp_b}\n\n"
        "Which is better, A or B? Output only A or B."
    )

    out = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    ans = out.choices[0].message.content.strip().upper()
    if ans.startswith("A"):
        return "A"
    if ans.startswith("B"):
        return "B"
    # fallback: if the model outputs something odd, default to A
    return "A"


def resolve_tie(
    prompt: str,
    resp_a: str,
    resp_b: str,
    criterion: str,
    tie_breaker: str = "openai",
    model_name_or_path: Optional[str] = None,
    device: Optional[str] = None,
) -> str:
    """
    Return "A" or "B".
    tie_breaker:
      - "openai": use OpenAI API
      - "local": use local open-source model (requires model_name_or_path)
      - "random": random fallback (not recommended)
    
    Args:
        model_name_or_path: Required if tie_breaker="local", HuggingFace model name or path
        device: Optional device for local model ("cuda", "cpu", or None for auto-detect)
    """
    if tie_breaker == "openai":
        return tiebreak_with_llm_openai(prompt, resp_a, resp_b, criterion=criterion)
    if tie_breaker == "local":
        if model_name_or_path is None:
            raise ValueError("model_name_or_path is required when tie_breaker='local'")
        return tiebreak_with_llm_local(
            prompt, resp_a, resp_b, criterion=criterion, 
            model_name_or_path=model_name_or_path, device=device
        )
    if tie_breaker == "random":
        return random.choice(["A", "B"])
    raise ValueError(f"Unknown tie_breaker={tie_breaker}")


# -----------------------------
# 5) Build table
# -----------------------------
def build_multipref_preference_table(
    split: str = "train",
    max_examples: Optional[int] = None,
    seed: int = 42,
    tie_breaker: str = "openai",
    model_name_or_path: Optional[str] = None,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Output columns (similar style to your UltraFeedback table):
      - prompt
      - response_0 (A), response_1 (B)
      - for each criterion c in {harmless, helpful, truthful, overall}:
          - pref_c : "A"/"B" after vote and tie-break
          - rm_c_reward_0, rm_c_reward_1 : {0,1}
          - chosen_c : 0 if A wins, 1 if B wins
    """
    random.seed(seed)
    np.random.seed(seed)

    ds = load_multipref(split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    rows: List[Dict[str, Any]] = []

    for i in tqdm(range(len(ds)), desc=f"Processing MultiPref ({split})"):
        ex = ds[i]
        prompt, resp_a, resp_b = extract_prompt_and_responses(ex)

        # must have two responses
        if (not resp_a) or (not resp_b):
            continue

        ann_list = extract_all_annotation_lists(ex)

        row: Dict[str, Any] = {
            "prompt": prompt,
            "response_0": resp_a,  # A
            "response_1": resp_b,  # B
        }

        for c in CRITERIA:
            pref = aggregate_pref(ann_list, c)  # "A" / "B" / "Tie"
            if pref == "Tie":
                pref = resolve_tie(
                    prompt, resp_a, resp_b, criterion=c, 
                    tie_breaker=tie_breaker,
                    model_name_or_path=model_name_or_path,
                    device=device
                )

            row[f"pref_{c}"] = pref

            if pref == "A":
                row[f"rm_{c}_reward_0"] = 1
                row[f"rm_{c}_reward_1"] = 0
                row[f"chosen_{c}"] = 0
            else:
                row[f"rm_{c}_reward_0"] = 0
                row[f"rm_{c}_reward_1"] = 1
                row[f"chosen_{c}"] = 1

        rows.append(row)

    return pd.DataFrame(rows)


def split_data_6_2_2(
    df: pd.DataFrame,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    print("Loading and processing multipref...")
    df_all = build_multipref_preference_table(
        split="train",
        max_examples=None,
        seed=42,
        tie_breaker="local",  # "openai", "local", or "random"
        model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",  # Required if tie_breaker="local"
        device=None,  # "cuda", "cpu", or None for auto-detect
    )

    print(f"\nTotal examples: {len(df_all)}")

    # Optional downsample
    print("\nDownsampling to 20% of the data...")
    df_all = df_all.sample(frac=0.2, random_state=42).reset_index(drop=True)
    print(f"After downsampling: {len(df_all)} examples")

    print("\nSplitting into train:validation:test = 6:2:2...")
    train_df, val_df, test_df = split_data_6_2_2(df_all, seed=42)

    train_path = "data_out/multipref_train.csv"
    val_path = "data_out/multipref_validation.csv"
    test_path = "data_out/multipref_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSaved:")
    print(f"  Train: {train_path} ({len(train_df)} rows)")
    print(f"  Validation: {val_path} ({len(val_df)} rows)")
    print(f"  Test: {test_path} ({len(test_df)} rows)")
