import os
import random
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


METRICS = ["rougeL", "bleu", "bertscore", "bleurt", "bartscore"]


def load_mix_instruct(split: str):
    return load_dataset("llm-blender/mix-instruct", split=split)


def build_prompt(instruction: str, inp: str) -> str:
    instruction = (instruction or "").strip()
    inp = (inp or "").strip()
    if instruction and inp:
        return f"Instruction:\n{instruction}\n\nInput:\n{inp}"
    if instruction:
        return f"Instruction:\n{instruction}"
    return inp


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def get_metric_scores(cand: Dict[str, Any]) -> Optional[Dict[str, float]]:
    scores = cand.get("scores", None)
    if not isinstance(scores, dict):
        return None

    out: Dict[str, float] = {}
    for m in METRICS:
        v = safe_float(scores.get(m, None))
        if v is None:
            return None
        out[m] = v
    return out


def process_split(
    split: str,
    seed: int = 42,
    require_distinct_text: bool = True,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    ds = load_mix_instruct(split=split)
    rows: List[Dict[str, Any]] = []

    for i in tqdm(range(len(ds)), desc=f"Processing mix-instruct ({split})"):
        ex = ds[i]
        prompt = build_prompt(ex.get("instruction", ""), ex.get("input", ""))
        if not prompt:
            continue

        candidates = ex.get("candidates", None)
        if not isinstance(candidates, list) or len(candidates) < 2:
            continue

        usable: List[Tuple[float, str, Dict[str, float]]] = []
        for c in candidates:
            if not isinstance(c, dict):
                continue

            text = c.get("text", "")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue

            ms = get_metric_scores(c)
            if ms is None:
                continue

            agg = float(np.mean([ms[m] for m in METRICS]))
            usable.append((agg, text, ms))

        if len(usable) < 2:
            continue

        usable.sort(key=lambda x: x[0])
        _, text_1, s_1 = usable[0]     # worst
        _, text_0, s_0 = usable[-1]    # best

        if require_distinct_text and text_0 == text_1:
            continue

        rows.append(
            {
                "prompt": prompt,
                "response_0": text_0,
                "response_1": text_1,
                "rougeL_0": s_0["rougeL"],
                "rougeL_1": s_1["rougeL"],
                "bleu_0": s_0["bleu"],
                "bleu_1": s_1["bleu"],
                "bertscore_0": s_0["bertscore"],
                "bertscore_1": s_1["bertscore"],
                "bleurt_0": s_0["bleurt"],
                "bleurt_1": s_1["bleurt"],
                "bartscore_0": s_0["bartscore"],
                "bartscore_1": s_1["bartscore"],
            }
        )

    return pd.DataFrame(rows)


def downsample_df(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    if frac >= 1.0:
        return df.reset_index(drop=True)
    # ensure at least 1 row if df is non-empty and frac is very small
    n = len(df)
    k = max(1, int(round(n * frac)))
    return df.sample(n=k, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    seed = 42
    frac = 0.1

    os.makedirs("data_out", exist_ok=True)

    for split in ["train", "validation", "test"]:
        df = process_split(split=split, seed=seed, require_distinct_text=True)
        print(f"\n[{split}] processed rows: {len(df)}")

        df_ds = downsample_df(df, frac=frac, seed=seed)
        print(f"[{split}] downsampled rows (frac={frac}): {len(df_ds)}")

        out_path = f"data_out/mix_instruct_pref_{split}_ds{str(frac).replace('.', '')}.csv"
        df_ds.to_csv(out_path, index=False)
        print(f"[{split}] saved: {out_path}")
