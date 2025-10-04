"""Utilities to load the local IMDb preference dataset with source metadata."""

from __future__ import annotations

import json
from typing import Optional, Sequence, Union

from datasets import DatasetDict, load_dataset

_DEFAULT_PATH = "./data_processing/imdb_preference_dataset_with_source.json"
_DEFAULT_SUBSET_PATH = "./data_processing/imdb_preference_dataset_with_source_subset_indices.json"


def _make_dialogue(prompt: str, response: str):
    user_turn = {"role": "user", "content": prompt}
    assistant_turn = {"role": "assistant", "content": response}
    return [user_turn, assistant_turn]


def _format_example(example: dict):
    prompt_text = example.get("prompt", "")
    chosen_text = example.get("chosen", "")
    rejected_text = example.get("rejected", "")

    formatted = {
        "prompt": [{"role": "user", "content": prompt_text}],
        "messages": _make_dialogue(prompt_text, chosen_text),
        "chosen": _make_dialogue(prompt_text, chosen_text),
        "rejected": _make_dialogue(prompt_text, rejected_text),
    }

    if "preference_source" in example:
        formatted["preference_source"] = example["preference_source"]

    return formatted


def _has_required_content(example: dict) -> bool:
    for key in ("prompt", "chosen", "rejected"):
        value = example.get(key)
        if not isinstance(value, str):
            return False
        if value.strip() == "":
            return False
    return True


def _resolve_test_size(test_size: Optional[float], dataset_size: int) -> float:
    if test_size is None:
        return 0.1
    
    if isinstance(test_size, float):
        if 0 < test_size < 1:
            return test_size
        raise ValueError("test_size as a float must be in the (0, 1) interval")

    if dataset_size <= 1:
        return 0.5

    # This should not happen with the new type, but keeping for safety
    if test_size <= 0 or test_size >= dataset_size:
        fallback = max(1, int(0.1 * dataset_size))
        return fallback if fallback < dataset_size else 1

    return test_size


def load_imdb_preference_with_source(
    data_path: str = _DEFAULT_PATH,
    test_size: Optional[float] = 0.1,
    seed: int = 42,
    subset_indices: Union[Sequence[int], None] = None,
    subset_id: Union[int, None] = None,
    subset_indices_path: str = _DEFAULT_SUBSET_PATH,
    sources: Optional[Sequence[str]] = None,
) -> DatasetDict:
    """Load the local IMDb preference dataset formatted for preference optimization."""

    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.filter(_has_required_content)
    if sources:
        if isinstance(sources, str):
            source_set = {sources}
        else:
            source_set = set(sources)

        def _keep_source(example):
            return example.get("preference_source") in source_set

        dataset = dataset.filter(_keep_source)
    formatted = dataset.map(_format_example, remove_columns=dataset.column_names)

    resolved_test_size = _resolve_test_size(test_size, len(formatted))
    splits = formatted.train_test_split(test_size=resolved_test_size, seed=seed, shuffle=True)

    if subset_indices is None and subset_id is not None:
        with open(subset_indices_path, "r") as f:
            stored_subsets = json.load(f)
        if subset_id < 0 or subset_id >= len(stored_subsets):
            raise IndexError(
                f"subset_id {subset_id} is out of bounds for stored subsets of length {len(stored_subsets)}."
            )
        indices_from_file = stored_subsets[subset_id]["subset_indices"]
        subset_indices = indices_from_file

    if subset_indices is not None:
        if len(splits["train"]) == 0:
            raise ValueError("Cannot apply subset indices because the training split is empty.")
        indices_list = list(subset_indices)
        if not indices_list:
            raise ValueError("subset_indices must contain at least one index when provided.")
        max_index = len(splits["train"]) - 1
        for idx in indices_list:
            if not isinstance(idx, int):
                raise ValueError(f"subset_indices expects integers, but received {type(idx).__name__}.")
            if idx < 0 or idx > max_index:
                raise IndexError(
                    f"subset index {idx} is out of bounds for training split of size {len(splits['train'])}."
                )
        splits["train"] = splits["train"].select(indices_list)

    return DatasetDict({"train": splits["train"], "test": splits["test"]})
