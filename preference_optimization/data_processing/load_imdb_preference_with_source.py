"""Utilities to load the local IMDb preference dataset with source metadata."""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence, List

from datasets import DatasetDict, load_dataset, Dataset

_DEFAULT_PATH = "./data_processing/imdb_processed_data/annotated_data.json"
_DEFAULT_TRAIN_PATH = "./data_processing/imdb_processed_data/train_data.json"
_DEFAULT_TEST_PATH = "./data_processing/imdb_processed_data/test_data.json"


def _make_dialogue(prompt: str, response: str):
    user_turn = {"role": "user", "content": prompt}
    assistant_turn = {"role": "assistant", "content": response}
    return [user_turn, assistant_turn]


def _format_example(example: dict):
    prompt_text = example.get("prompt", "")
    
    # Handle responses array - extract chosen and rejected based on chosen index
    responses = example.get("responses", [])
    chosen_idx = example.get("chosen", 0)
    
    if len(responses) >= 2:
        chosen_text = responses[chosen_idx]
        rejected_text = responses[1 - chosen_idx]  # The other response
    else:
        # Fallback to old format if available
        chosen_text = example.get("chosen", "")
        rejected_text = example.get("rejected", "")

    formatted = {
        "prompt": [{"role": "user", "content": prompt_text}],
        "messages": _make_dialogue(prompt_text, chosen_text),
        "chosen": _make_dialogue(prompt_text, chosen_text),
        "rejected": _make_dialogue(prompt_text, rejected_text),
    }

    # Map our source field to preference_source for compatibility
    if "source" in example:
        formatted["preference_source"] = example["source"]
    elif "preference_source" in example:
        formatted["preference_source"] = example["preference_source"]
    
    # Keep annotator information
    if "annotator" in example:
        formatted["annotator"] = example["annotator"]

    return formatted


def _has_required_content(example: dict) -> bool:
    # Check for prompt
    prompt = example.get("prompt")
    if not isinstance(prompt, str) or prompt.strip() == "":
        return False
    
    # Check for responses array format (new format)
    responses = example.get("responses", [])
    if isinstance(responses, list) and len(responses) >= 2:
        for response in responses:
            if not isinstance(response, str) or response.strip() == "":
                return False
        return True
    
    # Check for old format (chosen, rejected as strings)
    for key in ("chosen", "rejected"):
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
    train_data_path: str = _DEFAULT_TRAIN_PATH,
    test_data_path: str = _DEFAULT_TEST_PATH,
    test_size: Optional[float] = None,  # Not used when loading pre-split data
    seed: int = 42,
    sources: Optional[Sequence[str]] = None,
    annotator_ids: Optional[List[int]] = None,
) -> DatasetDict:
    """
    Load the local IMDb preference dataset formatted for preference optimization.
    
    Args:
        data_path: Path to the annotated data JSON file (if using single file)
        train_data_path: Path to the train data JSON file
        test_data_path: Path to the test data JSON file
        test_size: Not used when loading pre-split data
        seed: Random seed
        sources: Sources to filter by (positive, negative, long, short)
        annotator_ids: List of annotator IDs to filter by. If None, loads all annotators.
    
    Returns:
        DatasetDict with train and test splits
    """
    
    # Check if we have separate train/test files
    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        # Load pre-split data
        with open(train_data_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Filter train and test data separately
        filtered_train = _filter_data(train_data, sources, annotator_ids)
        filtered_test = _filter_data(test_data, sources, annotator_ids)
        
        train_items = filtered_train
        test_items = filtered_test
        
    else:
        # Load from single file and split
        dataset = load_dataset("json", data_files=data_path, split="train")
        dataset = dataset.filter(_has_required_content)
        
        # Convert to list for filtering
        all_data = list(dataset)
        filtered_data = _filter_data(all_data, sources, annotator_ids)
        
        # Convert back to dataset for splitting
        filtered_dataset = Dataset.from_list(filtered_data)
        formatted = filtered_dataset.map(_format_example, remove_columns=filtered_dataset.column_names)
        
        # Split data
        resolved_test_size = _resolve_test_size(test_size, len(formatted))
        splits = formatted.train_test_split(test_size=resolved_test_size, seed=seed, shuffle=True)
        
        train_items = splits["train"]
        test_items = splits["test"]
    
    # Ensure we have Dataset objects
    if not hasattr(train_items, 'map'):
        train_dataset = Dataset.from_list(train_items)
        train_dataset = train_dataset.map(_format_example, remove_columns=train_dataset.column_names)
    else:
        train_dataset = train_items
    
    if not hasattr(test_items, 'map'):
        test_dataset = Dataset.from_list(test_items)
        test_dataset = test_dataset.map(_format_example, remove_columns=test_dataset.column_names)
    else:
        test_dataset = test_items
    
    return DatasetDict({"train": train_dataset, "test": test_dataset})


def _filter_data(data: List[dict], sources: Optional[Sequence[str]], annotator_ids: Optional[List[int]]) -> List[dict]:
    """
    Filter data based on sources and annotator IDs.
    
    Args:
        data: List of data items
        sources: Sources to filter by
        annotator_ids: Annotator IDs to filter by
    
    Returns:
        Filtered list of data items
    """
    filtered = data
    
    # Filter by sources
    if sources:
        if isinstance(sources, str):
            source_set = {sources}
        else:
            source_set = set(sources)
        
        def _keep_source(example):
            source = example.get("source") or example.get("preference_source")
            return source in source_set
        
        filtered = [item for item in filtered if _keep_source(item)]
    
    # Filter by annotator IDs
    if annotator_ids is not None:
        annotator_set = set(annotator_ids)
        
        def _keep_annotator(example):
            annotator = example.get("annotator")
            return annotator in annotator_set
        
        filtered = [item for item in filtered if _keep_annotator(item)]
    
    return filtered
