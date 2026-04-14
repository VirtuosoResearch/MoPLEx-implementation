# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random
from typing import Any

import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets

from .configs import ScriptArguments


logger = logging.getLogger(__name__)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if value and isinstance(value[0], dict) and "content" in value[0]:
            return "\n".join(item.get("content", "") for item in value if isinstance(item, dict))
        return "\n".join(str(item) for item in value)
    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"])
        if "text" in value:
            return str(value["text"])
    return str(value)


def _extract_dimension_score(
    item: dict[str, Any],
    row: dict[str, Any],
    idx: int,
    dimension: str,
    scores_key: str, # not used in UltraFeedback
    annotations_key: str,
) -> float | None:
    # Most common format: completion[scores_key][dimension]
    if isinstance(item.get(scores_key), dict) and dimension in item[scores_key]:
        direct_score = _to_float(item[scores_key][dimension])
        if direct_score is not None:
            return direct_score

        # Handle nested annotation format: scores[dimension]['Rating']
        nested = item[scores_key][dimension]
        if isinstance(nested, dict):
            for key in ("Rating", "rating", "score", "Score"):
                score = _to_float(nested.get(key))
                if score is not None:
                    return score

    # Fallback format: completion[annotations_key][dimension]
    if isinstance(item.get(annotations_key), dict) and dimension in item[annotations_key]:
        direct_score = _to_float(item[annotations_key][dimension])
        if direct_score is not None:
            return direct_score

        nested = item[annotations_key][dimension]
        if isinstance(nested, dict):
            for key in ("Rating", "rating", "score", "Score"):
                score = _to_float(nested.get(key))
                if score is not None:
                    return score

    # Fallback format: completion[dimension]
    if dimension in item:
        score = _to_float(item[dimension])
        if score is not None:
            return score
    
    # Fallback format: row-level score list aligned with response index
    row_key = f"{dimension}_scores"
    if isinstance(row.get(row_key), list) and idx < len(row[row_key]):
        return _to_float(row[row_key][idx])
    return None


def _extract_candidates(row: dict[str, Any], args: ScriptArguments) -> list[tuple[str, float]]:
    raw_responses = row.get(args.listwise_responses_column)
    if raw_responses is None:
        # Allow common fallback names.
        raw_responses = row.get("responses", row.get("completions"))

    if not isinstance(raw_responses, list):
        return []

    pairs: list[tuple[str, float]] = [] # containing (response_text, score) pairs
    for idx, item in enumerate(raw_responses):
        text: str | None = None
        score: float | None = None

        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text_val = item.get(args.listwise_response_text_key, item.get("text", item.get("content")))
            if text_val is not None:
                text = _as_text(text_val)
            score = _extract_dimension_score(
                item=item,
                row=row,
                idx=idx,
                dimension=args.preference_dimension,
                scores_key=args.listwise_scores_key,
                annotations_key=args.listwise_annotations_key,
            )

        if text is None or score is None:
            continue
        text = text.strip()
        if not text:
            continue
        pairs.append((text, score))

    if not pairs:
        return []
    return pairs


def _to_listwise_dataset(dataset: Dataset, args: ScriptArguments) -> Dataset:
    rows = []
    rng = random.Random(args.dataset_mixture.seed)
    for row in dataset:
        raw_prompt = row.get(args.listwise_prompt_column)
        prompt = _as_text(raw_prompt).strip()
        if not prompt:
            continue

        candidates = _extract_candidates(row, args)
        if len(candidates) < args.listwise_min_responses:
            continue

        ranked = sorted(candidates, key=lambda x: x[1], reverse=True)
        if len(ranked) > args.listwise_num_responses:
            ranked = rng.sample(ranked, args.listwise_num_responses)
            ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
        if len(ranked) < args.listwise_min_responses:
            continue

        responses = [x[0] for x in ranked]
        scores = [float(x[1]) for x in ranked]
        rows.append(
            {
                "prompt": prompt,
                "responses": responses,
                "scores": scores,
                "preference_dimension": args.preference_dimension,
            }
        )

    if not rows:
        raise ValueError(
            "No listwise examples could be built from the dataset. "
            "Check listwise column/key settings and preference_dimension."
        )
    return Dataset.from_list(rows)


def _maybe_convert_to_listwise(dataset_dict: DatasetDict, args: ScriptArguments) -> DatasetDict:
    if args.dataset_format != "listwise":
        return dataset_dict

    converted = {}
    for split_name, split_data in dataset_dict.items():
        logger.info(
            "Converting split '%s' to listwise format for dimension '%s'",
            split_name,
            args.preference_dimension,
        )
        converted[split_name] = _to_listwise_dataset(split_data, args)
        logger.info("Built %d listwise examples for split '%s'", len(converted[split_name]), split_name)
    return DatasetDict(converted)


def get_dataset(args: ScriptArguments) -> DatasetDict:
    """Load a dataset or a mixture of datasets based on the configuration.

    Args:
        args (ScriptArguments): Script arguments containing dataset configuration.

    Returns:
        DatasetDict: The loaded datasets.
    """
    if args.dataset_name and not args.dataset_mixture:
        logger.info(f"Loading dataset: {args.dataset_name}")
        dataset = datasets.load_dataset(args.dataset_name, args.dataset_config)
        return _maybe_convert_to_listwise(dataset, args)
    elif args.dataset_mixture:
        logger.info(f"Creating dataset mixture with {len(args.dataset_mixture.datasets)} datasets")
        seed = args.dataset_mixture.seed
        datasets_list = []

        for dataset_config in args.dataset_mixture.datasets:
            logger.info(f"Loading dataset for mixture: {dataset_config.id} (config: {dataset_config.config})")
            ds = datasets.load_dataset(
                dataset_config.id,
                dataset_config.config,
                split=dataset_config.split,
            )
            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)
            if dataset_config.weight is not None:
                ds = ds.shuffle(seed=seed).select(range(int(len(ds) * dataset_config.weight)))
                logger.info(
                    f"Subsampled dataset '{dataset_config.id}' (config: {dataset_config.config}) with weight={dataset_config.weight} to {len(ds)} examples"
                )

            datasets_list.append(ds)

        if datasets_list:
            combined_dataset = concatenate_datasets(datasets_list)
            combined_dataset = combined_dataset.shuffle(seed=seed)
            logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

            if args.dataset_mixture.test_split_size is not None:
                combined_dataset = combined_dataset.train_test_split(
                    test_size=args.dataset_mixture.test_split_size, seed=seed
                )
                logger.info(
                    f"Split dataset into train and test sets with test size: {args.dataset_mixture.test_split_size}"
                )
                return _maybe_convert_to_listwise(combined_dataset, args)
            else:
                return _maybe_convert_to_listwise(DatasetDict({"train": combined_dataset}), args)
        else:
            raise ValueError("No datasets were loaded from the mixture configuration")

    else:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")
