from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict


DIMENSIONS = (
    "instruction_following",
    "honesty",
    "truthfulness",
    "helpfulness",
)


@dataclass
class CyclicFilterStats:
    total_rows: int = 0
    eligible_rows: int = 0
    valid_score_rows: int = 0
    cycle_rows: int = 0


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
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
            return "\n".join(str(item.get("content", "")) for item in value if isinstance(item, dict))
        return "\n".join(str(item) for item in value)
    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"])
        if "text" in value:
            return str(value["text"])
    return str(value)


def _extract_dimension_score(
    completion: dict[str, Any],
    dimension: str,
    scores_key: str,
    annotations_key: str,
) -> float | None:
    if isinstance(completion.get(scores_key), dict) and dimension in completion[scores_key]:
        value = completion[scores_key][dimension]
        direct = _to_float(value)
        if direct is not None:
            return direct
        if isinstance(value, dict):
            for nested_key in ("Rating", "rating", "score", "Score"):
                nested_score = _to_float(value.get(nested_key))
                if nested_score is not None:
                    return nested_score

    if isinstance(completion.get(annotations_key), dict) and dimension in completion[annotations_key]:
        value = completion[annotations_key][dimension]
        direct = _to_float(value)
        if direct is not None:
            return direct
        if isinstance(value, dict):
            for nested_key in ("Rating", "rating", "score", "Score"):
                nested_score = _to_float(value.get(nested_key))
                if nested_score is not None:
                    return nested_score

    if dimension in completion:
        return _to_float(completion[dimension])

    return None


def _has_strict_order(scores: list[float], order: tuple[int, int, int, int]) -> bool:
    ordered_scores = [scores[idx] for idx in order]
    return all(ordered_scores[i] > ordered_scores[i + 1] for i in range(len(ordered_scores) - 1))


def _rotate_order(order: tuple[int, int, int, int], shift: int) -> tuple[int, int, int, int]:
    shift = shift % len(order)
    return order[shift:] + order[:shift]


def _strict_desc_order(scores: list[float]) -> tuple[int, int, int, int] | None:
    if len(scores) != 4:
        return None
    order = tuple(sorted(range(4), key=lambda idx: scores[idx], reverse=True))
    if not _has_strict_order(scores, order):
        return None
    return order


def _validate_dimensions(dimensions: tuple[str, ...]) -> tuple[str, ...]:
    if len(dimensions) < 2:
        raise ValueError("`dimensions` must contain at least 2 entries for rotated filtering.")
    if len(dimensions) > 4:
        raise ValueError("`dimensions` can contain at most 4 entries.")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("`dimensions` must not contain duplicates.")
    invalid = [dim for dim in dimensions if dim not in DIMENSIONS]
    if invalid:
        raise ValueError(f"Unknown dimensions: {invalid}. Valid values: {list(DIMENSIONS)}")
    return dimensions


def _infer_shift(base_order: tuple[int, int, int, int], dim_order: tuple[int, int, int, int]) -> int | None:
    for shift in range(1, len(base_order)):
        if _rotate_order(base_order, shift) == dim_order:
            return shift
    return None


def _find_rotated_cycle_layout(
    scores_by_dimension: dict[str, list[float]],
    dimensions: tuple[str, ...],
) -> tuple[tuple[int, int, int, int], dict[str, int]] | None:
    dimensions = _validate_dimensions(dimensions)
    base_dimension = dimensions[0]
    base_scores = scores_by_dimension.get(base_dimension)
    if base_scores is None:
        return None
    base_order = _strict_desc_order(base_scores)
    if base_order is None:
        return None

    shifts: dict[str, int] = {base_dimension: 0}
    used_shifts = {0}

    for dimension in dimensions[1:]:
        if dimension not in scores_by_dimension:
            return None
        scores = scores_by_dimension[dimension]
        dim_order = _strict_desc_order(scores)
        if dim_order is None:
            return None
        shift = _infer_shift(base_order, dim_order)
        if shift is None:
            return None
        if shift in used_shifts:
            return None
        shifts[dimension] = shift
        used_shifts.add(shift)

    return base_order, shifts


def find_strict_rotated_cycle_base_order(
    scores_by_dimension: dict[str, list[float]],
    dimensions: tuple[str, ...] = DIMENSIONS,
) -> tuple[int, int, int, int] | None:
    """
    Return the base response order (A, B, C, D) if dimensions form a strict 4-cycle.

        Any permutation is accepted for the base order. The first selected dimension defines
        the base order P=[A, B, C, D]. Every other selected dimension must be a non-zero
        rotation of P, and non-zero shifts must be distinct across dimensions.
    """
    layout = _find_rotated_cycle_layout(scores_by_dimension, dimensions)
    if layout is None:
        return None
    return layout[0]


def has_strict_rotated_cycle(
    scores_by_dimension: dict[str, list[float]],
    dimensions: tuple[str, ...] = DIMENSIONS,
) -> bool:
    return _find_rotated_cycle_layout(scores_by_dimension, dimensions) is not None


def _extract_response_text(completion: Any, response_text_key: str) -> str | None:
    if isinstance(completion, str):
        text = completion.strip()
        return text if text else None
    if not isinstance(completion, dict):
        return None

    text_value = completion.get(response_text_key, completion.get("text", completion.get("content")))
    if text_value is None:
        return None
    text = _as_text(text_value).strip()
    return text if text else None


def _iter_four_candidate_groups(num_candidates: int, rng: random.Random) -> list[tuple[int, int, int, int]]:
    combos = list(itertools.combinations(range(num_candidates), 4))
    rng.shuffle(combos)
    return combos


def build_cyclic_rows(
    dataset: Dataset,
    *,
    prompt_column: str = "instruction",
    responses_column: str = "completions",
    response_text_key: str = "response",
    scores_key: str = "scores",
    annotations_key: str = "annotations",
    dimensions: tuple[str, ...] = DIMENSIONS,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], CyclicFilterStats]:
    dimensions = _validate_dimensions(dimensions)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    stats = CyclicFilterStats()

    for row_idx, row in enumerate(dataset):
        stats.total_rows += 1

        prompt_raw = row.get(prompt_column)
        prompt = _as_text(prompt_raw).strip()
        if not prompt:
            continue

        completions = row.get(responses_column)
        if not isinstance(completions, list) or len(completions) < 4:
            continue
        stats.eligible_rows += 1

        found = False
        for candidate_indices in _iter_four_candidate_groups(len(completions), rng):
            selected = [completions[idx] for idx in candidate_indices]

            responses: list[str] = []
            scores_by_dimension: dict[str, list[float]] = {dim: [] for dim in dimensions}
            valid = True

            for completion in selected:
                text = _extract_response_text(completion, response_text_key)
                if text is None:
                    valid = False
                    break
                responses.append(text)

                if not isinstance(completion, dict):
                    valid = False
                    break
                for dimension in dimensions:
                    score = _extract_dimension_score(
                        completion,
                        dimension=dimension,
                        scores_key=scores_key,
                        annotations_key=annotations_key,
                    )
                    if score is None:
                        valid = False
                        break
                    scores_by_dimension[dimension].append(score)
                if not valid:
                    break

            if not valid:
                continue

            stats.valid_score_rows += 1
            layout = _find_rotated_cycle_layout(scores_by_dimension, dimensions=dimensions)
            if layout is not None:
                base_order, shifts = layout
                # Emit one listwise row per dimension, keeping a consistent cyclic candidate set.
                for dimension in dimensions:
                    dim_order = _rotate_order(base_order, shifts[dimension])
                    ordered_responses = [responses[idx] for idx in dim_order]
                    ordered_scores = [scores_by_dimension[dimension][idx] for idx in dim_order]
                    rows.append(
                        {
                            "prompt": prompt,
                            "responses": ordered_responses,
                            "scores": ordered_scores,
                            "preference_dimension": dimension,
                            "source_index": int(row_idx),
                        }
                    )
                stats.cycle_rows += 1
                found = True
                break

        if not found:
            continue

    return rows, stats


def split_rows_to_dataset_dict(rows: list[dict[str, Any]], seed: int = 0) -> DatasetDict:
    if not rows:
        raise ValueError("No cyclic rows found. Nothing to split.")

    dataset = Dataset.from_list(rows)
    first_split = dataset.train_test_split(test_size=0.2, seed=seed)
    second_split = first_split["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict(
        {
            "train": first_split["train"],
            "validation": second_split["train"],
            "test": second_split["test"],
        }
    )