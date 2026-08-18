#!/usr/bin/env python
"""Rank original and augmented responses with criterion-specific PL/DPO adapters."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger(__name__)
DEFAULT_CRITERIA = (
    "instruction_following",
    "honesty",
    "truthfulness",
    "helpfulness",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--criterion_checkpoint",
        action="append",
        default=[],
        metavar="CRITERION=PATH",
        help="Criterion-to-LoRA checkpoint mapping. Repeat once per criterion.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_rows", type=int, default=None, help="Optional row limit per criterion.")
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--expected_num_original", type=int, default=4)
    parser.add_argument("--expected_num_augmented", type=int, default=4)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def parse_criterion_checkpoints(values: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --criterion_checkpoint {value!r}; expected CRITERION=PATH.")
        criterion, checkpoint = value.split("=", 1)
        criterion, checkpoint = criterion.strip(), checkpoint.strip()
        if not criterion or not checkpoint:
            raise ValueError(f"Invalid --criterion_checkpoint {value!r}; both fields must be non-empty.")
        if criterion in mapping:
            raise ValueError(f"Duplicate checkpoint mapping for criterion {criterion!r}.")
        mapping[criterion] = checkpoint
    if not mapping:
        raise ValueError("Provide at least one --criterion_checkpoint CRITERION=PATH mapping.")
    return mapping


def validate_checkpoints(mapping: dict[str, str]) -> tuple[str, dict[str, str]]:
    base_models: dict[str, str] = {}
    resolved: dict[str, str] = {}
    for criterion, raw_path in mapping.items():
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory for {criterion!r} does not exist: {path}")
        config_path = path / "adapter_config.json"
        weights_path = path / "adapter_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"Incomplete checkpoint for {criterion!r} at {path}; expected adapter_config.json "
                "and adapter_model.safetensors."
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        base_model = config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError(f"Checkpoint {path} does not declare base_model_name_or_path.")
        base_models[criterion] = str(base_model)
        resolved[criterion] = str(path)
    unique_base_models = set(base_models.values())
    if len(unique_base_models) != 1:
        raise ValueError(f"All criterion adapters must share one base model; found {base_models}.")
    return next(iter(unique_base_models)), resolved


def load_dataset_dict(dataset_name: str, dataset_config: str | None = None) -> DatasetDict:
    if os.path.isdir(dataset_name):
        dataset = load_from_disk(dataset_name)
    else:
        dataset = load_dataset(dataset_name, dataset_config)
    if isinstance(dataset, Dataset):
        return DatasetDict({"train": dataset})
    if isinstance(dataset, DatasetDict):
        return dataset
    raise TypeError(f"Unsupported dataset object loaded from {dataset_name!r}: {type(dataset)}")


def label_candidates(
    row: dict[str, Any],
    *,
    expected_num_original: int = 4,
    expected_num_augmented: int = 4,
) -> list[dict[str, Any]]:
    responses = row.get("responses")
    if not isinstance(responses, list):
        raise ValueError("Each row must contain a list-valued `responses` field.")
    if "ranked_prefix_length" not in row or "num_generated_responses" not in row:
        raise ValueError("Each row must contain `ranked_prefix_length` and `num_generated_responses` metadata.")
    try:
        num_original = int(row["ranked_prefix_length"])
        num_augmented = int(row["num_generated_responses"])
    except (TypeError, ValueError) as exc:
        raise ValueError("`ranked_prefix_length` and `num_generated_responses` must be integers.") from exc
    if num_original != expected_num_original or num_augmented != expected_num_augmented:
        raise ValueError(
            "Unexpected original/augmented response counts for "
            f"criterion={row.get('preference_dimension')!r}, source_index={row.get('source_index')!r}: "
            f"got {num_original}+{num_augmented}, expected {expected_num_original}+{expected_num_augmented}."
        )
    if len(responses) != num_original + num_augmented:
        raise ValueError(
            f"Response count {len(responses)} does not equal ranked_prefix_length + num_generated_responses "
            f"({num_original + num_augmented})."
        )

    candidates = []
    for position, response in enumerate(responses):
        response_type = "original" if position < num_original else "augmented"
        type_position = position + 1 if response_type == "original" else position - num_original + 1
        candidates.append(
            {
                "candidate_label": f"{response_type}_{type_position}",
                "response_type": response_type,
                "response": str(response),
                "dataset_position": position,
            }
        )
    return candidates


def rank_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidates:
        raise ValueError("Cannot rank an empty candidate list.")
    ranked = sorted(candidates, key=lambda item: (-float(item["pl_utility"]), int(item["dataset_position"])))
    for rank, candidate in enumerate(ranked, start=1):
        candidate["predicted_rank"] = rank

    originals = [item for item in candidates if item["response_type"] == "original"]
    augmented = [item for item in candidates if item["response_type"] == "augmented"]
    if not originals or not augmented:
        raise ValueError("Ranking metrics require at least one original and one augmented response.")
    original_scores = [float(item["pl_utility"]) for item in originals]
    augmented_scores = [float(item["pl_utility"]) for item in augmented]
    min_original = min(original_scores)
    max_augmented = max(augmented_scores)
    pairwise_lower = sum(a_score < o_score for a_score in augmented_scores for o_score in original_scores)
    pairwise_ties = sum(a_score == o_score for a_score in augmented_scores for o_score in original_scores)
    below_all = sum(a_score < min_original for a_score in augmented_scores)
    top_augmented_rank = min(int(item["predicted_rank"]) for item in augmented)

    flags = {
        "num_original": len(originals),
        "num_augmented": len(augmented),
        "num_original_augmented_pairs": len(originals) * len(augmented),
        "num_augmented_lower_pairs": pairwise_lower,
        "num_original_augmented_ties": pairwise_ties,
        "num_augmented_below_all_originals": below_all,
        "all_augmented_strictly_bottom": max_augmented < min_original,
        "all_augmented_weakly_bottom": max_augmented <= min_original,
        "boundary_tie": max_augmented == min_original,
        "top_augmented_rank": top_augmented_rank,
        "sum_original_ranks": sum(int(item["predicted_rank"]) for item in originals),
        "sum_augmented_ranks": sum(int(item["predicted_rank"]) for item in augmented),
        "ranking": " > ".join(item["candidate_label"] for item in ranked),
    }
    return ranked, flags


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def summarize_rankings(records: list[dict[str, Any]]) -> dict[str, Any]:
    num_rows = len(records)
    num_original = sum(int(row["num_original"]) for row in records)
    num_augmented = sum(int(row["num_augmented"]) for row in records)
    num_pairs = sum(int(row["num_original_augmented_pairs"]) for row in records)
    lower_pairs = sum(int(row["num_augmented_lower_pairs"]) for row in records)
    tied_pairs = sum(int(row["num_original_augmented_ties"]) for row in records)
    below_all = sum(int(row["num_augmented_below_all_originals"]) for row in records)
    strict_bottom = sum(bool(row["all_augmented_strictly_bottom"]) for row in records)
    weak_bottom = sum(bool(row["all_augmented_weakly_bottom"]) for row in records)
    boundary_ties = sum(bool(row["boundary_tie"]) for row in records)
    rank_histogram = Counter(int(row["top_augmented_rank"]) for row in records)
    return {
        "num_prompt_criterion_rows": num_rows,
        "num_responses": num_original + num_augmented,
        "num_original_responses": num_original,
        "num_augmented_responses": num_augmented,
        "num_original_augmented_pairs": num_pairs,
        "num_augmented_lower_pairs": lower_pairs,
        "augmented_below_original_pairwise_rate": _safe_rate(lower_pairs, num_pairs),
        "num_original_augmented_ties": tied_pairs,
        "original_augmented_pairwise_tie_rate": _safe_rate(tied_pairs, num_pairs),
        "num_augmented_below_all_originals": below_all,
        "augmented_below_all_originals_rate": _safe_rate(below_all, num_augmented),
        "num_prompts_all_augmented_strictly_bottom": strict_bottom,
        "all_augmented_strictly_bottom_rate": _safe_rate(strict_bottom, num_rows),
        "num_prompts_all_augmented_weakly_bottom": weak_bottom,
        "all_augmented_weakly_bottom_rate": _safe_rate(weak_bottom, num_rows),
        "num_boundary_ties": boundary_ties,
        "boundary_tie_rate": _safe_rate(boundary_ties, num_rows),
        "mean_top_augmented_rank": _safe_rate(sum(int(row["top_augmented_rank"]) for row in records), num_rows),
        "top_augmented_rank_histogram": {str(rank): rank_histogram.get(rank, 0) for rank in range(1, 6)},
        "mean_original_rank": _safe_rate(sum(int(row["sum_original_ranks"]) for row in records), num_original),
        "mean_augmented_rank": _safe_rate(sum(int(row["sum_augmented_ranks"]) for row in records), num_augmented),
    }


def macro_average_summaries(by_criterion: dict[str, dict[str, Any]]) -> dict[str, float]:
    rate_fields = (
        "augmented_below_original_pairwise_rate",
        "original_augmented_pairwise_tie_rate",
        "augmented_below_all_originals_rate",
        "all_augmented_strictly_bottom_rate",
        "all_augmented_weakly_bottom_rate",
        "boundary_tie_rate",
        "mean_top_augmented_rank",
        "mean_original_rank",
        "mean_augmented_rank",
    )
    if not by_criterion:
        return {field: 0.0 for field in rate_fields}
    return {
        field: sum(float(summary[field]) for summary in by_criterion.values()) / len(by_criterion)
        for field in rate_fields
    }


def build_sequence(
    tokenizer,
    prompt: str,
    response: str,
    *,
    max_length: int,
    max_prompt_length: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
        response_ids = response_ids + [eos_id]
    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        original_prompt_length = len(prompt_ids)
        if overflow < original_prompt_length:
            prompt_ids = prompt_ids[overflow:]
        else:
            prompt_ids = []
            response_ids = response_ids[overflow - original_prompt_length :]
        input_ids = prompt_ids + response_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids,
    }


def pad_sequences(tokenizer, sequences: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")
    max_length = max(len(sequence["input_ids"]) for sequence in sequences)
    padded = {"input_ids": [], "attention_mask": [], "labels": []}
    for sequence in sequences:
        pad_length = max_length - len(sequence["input_ids"])
        padded["input_ids"].append(sequence["input_ids"] + [pad_id] * pad_length)
        padded["attention_mask"].append(sequence["attention_mask"] + [0] * pad_length)
        padded["labels"].append(sequence["labels"] + [-100] * pad_length)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in padded.items()}


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid, 0)
    label_logits = shift_logits.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = label_logits - torch.logsumexp(shift_logits, dim=-1)
    return (token_logps * valid).sum(dim=-1)


def _model_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


def _adapter_name(criterion: str) -> str:
    return "criterion_" + (re.sub(r"[^0-9A-Za-z_]+", "_", criterion).strip("_") or "unknown")


def _dtype_from_name(name: str) -> Any:
    if name == "auto":
        return "auto"
    if not hasattr(torch, name):
        raise ValueError(f"Unknown torch dtype {name!r}.")
    return getattr(torch, name)


class CriterionAdapterScorer:
    def __init__(
        self,
        *,
        base_model_name: str,
        checkpoint_map: dict[str, str],
        batch_size: int,
        max_length: int,
        max_prompt_length: int,
        torch_dtype: str,
        device_map: str,
        attn_implementation: str | None,
        trust_remote_code: bool,
        disable_tqdm: bool,
    ) -> None:
        from peft import PeftModel

        self.batch_size = batch_size
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.disable_tqdm = disable_tqdm
        first_checkpoint = next(iter(checkpoint_map.values()))
        self.tokenizer = AutoTokenizer.from_pretrained(first_checkpoint, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        model_kwargs: dict[str, Any] = {
            "torch_dtype": _dtype_from_name(torch_dtype),
            "trust_remote_code": trust_remote_code,
        }
        if device_map != "none":
            model_kwargs["device_map"] = device_map
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
        first_criterion, first_checkpoint = next(iter(checkpoint_map.items()))
        first_adapter = _adapter_name(first_criterion)
        self.model = PeftModel.from_pretrained(base_model, first_checkpoint, adapter_name=first_adapter)
        self.criterion_to_adapter = {first_criterion: first_adapter}
        for criterion, checkpoint in list(checkpoint_map.items())[1:]:
            adapter_name = _adapter_name(criterion)
            self.model.load_adapter(checkpoint, adapter_name=adapter_name)
            self.criterion_to_adapter[criterion] = adapter_name
        self.model.eval()

    @torch.inference_mode()
    def score_pairs(self, pairs: list[tuple[str, str]], criterion: str | None) -> list[float]:
        values: list[float] = []
        description = "Scoring base model" if criterion is None else f"Scoring {criterion}"
        iterator = range(0, len(pairs), self.batch_size)
        for start in tqdm(iterator, desc=description, unit="batch", disable=self.disable_tqdm):
            sequences = [
                build_sequence(
                    self.tokenizer,
                    prompt,
                    response,
                    max_length=self.max_length,
                    max_prompt_length=self.max_prompt_length,
                )
                for prompt, response in pairs[start : start + self.batch_size]
            ]
            batch = pad_sequences(self.tokenizer, sequences)
            device = _model_input_device(self.model)
            batch = {key: tensor.to(device) for key, tensor in batch.items()}
            if criterion is None:
                context = self.model.disable_adapter()
            else:
                self.model.set_adapter(self.criterion_to_adapter[criterion])
                context = nullcontext()
            with context:
                logits = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
            logps = sequence_logps(logits, batch["labels"])
            values.extend(float(value) for value in logps.detach().float().cpu().tolist())
            del logits, logps, batch
        return values


def prepare_rows(
    dataset: Dataset,
    checkpoint_map: dict[str, str],
    *,
    max_rows: int | None,
    expected_num_original: int,
    expected_num_augmented: int,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_criterion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_keys: set[tuple[str, Any]] = set()
    for dataset_row_index, raw_row in enumerate(dataset):
        row = dict(raw_row)
        criterion = str(row.get("preference_dimension", ""))
        if criterion not in checkpoint_map:
            continue
        if max_rows is not None and len(rows_by_criterion[criterion]) >= max_rows:
            continue
        key = (criterion, row.get("source_index"))
        if key in seen_keys:
            raise ValueError(f"Duplicate criterion/source_index row found: {key}")
        seen_keys.add(key)
        row["dataset_row_index"] = dataset_row_index
        row["candidates"] = label_candidates(
            row,
            expected_num_original=expected_num_original,
            expected_num_augmented=expected_num_augmented,
        )
        rows_by_criterion[criterion].append(row)

    missing = [criterion for criterion in checkpoint_map if not rows_by_criterion.get(criterion)]
    if missing:
        raise ValueError(f"Dataset has no selected rows for criterion checkpoint(s): {missing}")
    dataset_criteria = {str(value) for value in dataset["preference_dimension"]}
    unmapped = sorted(dataset_criteria.difference(checkpoint_map))
    if unmapped:
        raise ValueError(f"Dataset contains criteria without checkpoints: {unmapped}")
    return dict(rows_by_criterion)


def score_and_rank_rows(
    rows_by_criterion: dict[str, list[dict[str, Any]]],
    scorer: CriterionAdapterScorer,
    *,
    beta: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique_pairs: list[tuple[str, str]] = []
    pair_to_index: dict[tuple[str, str], int] = {}
    for rows in rows_by_criterion.values():
        for row in rows:
            prompt = str(row["prompt"])
            for candidate in row["candidates"]:
                pair = (prompt, candidate["response"])
                if pair not in pair_to_index:
                    pair_to_index[pair] = len(unique_pairs)
                    unique_pairs.append(pair)
    LOGGER.info("Scoring %d unique prompt-response pairs with the base model.", len(unique_pairs))
    base_values = scorer.score_pairs(unique_pairs, criterion=None)
    base_cache = {pair: base_values[index] for pair, index in pair_to_index.items()}

    ranking_records: list[dict[str, Any]] = []
    response_records: list[dict[str, Any]] = []
    for criterion, rows in rows_by_criterion.items():
        candidate_refs = [(row, candidate) for row in rows for candidate in row["candidates"]]
        pairs = [(str(row["prompt"]), candidate["response"]) for row, candidate in candidate_refs]
        adapter_values = scorer.score_pairs(pairs, criterion=criterion)
        for (row, candidate), adapter_logp in zip(candidate_refs, adapter_values):
            base_logp = base_cache[(str(row["prompt"]), candidate["response"])]
            difference = adapter_logp - base_logp
            candidate["adapter_log_probability"] = adapter_logp
            candidate["base_log_probability"] = base_logp
            candidate["log_probability_difference"] = difference
            candidate["pl_utility"] = beta * difference

        for row in rows:
            ranked, flags = rank_candidates(row["candidates"])
            record = {
                "dataset_row_index": row["dataset_row_index"],
                "source_index": row.get("source_index"),
                "criterion": criterion,
                "prompt": str(row["prompt"]),
                **flags,
            }
            candidates_by_label = {candidate["candidate_label"]: candidate for candidate in row["candidates"]}
            for candidate_number in range(1, 5):
                for response_type in ("original", "augmented"):
                    label = f"{response_type}_{candidate_number}"
                    candidate = candidates_by_label[label]
                    record[f"{label}_response"] = candidate["response"]
                    record[f"{label}_pl_utility"] = candidate["pl_utility"]
                    record[f"{label}_rank"] = candidate["predicted_rank"]
            ranking_records.append(record)

            for candidate in row["candidates"]:
                response_records.append(
                    {
                        "dataset_row_index": row["dataset_row_index"],
                        "source_index": row.get("source_index"),
                        "criterion": criterion,
                        "prompt": str(row["prompt"]),
                        "candidate_label": candidate["candidate_label"],
                        "response_type": candidate["response_type"],
                        "response": candidate["response"],
                        "dataset_position": candidate["dataset_position"],
                        "predicted_rank": candidate["predicted_rank"],
                        "adapter_log_probability": candidate["adapter_log_probability"],
                        "base_log_probability": candidate["base_log_probability"],
                        "log_probability_difference": candidate["log_probability_difference"],
                        "pl_utility": candidate["pl_utility"],
                    }
                )
    return ranking_records, response_records


def build_statistics(
    ranking_records: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ranking_records:
        grouped[str(record["criterion"])].append(record)
    by_criterion = {criterion: summarize_rankings(rows) for criterion, rows in sorted(grouped.items())}
    overall_macro = macro_average_summaries(by_criterion)
    return {
        "metadata": metadata,
        "overall_micro": summarize_rankings(ranking_records),
        "overall_macro": overall_macro,
        "overall_macro_by_criterion": overall_macro,
        "per_criterion": by_criterion,
        "by_criterion": by_criterion,
    }


def prepare_output_dir(path: str, overwrite: bool) -> Path:
    output_dir = Path(path)
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is non-empty: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.split != "train":
        raise ValueError("This evaluator only supports --split train because only that split was augmented.")
    if args.batch_size < 1:
        raise ValueError("`batch_size` must be at least 1.")
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("`max_rows` must be at least 1 when provided.")
    if not math.isfinite(args.beta) or args.beta <= 0:
        raise ValueError("`beta` must be a positive finite value.")

    started_at = datetime.now(timezone.utc)
    wall_start = time.perf_counter()
    checkpoint_map = parse_criterion_checkpoints(args.criterion_checkpoint)
    base_model_name, checkpoint_map = validate_checkpoints(checkpoint_map)
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    dataset_dict = load_dataset_dict(args.dataset_name, args.dataset_config)
    if args.split not in dataset_dict:
        raise ValueError(f"Split {args.split!r} not found; available splits: {list(dataset_dict)}")
    rows_by_criterion = prepare_rows(
        dataset_dict[args.split],
        checkpoint_map,
        max_rows=args.max_rows,
        expected_num_original=args.expected_num_original,
        expected_num_augmented=args.expected_num_augmented,
    )
    scorer = CriterionAdapterScorer(
        base_model_name=base_model_name,
        checkpoint_map=checkpoint_map,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        disable_tqdm=args.disable_tqdm,
    )
    ranking_records, response_records = score_and_rank_rows(rows_by_criterion, scorer, beta=args.beta)
    elapsed = time.perf_counter() - wall_start
    metadata = {
        "dataset_name": str(Path(args.dataset_name).resolve()) if os.path.exists(args.dataset_name) else args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "base_model_name_or_path": base_model_name,
        "criterion_checkpoints": checkpoint_map,
        "beta": args.beta,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "expected_num_original": args.expected_num_original,
        "expected_num_augmented": args.expected_num_augmented,
        "batch_size": args.batch_size,
        "max_rows_per_criterion": args.max_rows,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": elapsed,
        "cuda_available": torch.cuda.is_available(),
        "num_ranking_rows": len(ranking_records),
        "num_response_score_rows": len(response_records),
    }
    statistics = build_statistics(ranking_records, metadata=metadata)
    write_csv(output_dir / "rankings.csv", ranking_records)
    write_csv(output_dir / "response_scores.csv", response_records)
    with (output_dir / "statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    LOGGER.info("Wrote %d rankings and %d response scores to %s", len(ranking_records), len(response_records), output_dir)
    return statistics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    evaluate(parse_args())


if __name__ == "__main__":
    main()
