from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from datasets import Dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback

from .configs import DPOConfig
from .listwise_dpo import ListwiseDPODataCollator
from .ranking_eval import RankingMetricAccumulator


PAIRWISE_STRATEGIES = {"all_pairs", "extreme"}


@dataclass
class GPOConfig(DPOConfig):
    """Training configuration for a General Preference embedding ranker."""

    gpo_tau: float = field(default=0.1, metadata={"help": "Temperature for the GP pairwise loss."})
    gpo_value_head_dim: int = field(
        default=4,
        metadata={"help": "Even-dimensional GP value head size."},
    )
    gpo_pairwise_strategy: str = field(
        default="all_pairs",
        metadata={"help": "How to convert listwise rankings to pairs: 'all_pairs' or 'extreme'."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.gpo_tau <= 0:
            raise ValueError("`gpo_tau` must be positive.")
        if self.gpo_value_head_dim < 2 or self.gpo_value_head_dim % 2 != 0:
            raise ValueError("`gpo_value_head_dim` must be an even integer >= 2.")
        if self.gpo_pairwise_strategy not in PAIRWISE_STRATEGIES:
            raise ValueError("`gpo_pairwise_strategy` must be either 'all_pairs' or 'extreme'.")


def _metadata_from_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = {}
    for key in (
        "source_index",
        "source_dataset",
        "persona_id",
        "persona_text",
        "instruction",
        "ranked_prefix_length",
        "preference_dimension",
    ):
        if key in row:
            metadata[key] = row[key]
    return metadata


def listwise_to_gpo_pairwise_dataset(dataset: Dataset, strategy: str = "all_pairs") -> Dataset:
    """Convert ranked response rows into chosen/rejected pairs for GP training."""
    if strategy not in PAIRWISE_STRATEGIES:
        raise ValueError(f"`strategy` must be one of {sorted(PAIRWISE_STRATEGIES)}.")

    rows: list[dict[str, Any]] = []
    for row in dataset:
        responses = row.get("responses")
        scores = row.get("scores")
        if not isinstance(responses, list) or not isinstance(scores, list):
            continue
        n = min(len(responses), len(scores))
        if n < 2:
            continue

        ranked_prefix_length = row.get("ranked_prefix_length")
        if ranked_prefix_length is not None:
            ranked_prefix_length = max(0, min(int(ranked_prefix_length), n))
            pair_indices = [
                (chosen_idx, rejected_idx)
                for chosen_idx in range(ranked_prefix_length)
                for rejected_idx in range(chosen_idx + 1, n)
            ]
        elif strategy == "extreme":
            pair_indices = [(0, n - 1)]
        else:
            pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]

        metadata = _metadata_from_row(row)
        for chosen_idx, rejected_idx in pair_indices:
            rows.append(
                {
                    "prompt": str(row["prompt"]),
                    "chosen": str(responses[chosen_idx]),
                    "rejected": str(responses[rejected_idx]),
                    "chosen_score": float(scores[chosen_idx]),
                    "rejected_score": float(scores[rejected_idx]),
                    "chosen_rank": int(chosen_idx),
                    "rejected_rank": int(rejected_idx),
                    **metadata,
                }
            )

    if not rows:
        raise ValueError("No GP pairwise rows could be built from the listwise dataset.")
    return Dataset.from_list(rows)


@dataclass
class GPOPairwiseDataCollator:
    tokenizer: Any
    max_length: int
    max_prompt_length: int

    def _truncate_prompt(self, prompt_ids: list[int]) -> list[int]:
        if len(prompt_ids) <= self.max_prompt_length:
            return prompt_ids
        return prompt_ids[-self.max_prompt_length :]

    def _build_sequence(self, prompt: str, response: str) -> tuple[list[int], list[int]]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_ids = self._truncate_prompt(prompt_ids)
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
            response_ids = response_ids + [eos_id]

        input_ids = prompt_ids + response_ids
        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            original_prompt_len = len(prompt_ids)
            if overflow < original_prompt_len:
                prompt_ids = prompt_ids[overflow:]
            else:
                prompt_ids = []
                response_ids = response_ids[overflow - original_prompt_len :]
            input_ids = prompt_ids + response_ids
        return input_ids, [1] * len(input_ids)

    @staticmethod
    def _left_pad(sequences: list[list[int]], value: int) -> torch.Tensor:
        max_len = max(len(seq) for seq in sequences)
        padded = [[value] * (max_len - len(seq)) + seq for seq in sequences]
        return torch.tensor(padded, dtype=torch.long)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        chosen_ids = []
        chosen_masks = []
        rejected_ids = []
        rejected_masks = []
        dimensions = []

        for feature in features:
            chosen_input_ids, chosen_attention_mask = self._build_sequence(feature["prompt"], feature["chosen"])
            rejected_input_ids, rejected_attention_mask = self._build_sequence(feature["prompt"], feature["rejected"])
            chosen_ids.append(chosen_input_ids)
            chosen_masks.append(chosen_attention_mask)
            rejected_ids.append(rejected_input_ids)
            rejected_masks.append(rejected_attention_mask)
            dimensions.append(str(feature.get("preference_dimension", "unknown")))

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")

        return {
            "chosen_input_ids": self._left_pad(chosen_ids, pad_id),
            "chosen_attention_mask": self._left_pad(chosen_masks, 0),
            "rejected_input_ids": self._left_pad(rejected_ids, pad_id),
            "rejected_attention_mask": self._left_pad(rejected_masks, 0),
            "preference_dimension": dimensions,
        }


def gp_pairwise_scores(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    *,
    value_head_dim: Optional[int] = None,
) -> torch.Tensor:
    """Return GP preference scores where positive means chosen beats rejected."""
    if chosen_rewards.shape != rejected_rewards.shape:
        raise ValueError("chosen and rejected reward tensors must have the same shape.")
    if chosen_rewards.ndim != 2:
        raise ValueError("reward tensors must have shape [batch, dim].")

    dim = value_head_dim or chosen_rewards.shape[-1]
    if dim != chosen_rewards.shape[-1]:
        raise ValueError("`value_head_dim` must match reward tensor width.")
    if dim < 2 or dim % 2 != 0:
        raise ValueError("GP reward dimension must be an even integer >= 2.")

    if dim == 2:
        return chosen_rewards[:, 0] * rejected_rewards[:, 1] - chosen_rewards[:, 1] * rejected_rewards[:, 0]

    matrix = chosen_rewards.new_zeros((dim, dim))
    for idx in range(0, dim, 2):
        matrix[idx, idx + 1] = -1
        matrix[idx + 1, idx] = 1
    transformed = torch.matmul(chosen_rewards, matrix.T)
    return torch.bmm(transformed.unsqueeze(1), rejected_rewards.unsqueeze(-1)).view(chosen_rewards.shape[0])


def aggregate_gp_pairwise_utilities(pairwise_scores: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    """Aggregate antisymmetric pair scores into one ranking utility per candidate."""
    if pairwise_scores.ndim != 3:
        raise ValueError("`pairwise_scores` must have shape [batch, candidates, candidates].")
    if candidate_mask.shape != pairwise_scores.shape[:2]:
        raise ValueError("`candidate_mask` shape must match pairwise score candidates.")

    valid_pair_mask = candidate_mask.unsqueeze(1) & candidate_mask.unsqueeze(2)
    valid_pair_mask = valid_pair_mask & ~torch.eye(
        pairwise_scores.shape[1],
        dtype=torch.bool,
        device=pairwise_scores.device,
    ).unsqueeze(0)
    masked_scores = pairwise_scores.masked_fill(~valid_pair_mask, 0.0)
    denom = valid_pair_mask.sum(dim=-1).clamp_min(1)
    utilities = masked_scores.sum(dim=-1) / denom
    return utilities.masked_fill(~candidate_mask, -torch.inf)


class GPORewardTrainer(Trainer):
    """HF Trainer wrapper for the official GP reward model."""

    def __init__(self, *args, gpo_tau: float = 0.1, gpo_value_head_dim: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpo_tau = gpo_tau
        self.gpo_value_head_dim = gpo_value_head_dim

    def _concatenated_forward(
        self,
        model: torch.nn.Module,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(chosen_input_ids.shape[1], rejected_input_ids.shape[1])
        pad_id = self.processing_class.pad_token_id
        chosen_input_ids = F.pad(chosen_input_ids, (max_len - chosen_input_ids.shape[1], 0), value=pad_id)
        rejected_input_ids = F.pad(rejected_input_ids, (max_len - rejected_input_ids.shape[1], 0), value=pad_id)
        chosen_attention_mask = F.pad(chosen_attention_mask, (max_len - chosen_attention_mask.shape[1], 0), value=0)
        rejected_attention_mask = F.pad(
            rejected_attention_mask,
            (max_len - rejected_attention_mask.shape[1], 0),
            value=0,
        )

        input_ids = torch.cat((chosen_input_ids, rejected_input_ids), dim=0)
        attention_mask = torch.cat((chosen_attention_mask, rejected_attention_mask), dim=0)
        rewards, _ = model.custom_forward(input_ids=input_ids, attention_mask=attention_mask)
        return rewards[: chosen_input_ids.shape[0]], rewards[chosen_input_ids.shape[0] :]

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        del kwargs
        chosen_rewards, rejected_rewards = self._concatenated_forward(
            model,
            inputs["chosen_input_ids"],
            inputs["chosen_attention_mask"],
            inputs["rejected_input_ids"],
            inputs["rejected_attention_mask"],
        )
        scores = gp_pairwise_scores(
            chosen_rewards,
            rejected_rewards,
            value_head_dim=self.gpo_value_head_dim,
        )
        loss = -F.logsigmoid(scores / self.gpo_tau).mean()
        if return_outputs:
            return loss, {"gp_scores": scores}
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        del prediction_loss_only, ignore_keys
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None


def _score_listwise_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    value_head_dim: int,
) -> torch.Tensor:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    candidate_mask = batch["candidate_mask"]
    batch_size, num_candidates, seq_len = input_ids.shape

    flat_input_ids = input_ids.view(batch_size * num_candidates, seq_len)
    flat_attention_mask = attention_mask.view(batch_size * num_candidates, seq_len)
    rewards, _ = model.custom_forward(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
    rewards = rewards.view(batch_size, num_candidates, value_head_dim)

    pairwise_scores = rewards.new_zeros((batch_size, num_candidates, num_candidates))
    for chosen_idx in range(num_candidates):
        chosen_rewards = rewards[:, chosen_idx, :]
        for rejected_idx in range(num_candidates):
            if chosen_idx == rejected_idx:
                continue
            pairwise_scores[:, chosen_idx, rejected_idx] = gp_pairwise_scores(
                chosen_rewards,
                rewards[:, rejected_idx, :],
                value_head_dim=value_head_dim,
            )
    return aggregate_gp_pairwise_utilities(pairwise_scores, candidate_mask)


def evaluate_gpo_ranking_split(
    *,
    trainer: GPORewardTrainer,
    split_dataset: Dataset,
    tokenizer: Any,
    training_args: GPOConfig,
) -> dict[str, float]:
    collator = ListwiseDPODataCollator(
        tokenizer=tokenizer,
        max_length=getattr(training_args, "max_length", 1024),
        max_prompt_length=getattr(training_args, "max_prompt_length", 512),
    )
    dataloader = DataLoader(
        split_dataset,
        batch_size=getattr(training_args, "per_device_eval_batch_size", 8),
        shuffle=False,
        collate_fn=collator,
    )

    accumulator = RankingMetricAccumulator()
    trainer.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = trainer._prepare_inputs(batch)
            utilities = _score_listwise_batch(
                trainer.model,
                inputs,
                value_head_dim=training_args.gpo_value_head_dim,
            )
            accumulator.add_batch(
                utilities,
                inputs["candidate_mask"],
                [str(dimension) for dimension in inputs["preference_dimension"]],
                inputs.get("ranked_prefix_length"),
            )
    return accumulator.metrics("ranking")


def evaluate_gpo_ranking_splits(
    *,
    trainer: GPORewardTrainer,
    ranking_dataset: dict[str, Dataset],
    tokenizer: Any,
    script_args: Any,
    training_args: GPOConfig,
    split_names: Sequence[str] = ("train", "validation", "test"),
) -> dict[str, dict[str, float]]:
    del script_args
    results = {}
    for split_name in split_names:
        if split_name not in ranking_dataset:
            continue
        results[split_name] = evaluate_gpo_ranking_split(
            trainer=trainer,
            split_dataset=ranking_dataset[split_name],
            tokenizer=tokenizer,
            training_args=training_args,
        )
    return results


class GPORankingEvaluationCallback(TrainerCallback):
    """Log listwise ranking metrics whenever Trainer.evaluate() runs."""

    def __init__(self, ranking_dataset, tokenizer, script_args, training_args):
        self.ranking_dataset = ranking_dataset
        self.tokenizer = tokenizer
        self.script_args = script_args
        self.training_args = training_args
        self.eval_split = script_args.dataset_test_split
        self.trainer: Optional[GPORewardTrainer] = None

        max_samples = getattr(script_args, "ranking_eval_during_training_max_samples", None)
        if (
            max_samples is not None
            and max_samples > 0
            and self.eval_split in ranking_dataset
            and len(ranking_dataset[self.eval_split]) > max_samples
        ):
            ranking_dataset = dict(ranking_dataset)
            ranking_dataset[self.eval_split] = (
                ranking_dataset[self.eval_split].shuffle(seed=training_args.seed).select(range(max_samples))
            )
            self.ranking_dataset = ranking_dataset

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        del args, state, control, kwargs
        if self.trainer is None or self.eval_split not in self.ranking_dataset:
            return
        split_metrics = evaluate_gpo_ranking_split(
            trainer=self.trainer,
            split_dataset=self.ranking_dataset[self.eval_split],
            tokenizer=self.tokenizer,
            training_args=self.training_args,
        )
        logged_metrics = {f"ranking_{self.eval_split}/{key}": value for key, value in split_metrics.items()}
        best_model_aliases = {f"eval_{key}": value for key, value in logged_metrics.items()}
        if metrics is not None:
            metrics.update(logged_metrics)
            metrics.update(best_model_aliases)
