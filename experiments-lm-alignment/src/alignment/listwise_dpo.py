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

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import re
from typing import Any, Literal, Union

from accelerate import PartialState
from datasets import Dataset, IterableDataset
from transformers import BaseImageProcessor, FeatureExtractionMixin, PreTrainedTokenizerBase, ProcessorMixin
import torch
import torch.nn.functional as F
from trl import DPOTrainer


@dataclass
class ListwiseDPODataCollator:
    tokenizer: Any
    max_length: int
    max_prompt_length: int

    def _truncate_prompt(self, prompt_ids: list[int]) -> list[int]:
        if len(prompt_ids) <= self.max_prompt_length:
            return prompt_ids
        return prompt_ids[-self.max_prompt_length :]

    def _build_sequence(self, prompt: str, response: str) -> tuple[list[int], list[int], list[int]]:
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

        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + response_ids
        return input_ids, attention_mask, labels

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_candidate_mask = []
        batch_preference_dimensions = []

        max_candidates = max(len(feature["responses"]) for feature in features)

        for feature in features:
            prompt = feature["prompt"]
            responses = feature["responses"]
            batch_preference_dimensions.append(feature.get("preference_dimension", "unknown"))

            item_input_ids = []
            item_attention_mask = []
            item_labels = []
            item_candidate_mask = [1] * len(responses) + [0] * (max_candidates - len(responses))

            for response in responses:
                input_ids, attention_mask, labels = self._build_sequence(prompt, response)
                item_input_ids.append(input_ids)
                item_attention_mask.append(attention_mask)
                item_labels.append(labels)

            for _ in range(max_candidates - len(responses)):
                item_input_ids.append([])
                item_attention_mask.append([])
                item_labels.append([])

            batch_input_ids.append(item_input_ids)
            batch_attention_mask.append(item_attention_mask)
            batch_labels.append(item_labels)
            batch_candidate_mask.append(item_candidate_mask)

        max_seq_len = max(len(seq) for item in batch_input_ids for seq in item)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id for listwise collation.")

        for i in range(len(batch_input_ids)):
            for j in range(len(batch_input_ids[i])):
                seq_len = len(batch_input_ids[i][j])
                pad_len = max_seq_len - seq_len
                batch_input_ids[i][j] = batch_input_ids[i][j] + [pad_id] * pad_len
                batch_attention_mask[i][j] = batch_attention_mask[i][j] + [0] * pad_len
                batch_labels[i][j] = batch_labels[i][j] + [-100] * pad_len

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "candidate_mask": torch.tensor(batch_candidate_mask, dtype=torch.bool),
            "preference_dimension": batch_preference_dimensions,
        }


class ListwiseDPOTrainer(DPOTrainer):
    def __init__(self, *args, listwise_beta: float | None = None, **kwargs):
        self.listwise_beta_override = listwise_beta
        if kwargs.get("data_collator") is None:
            processing_class = kwargs.get("processing_class") or kwargs.get("tokenizer")
            if processing_class is None:
                raise ValueError("A tokenizer/processing_class is required for listwise data collation.")
            max_length = kwargs.get("max_length")
            if max_length is None and kwargs.get("args") is not None:
                max_length = getattr(kwargs["args"], "max_length", None)
            if max_length is None:
                max_length = 1024
            max_prompt_length = kwargs.get("max_prompt_length")
            if max_prompt_length is None and kwargs.get("args") is not None:
                max_prompt_length = getattr(kwargs["args"], "max_prompt_length", None)
            if max_prompt_length is None:
                max_prompt_length = 512
            kwargs["data_collator"] = ListwiseDPODataCollator(
                tokenizer=processing_class,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
            )
        super().__init__(*args, **kwargs)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "input_ids",
                "attention_mask",
                "labels",
                "candidate_mask",
            ]

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: Any,
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        """
        For listwise datasets, skip pairwise-specific tokenization and column removal.
        Listwise datasets already have prompt/responses structure; collator handles sequencing.
        """
        from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt
        
        with PartialState().main_process_first():
            if isinstance(dataset, Dataset):
                map_kwargs = {"num_proc": args.dataset_num_proc, "writer_batch_size": 10}
            else:
                map_kwargs = {}

            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={"tokenizer": processing_class, "tools": args.tools},
                **map_kwargs,
            )

        return dataset

    @staticmethod
    def _sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid = shift_labels != -100

        safe_labels = shift_labels.masked_fill(~valid, 0)
        token_logps = F.log_softmax(shift_logits, dim=-1).gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logps = token_logps * valid
        return token_logps.sum(dim=-1)

    def _forward_ref(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.ref_model is None:
                with self.model.disable_adapter():
                    ref_logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            else:
                ref_logits = self.ref_model(input_ids=input_ids, attention_mask=attention_mask).logits
        return self._sequence_logps(ref_logits, labels)

    @staticmethod
    def _pl_negative_log_likelihood(utilities: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        # Mask padded candidates and compute suffix logsumexp in parallel.
        masked_utilities = utilities.masked_fill(~candidate_mask, float("-inf"))
        suffix_lse = torch.flip(torch.logcumsumexp(torch.flip(masked_utilities, dims=[1]), dim=1), dims=[1])

        # Per-position PL term: -(u_k - logsumexp(u_k, ..., u_m)), only on valid candidates.
        diff = torch.where(candidate_mask, masked_utilities - suffix_lse, torch.zeros_like(utilities))
        row_losses = (-diff).sum(dim=1)

        valid_rows = candidate_mask.sum(dim=1) >= 2
        if not torch.any(valid_rows):
            return utilities.new_zeros(())
        return row_losses[valid_rows].mean()

    @staticmethod
    def _metric_dimension_key(dimension: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(dimension).strip()).strip("_")
        return normalized or "unknown"

    @staticmethod
    def _listwise_metrics(utilities: torch.Tensor, candidate_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        pred_order = torch.argsort(utilities, dim=1, descending=True)
        top1_acc = (pred_order[:, 0] == 0).float().mean()

        num_candidates = utilities.shape[1]
        upper_tri = torch.triu(
            torch.ones((num_candidates, num_candidates), dtype=torch.bool, device=utilities.device),
            diagonal=1,
        )
        valid_pairs = candidate_mask.unsqueeze(2) & candidate_mask.unsqueeze(1) & upper_tri.unsqueeze(0)
        pairwise_correct = (utilities.unsqueeze(2) > utilities.unsqueeze(1)) & valid_pairs

        total_pairs = valid_pairs.sum(dim=(1, 2))
        correct_pairs = pairwise_correct.sum(dim=(1, 2))
        row_pairwise_acc = correct_pairs.float() / total_pairs.clamp_min(1).float()
        valid_rows = total_pairs > 0
        pairwise_acc = row_pairwise_acc[valid_rows].mean() if torch.any(valid_rows) else utilities.new_zeros(())

        utility_first = torch.where(
            candidate_mask[:, 0],
            utilities[:, 0],
            torch.zeros_like(utilities[:, 0]),
        ).mean()

        valid_counts = candidate_mask.sum(dim=1)
        valid_utility_rows = valid_counts > 0
        if torch.any(valid_utility_rows):
            last_indices = valid_counts[valid_utility_rows] - 1
            row_indices = torch.arange(last_indices.shape[0], device=utilities.device)
            utility_last = utilities[valid_utility_rows][row_indices, last_indices].mean()
        else:
            utility_last = utilities.new_zeros(())

        return {
            "listwise/top1_acc": top1_acc,
            "listwise/pairwise_acc": pairwise_acc,
            "listwise/utility_first": utility_first,
            "listwise/utility_last": utility_last,
            "listwise/utility_mean": utilities.mean(),
        }

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Any],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        candidate_mask = batch["candidate_mask"]
        preference_dimensions = batch.get("preference_dimension")

        batch_size, num_candidates, seq_len = input_ids.shape
        flat_input_ids = input_ids.view(batch_size * num_candidates, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_candidates, seq_len)
        flat_labels = labels.view(batch_size * num_candidates, seq_len)

        policy_logits = model(input_ids=flat_input_ids, attention_mask=flat_attention_mask).logits
        policy_logps = self._sequence_logps(policy_logits, flat_labels).view(batch_size, num_candidates)
        ref_logps = self._forward_ref(flat_input_ids, flat_attention_mask, flat_labels).view(batch_size, num_candidates)

        beta = self.listwise_beta_override if self.listwise_beta_override is not None else self.beta
        utilities = beta * (policy_logps - ref_logps)
        loss = self._pl_negative_log_likelihood(utilities, candidate_mask)

        metric_tensors = self._listwise_metrics(utilities, candidate_mask)

        if preference_dimensions is not None:
            dim_to_indices: dict[str, list[int]] = {}
            for idx, dimension in enumerate(preference_dimensions):
                dim_key = self._metric_dimension_key(dimension)
                dim_to_indices.setdefault(dim_key, []).append(idx)

            for dim_key, indices in dim_to_indices.items():
                dim_index_tensor = torch.tensor(indices, device=utilities.device, dtype=torch.long)
                dim_metrics = self._listwise_metrics(
                    utilities.index_select(0, dim_index_tensor),
                    candidate_mask.index_select(0, dim_index_tensor),
                )
                for metric_name, metric_value in dim_metrics.items():
                    suffix = metric_name.removeprefix("listwise/")
                    metric_tensors[f"listwise/by_dimension/{dim_key}/{suffix}"] = metric_value

        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {
            f"{prefix}{name}": self.accelerator.gather_for_metrics(value.detach()).mean().item()
            for name, value in metric_tensors.items()
        }

        return loss, metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        compute_loss_context_manager = (
            torch.autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        loss = loss.to(self.args.device)
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics

        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        del ignore_keys
        prediction_context_manager = (
            torch.autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = torch.tensor(
            [
                metrics["eval_listwise/utility_first"],
                metrics["eval_listwise/utility_last"],
            ],
            device=self.accelerator.device,
        )
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)
