"""
RLCHF Trainer: Reinforcement Learning from Collective Human Feedback with latent groups.
Implements the methodology from the Chinese document.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from torch.optim import AdamW
from typing import Optional, Union, Dict, List, Tuple, Callable, Iterable, Any, Mapping
import numpy as np
import random
import typing
from transformers import DataCollatorForLanguageModeling
from torch.optim import Adam
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import datasets
from copy import deepcopy
import tqdm

from trainers.rlchf_config import RLCHFConfig
from trainers.rlchf_model import AutoModelForCausalLMWithGroupConditionalScoring
from trainers.utils import (
    RunningMoments,
    AdaptiveKLController,
    FixedKLController,
    logprobs_from_logits,
    masked_mean,
    flatten_dict,
    set_seed,
    is_torch_greater_2_0,
    create_reference_model,
)

PreTrainedModelWrapper = typing.Union[nn.Module, nn.DataParallel]


class RLCHFTrainer:
    """
    RLCHF Trainer implementing:
    1. Latent group variable z ∈ {1, ..., K}
    2. Group-conditional scoring model s_θ(x, y, z)
    3. Mixture preference fitting
    4. Collective aggregation (weighted average or log-sum-exp)
    5. DPO loss integration
    """
    
    def __init__(
        self,
        config: RLCHFConfig = None,
        model: PreTrainedModelWrapper = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        tokenizer: PreTrainedTokenizerBase = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[typing.Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        additional_config_kwargs: Optional[dict] = None,
    ):
        self.config = config
        set_seed(self.config.seed)
        
        # Initialize Accelerator
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            deepspeed_plugin=None,
            **config.accelerator_kwargs,
        )
        
        self.model = model
        if ref_model is None:
            self.ref_model = create_reference_model(self.model, num_shared_layers=num_shared_layers)
        else:
            self.ref_model = ref_model
        self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        if self.is_encoder_decoder:
            raise ValueError("RLCHF does not support encoder-decoder models.")
        
        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        config.is_encoder_decoder = self.is_encoder_decoder
        config.is_peft_model = self.is_peft_model
        
        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        current_config = dict(trl_rlchf_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict()
        current_config.update(flatten_dict(additional_config_kwargs or {}))
        
        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=current_config,
            init_kwargs=config.tracker_kwargs,
        )
        self.tokenizer = tokenizer
        
        # Initialize optimizer and data collator
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        if optimizer is None:
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate,
            )
        else:
            self.optimizer = optimizer
        
        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is not None:
            lr_scheduler_class = (
                torch.optim.lr_scheduler._LRScheduler
                if not is_torch_greater_2_0()
                else torch.optim.lr_scheduler.LRScheduler
            )
            if not isinstance(self.lr_scheduler, lr_scheduler_class):
                raise ValueError(
                    "lr_scheduler must be a torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.LRScheduler (for torch >= 2.0)"
                )
        
        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(self.config.init_kl_coef, self.config.target, self.config.horizon)
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)
        
        # Prepare models with accelerator
        (
            self.model,
            self.optimizer,
            self.data_collator,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.data_collator,
            self.lr_scheduler,
        )
        
        self.ref_model = self.accelerator.prepare(self.ref_model)
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"
        
        self.current_step = 0
        self.current_device = self.accelerator.device
        self.running = RunningMoments(self.accelerator)
    
    def _compute_group_scores(
        self,
        model_output: tuple,
        attention_mask: torch.Tensor,
        return_gating: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract group-conditional scores from model output.
        
        Returns:
            group_scores: (batch_size, seq_len, num_groups) - scores for each group
            gating_weights: (batch_size, num_groups) - mixing weights π_z(x)
        """
        # Model output: (lm_logits, loss, group_scores, gating_weights)
        if len(model_output) >= 3:
            group_scores = model_output[2]  # (batch_size, seq_len, num_groups)
            gating_weights = model_output[3] if len(model_output) >= 4 and return_gating else None
        else:
            raise ValueError("Model output should contain group scores")
        
        return group_scores, gating_weights
    
    def _get_sequence_group_scores(
        self,
        group_scores: torch.Tensor,  # (batch_size, seq_len, num_groups)
        attention_mask: torch.Tensor,  # (batch_size, seq_len)
        query_length: int,
    ) -> torch.Tensor:
        """
        Get sequence-level group scores by masking query tokens and aggregating.
        
        Returns:
            seq_scores: (batch_size, num_groups) - aggregated scores for each group
        """
        # Mask out query tokens (first query_length tokens)
        mask = attention_mask.clone()
        if query_length > 1:
            mask[:, :query_length - 1] = 0
        
        # Mean pool over response tokens only
        # Expand mask for broadcasting: (batch_size, seq_len, 1)
        mask_expanded = mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        
        # Masked mean
        masked_scores = group_scores * mask_expanded  # (batch_size, seq_len, num_groups)
        sum_scores = masked_scores.sum(dim=1)  # (batch_size, num_groups)
        sum_mask = mask_expanded.sum(dim=1)  # (batch_size, 1)
        seq_scores = sum_scores / (sum_mask + 1e-8)  # (batch_size, num_groups)
        
        return seq_scores
    
    def _compute_collective_logit_difference(
        self,
        delta_z: torch.Tensor,  # (batch_size, num_groups) - logit differences per group
        gating_weights: torch.Tensor,  # (batch_size, num_groups) - mixing weights π_z(x)
    ) -> torch.Tensor:
        """
        Compute collective logit difference Δ_col using aggregation method.
        
        Args:
            delta_z: (batch_size, num_groups) - Δ_z(x, y⁺, y⁻) for each group
            gating_weights: (batch_size, num_groups) - π_z(x) mixing weights
        
        Returns:
            delta_col: (batch_size,) - aggregated collective logit difference
        """
        if self.config.aggregation_type == "weighted_avg":
            # Option 1: Weighted average aggregation
            # Δ_col = Σ_z w_z * Δ_z
            if self.config.stakeholder_weights is not None:
                weights = torch.tensor(
                    self.config.stakeholder_weights,
                    device=delta_z.device,
                    dtype=delta_z.dtype
                ).unsqueeze(0)  # (1, num_groups)
                weights = weights.expand_as(gating_weights)  # (batch_size, num_groups)
            else:
                weights = gating_weights  # Use learned π_z(x)
            
            delta_col = (weights * delta_z).sum(dim=1)  # (batch_size,)
            
        elif self.config.aggregation_type == "log_sum_exp":
            # Option 2: Log-sum-exp aggregation (soft aggregation)
            # Δ_col = τ * log(Σ_z w_z * exp(Δ_z / τ))
            if self.config.stakeholder_weights is not None:
                weights = torch.tensor(
                    self.config.stakeholder_weights,
                    device=delta_z.device,
                    dtype=delta_z.dtype
                ).unsqueeze(0)
                weights = weights.expand_as(gating_weights)
            else:
                weights = gating_weights
            
            tau = self.config.aggregation_temperature
            # Compute weighted sum inside log
            exp_term = weights * torch.exp(delta_z / tau)  # (batch_size, num_groups)
            sum_term = exp_term.sum(dim=1)  # (batch_size,)
            delta_col = tau * torch.log(sum_term + 1e-8)  # (batch_size,)
        else:
            raise ValueError(f"Unknown aggregation_type: {self.config.aggregation_type}")
        
        return delta_col
    
    def _compute_entropy_regularization(
        self,
        gating_weights: torch.Tensor,  # (batch_size, num_groups)
    ) -> torch.Tensor:
        """
        Compute entropy regularization to prevent collapse.
        
        Returns:
            entropy_reg: scalar - entropy regularization term
        """
        # Entropy: -Σ_z π_z * log(π_z)
        log_probs = torch.log(gating_weights + 1e-8)
        entropy = -(gating_weights * log_probs).sum(dim=1)  # (batch_size,)
        entropy_reg = -entropy.mean()  # Negative entropy (we want to maximize entropy)
        return entropy_reg
    
    def _step(
        self,
        queries: torch.LongTensor,  # (batch_size, Lq)
        responses_w: torch.LongTensor,  # (batch_size, Lw) - chosen
        responses_l: torch.LongTensor,  # (batch_size, Ll) - rejected
        preference_mask: Optional[torch.BoolTensor] = None,  # (batch_size,)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Single training step for RLCHF.
        
        Implements:
        1. Compute group-conditional scores s_θ(x, y⁺, z) and s_θ(x, y⁻, z)
        2. Compute group preference probabilities p_z = σ(s_θ(x, y⁺, z) - s_θ(x, y⁻, z))
        3. Compute mixture preference p(y⁺ > y⁻ | x) = Σ_z π_z(x) * p_z
        4. Compute collective aggregation Δ_col
        5. Compute DPO loss with Δ_col
        """
        pad_id = self.tokenizer.pad_token_id
        device = queries.device
        
        # Concatenate queries and responses
        input_ids_w = torch.cat((queries, responses_w), dim=1)  # [B, Lq+Lw]
        input_ids_l = torch.cat((queries, responses_l), dim=1)  # [B, Lq+Ll]
        q_len = queries.size(1)
        
        def process_input_ids(input_ids, return_group_scores=True):
            attention_mask = (input_ids != pad_id).long()
            input_data = {"input_ids": input_ids, "attention_mask": attention_mask}
            
            # Forward pass through model with group scoring
            model_output = self.model(
                **input_data,
                return_group_scores=return_group_scores,
                return_gating_weights=return_group_scores,
            )
            
            # Extract outputs
            lm_logits = model_output[0]
            loss = model_output[1] if len(model_output) > 1 else None
            
            if return_group_scores:
                group_scores, gating_weights = self._compute_group_scores(model_output, attention_mask, return_gating=True)
                # Shift group_scores to align with logprobs (remove first token, same as logprobs)
                # group_scores: (batch_size, seq_len, num_groups) -> (batch_size, seq_len-1, num_groups)
                group_scores = group_scores[:, 1:, :]
            else:
                group_scores, gating_weights = None, None
            
            # Compute logprobs for reference model (standard DPO)
            with torch.no_grad():
                ref_output = self.ref_model(**input_data)
                if isinstance(ref_output, tuple):
                    ref_logits = ref_output[0]
                else:
                    ref_logits = ref_output.logits
                old_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], input_ids[:, 1:])
            
            # Compute logprobs for policy model
            logprobs = logprobs_from_logits(lm_logits[:, :-1, :], input_ids[:, 1:])
            attn_mask_shifted = attention_mask[:, 1:]
            
            return logprobs, old_logprobs, attn_mask_shifted, group_scores, gating_weights
        
        # Process chosen and rejected responses
        logprobs_w, old_logprobs_w, attn_shift_w, group_scores_w, gating_weights_w = process_input_ids(input_ids_w)
        logprobs_l, old_logprobs_l, attn_shift_l, group_scores_l, gating_weights_l = process_input_ids(input_ids_l)
        
        # Mask query tokens
        mask_w = attn_shift_w.clone()
        mask_l = attn_shift_l.clone()
        if q_len > 1:
            # Since we've shifted by 1 token, query_length in shifted space is q_len - 1
            mask_w[:, :q_len - 1] = 0
            mask_l[:, :q_len - 1] = 0
        
        # Compute sequence-level logprobs (standard DPO)
        seq_logprob_w = (logprobs_w * mask_w).sum(dim=1)  # [B]
        seq_logprob_l = (logprobs_l * mask_l).sum(dim=1)  # [B]
        seq_old_logprob_w = (old_logprobs_w * mask_w).sum(dim=1)  # [B]
        seq_old_logprob_l = (old_logprobs_l * mask_l).sum(dim=1)  # [B]
        
        # Get sequence-level group scores
        # Note: group_scores and attn_shift are already aligned (both shifted by 1)
        # query_length needs to be adjusted: original q_len -> shifted space q_len - 1
        seq_scores_w = self._get_sequence_group_scores(group_scores_w, attn_shift_w, max(1, q_len - 1))  # [B, K]
        seq_scores_l = self._get_sequence_group_scores(group_scores_l, attn_shift_l, max(1, q_len - 1))  # [B, K]
        
        # Compute group logit differences: Δ_z(x, y⁺, y⁻) = s_θ(x, y⁺, z) - s_θ(x, y⁻, z)
        delta_z = seq_scores_w - seq_scores_l  # [B, K]
        
        # Use gating weights from query (use chosen side, should be similar)
        gating_weights = gating_weights_w  # [B, K]
        
        # Compute collective logit difference
        delta_col = self._compute_collective_logit_difference(delta_z, gating_weights)  # [B]
        
        # Compute reference logit difference for DPO
        ref_logratios = seq_old_logprob_w - seq_old_logprob_l  # [B]
        ref_delta_col = ref_logratios  # For reference, use standard logratios
        
        # Compute DPO loss: L_DPO = -log σ(β * (Δ_col(θ) - Δ_ref))
        beta = self.config.temperature  # DPO temperature coefficient
        dpo_logits = beta * (delta_col - ref_delta_col)  # [B]
        
        if self.config.ipo_loss:
            dpo_loss_vec = (dpo_logits - 1.0 / (2 * beta)) ** 2
        else:
            dpo_loss_vec = -F.logsigmoid(dpo_logits)  # [B]
        
        # Preference mask
        if preference_mask is not None:
            sample_mask = preference_mask.to(device).float()
        else:
            sample_mask = torch.ones(dpo_loss_vec.size(0), device=device)
        
        denom = sample_mask.sum()
        if denom.item() == 0:
            dpo_loss = dpo_loss_vec.mean()
        else:
            dpo_loss = (dpo_loss_vec * sample_mask).sum() / denom
        
        # Add entropy regularization to prevent collapse
        entropy_reg = self._compute_entropy_regularization(gating_weights)  # scalar
        entropy_loss = self.config.entropy_reg_coef * entropy_reg
        
        # Total loss
        total_loss = dpo_loss + entropy_loss
        
        # Compute statistics
        rewards_chosen = self.config.temperature * (seq_logprob_w - seq_old_logprob_w).detach()
        rewards_rejected = self.config.temperature * (seq_logprob_l - seq_old_logprob_l).detach()
        reward_margin = rewards_chosen - rewards_rejected
        
        stats = dict(
            loss=dict(
                dpo_loss=dpo_loss.detach(),
                entropy_loss=entropy_loss.detach(),
                total_loss=total_loss.detach(),
            ),
            policy=dict(
                rewards_chosen=rewards_chosen.mean().detach(),
                rewards_rejected=rewards_rejected.mean().detach(),
                reward_margin=reward_margin.mean().detach(),
                logprobs_w=seq_logprob_w.mean().detach(),
                logprobs_l=seq_logprob_l.mean().detach(),
                dpo_logit_mean=dpo_logits.mean().detach(),
                delta_col_mean=delta_col.mean().detach(),
                classifier_accuracy=(reward_margin > 0).float().mean().detach(),
            ),
            gating=dict(
                entropy=entropy_reg.detach(),
                gating_weights_mean=gating_weights.mean(dim=0).detach().cpu().tolist(),
            ),
        )
        
        return total_loss, flatten_dict(stats)
    
    def step(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
        preference_mask: Optional[torch.BoolTensor] = None,
    ):
        """Training step with mini-batching."""
        assert queries.ndim == 2 and responses_w.ndim == 2 and responses_l.ndim == 2
        self.model.train()
        bs = self.config.batch_size
        sub_bs = self.config.mini_batch_size
        assert bs % sub_bs == 0
        
        all_stats = []
        
        for i in range(0, bs, sub_bs):
            queries_ = queries[i : i + sub_bs]
            responses_w_ = responses_w[i : i + sub_bs]
            responses_l_ = responses_l[i : i + sub_bs]
            preference_mask_ = preference_mask[i : i + sub_bs] if preference_mask is not None else None
            
            loss, stats = self._step(
                queries=queries_,
                responses_w=responses_w_,
                responses_l=responses_l_,
                preference_mask=preference_mask_,
            )
            
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.current_step += 1
            
            all_stats.append(stats)
        
        # Average stats across mini-batches
        if len(all_stats) > 1:
            avg_stats = {}
            for key in all_stats[0].keys():
                if isinstance(all_stats[0][key], (int, float)):
                    avg_stats[key] = np.mean([s[key] for s in all_stats])
                elif isinstance(all_stats[0][key], list):
                    # For lists (like gating_weights_mean), take element-wise mean
                    avg_stats[key] = [
                        np.mean([s[key][j] for s in all_stats])
                        for j in range(len(all_stats[0][key]))
                    ]
                else:
                    avg_stats[key] = all_stats[0][key]
            return avg_stats
        else:
            return all_stats[0]
    
    @torch.no_grad()
    def evaluate(self, eval_dataloader):
        """Evaluation loop."""
        self.model.eval()
        self.ref_model.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        total_examples = 0
        
        for batch in eval_dataloader:
            preference_mask = batch.get("preference_mask", None)
            
            if isinstance(batch["query"], list) or isinstance(batch["query"], tuple):
                queries = self.tokenizer(
                    batch["query"],
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).input_ids.to(self.current_device)
                
                all_pref = batch["response_w"] + batch["response_l"]
                tokenized = self.tokenizer(
                    all_pref,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).input_ids.to(self.current_device)
                
                n_w = len(batch["response_w"])
                responses_w = tokenized[:n_w]
                responses_l = tokenized[n_w:]
                
                if preference_mask is not None:
                    preference_mask = torch.as_tensor(preference_mask, device=self.current_device)
            else:
                queries = batch["query"].to(self.current_device)
                responses_w = batch["response_w"].to(self.current_device)
                responses_l = batch["response_l"].to(self.current_device)
                if preference_mask is not None:
                    preference_mask = preference_mask.to(self.current_device)
            
            loss, stats = self._step(
                queries=queries,
                responses_w=responses_w,
                responses_l=responses_l,
                preference_mask=preference_mask,
            )
            
            batch_size = queries.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
            batch_acc = stats["policy/classifier_accuracy"].item()
            total_acc += batch_acc * batch_size
        
        if total_examples == 0:
            mean_loss = 0.0
            mean_acc = 0.0
        else:
            mean_loss = total_loss / total_examples
            mean_acc = total_acc / total_examples
        
        metrics = {
            "eval/rlchf_loss": mean_loss,
            "eval/classification_accuracy": mean_acc,
        }
        return metrics
    
    def end_of_epoch_step(self, epoch: int):
        """Performs tasks at the end of an epoch, like saving models."""
        pass  # Add model saving logic if needed
    
    def log_stats(
        self,
        stats: dict,
        batch: dict,
        rewards: List[torch.FloatTensor],
        columns_to_log: List[str] = ["query", "response"],
    ):
        """
        A function that logs all the training stats. Call it at the end of each epoch.

        Args:
            stats (dict[str, Any]):
                A dictionary of training stats.
            batch (dict[str, Any]):
                A dictionary of batch data, this contains the queries and responses.
            rewards (`List[torch.FloatTensor]`):
                A tensor of rewards.
        """
        # Log only if we are in the main process
        if self.accelerator.is_main_process:
            logs = {}

            # Log stats
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards).to(self.current_device)

            if self.config.log_with == "wandb":
                import wandb

                if any([column_to_log not in batch.keys() for column_to_log in columns_to_log]):
                    raise ValueError(f"Columns to log {columns_to_log} are not present in the batch {batch.keys()}.")

                batch_list = [batch[column_to_log] for column_to_log in columns_to_log]

                table_rows = [list(r) for r in zip(*batch_list, rewards.cpu().tolist())]
                logs.update({"game_log": wandb.Table(columns=[*columns_to_log, "reward"], rows=table_rows)})

            logs.update(stats)

            # manually cast in fp32 for bf16 torch tensors
            for k, v in logs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    logs[k] = v.float()

            self.accelerator.log(
                logs,
                step=self.current_step if self.config.log_with == "tensorboard" else None,
            )

