import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import logging
from typing import Optional, Union, Dict, List, Tuple, NamedTuple, Callable, Iterable, Any, Mapping
import numpy as np
import random
import typing
from trainers.dpo_config import DPOConfig
from trainers.utils import (
    RunningMoments,
    AdaptiveKLController,
    FixedKLController,
    logprobs_from_logits,
    entropy_from_logits,
    masked_mean,
    masked_mean_sum,
    flatten_dict,
    set_seed,
    is_torch_greater_2_0,
    create_reference_model,
    empty_cache,
    empty_cache_decorator
)
from trainers.model_value_head import AutoModelForCausalLMWithValueHead
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, is_deepspeed_available
import warnings
from transformers import DataCollatorForLanguageModeling
from torch.optim import Adam
import sys
import inspect
from packaging import version
import datasets
from copy import deepcopy
import tqdm

PreTrainedModelWrapper = typing.Union[nn.Module, nn.DataParallel]


class PPORLBatch(NamedTuple):
    """
    A batch of data for PPO training.
    """
    query_tensors: List[torch.LongTensor]
    response_tensors: List[torch.LongTensor]
    logprobs: List[torch.FloatTensor]
    values: List[torch.FloatTensor]
    rewards: List[torch.FloatTensor]
    attention_mask: List[torch.LongTensor]


class RLHFTrainer:
    """
    RLHF Trainer using PPO algorithm.
    
    This trainer implements the PPO algorithm for training language models with reward models.
    The training process includes:
    1. Generate responses using the policy model
    2. Compute rewards using the reward model
    3. Compute advantages using GAE (Generalized Advantage Estimation)
    4. Compute PPO loss (policy loss + value loss + KL penalty)
    5. Update the model
    """
    
    def __init__(
        self,
        config: DPOConfig = None,
        model: PreTrainedModelWrapper = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        reward_model: Optional[PreTrainedModel] = None,
        reward_model_pos_label_idx: Optional[int] = None,
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
        
        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            deepspeed_plugin=None,
            **config.accelerator_kwargs,
        )
        
        # Wrap model with value head if needed
        if not isinstance(model, AutoModelForCausalLMWithValueHead):
            # Assume model is a base model, wrap it with value head
            self.model = AutoModelForCausalLMWithValueHead(model)
        else:
            self.model = model
            
        if ref_model is None:
            self.ref_model = create_reference_model(self.model.pretrained_model, num_shared_layers=num_shared_layers)
        else:
            self.ref_model = ref_model
            
        self.reward_model = reward_model
        self.reward_model_pos_label_idx = reward_model_pos_label_idx
        if self.reward_model is not None:
            self.reward_model.eval()
            
        self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.is_encoder_decoder = hasattr(self.model.pretrained_model, "is_encoder_decoder")
        if self.is_encoder_decoder: 
            raise ValueError("RLHF does not support encoder-decoder models.")

        self.is_peft_model = getattr(self.model.pretrained_model, "is_peft_model", False)
        config.is_encoder_decoder = self.is_encoder_decoder
        config.is_peft_model = self.is_peft_model

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        current_config = dict(trl_rlhf_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict()
        current_config.update(flatten_dict(additional_config_kwargs or {}))
        
        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=current_config,
            init_kwargs=config.tracker_kwargs,
        )
        self.tokenizer = tokenizer
        
        self._signature_columns = None

        # Step 3: Initialize optimizer and data collator
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

        # Safety checkers for DS integration
        is_deepspeed_used = self.accelerator.distributed_type == "DEEPSPEED" and hasattr(
            self.accelerator.state, "deepspeed_plugin"
        )

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
        if is_deepspeed_used:
            if not self.is_peft_model and not (
                getattr(self.ref_model, "is_loaded_in_8bit", False)
                or getattr(self.ref_model, "is_loaded_in_4bit", False)
            ):
                self.ref_model = self._prepare_deepspeed(self.ref_model)
        else:
            self.ref_model = self.accelerator.prepare(self.ref_model)
            
        if self.reward_model is not None:
            self.reward_model = self.accelerator.prepare(self.reward_model)
            
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"

        self.current_step = 0

        if config.push_to_hub_if_best_kwargs:
            if "repo_id" not in config.push_to_hub_if_best_kwargs:
                raise ValueError("You have to specify repo_id in order to push the model to the hub!")
            self.push_to_hub_kwargs = config.push_to_hub_if_best_kwargs
            self.compare_step = 0
            self.highest_reward = torch.tensor(-float("inf"))

        self.current_device = self.accelerator.device
        self.running = RunningMoments(self.accelerator)
        
        # Generation parameters
        self.max_length = getattr(config, "max_length", 512)
        self.max_new_tokens = getattr(config, "max_new_tokens", None)
        
    def _prepare_deepspeed(self, model):
        # Placeholder for deepspeed preparation
        return model

    def generate(
        self,
        query_tensor: torch.Tensor,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate response given a query tensor.
        
        Args:
            query_tensor: Tensor of shape (seq_len,)
            generation_kwargs: Additional generation arguments
            
        Returns:
            response_tensor: Generated response tensor
            full_tensor: Concatenated query + response tensor
        """
        self.model.eval()
        query_tensor = query_tensor.unsqueeze(0).to(self.current_device)
        
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens or (self.max_length - query_tensor.shape[1]),
            "top_p": getattr(self.config, "top_p", 0.9),
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)
            
        with torch.no_grad():
            response = self.model.pretrained_model.generate(
                input_ids=query_tensor,
                **gen_kwargs
            )
            
        response_tensor = response.squeeze(0)[query_tensor.shape[1]:]
        full_tensor = response.squeeze(0)
        
        return response_tensor, full_tensor

    def compute_rewards(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
    ) -> List[torch.FloatTensor]:
        """
        Compute rewards for query-response pairs using the reward model.
        
        Args:
            queries: List of query tensors
            responses: List of response tensors
            
        Returns:
            List of reward tensors (scalar per response)
        """
        if self.reward_model is None:
            raise ValueError("Reward model is not provided. Cannot compute rewards.")
            
        rewards = []
        self.reward_model.eval()
        
        with torch.no_grad():
            for query, response in zip(queries, responses):
                # Ensure both query and response are on the same device before concatenation
                query = query.to(self.current_device)
                response = response.to(self.current_device)
                full_input = torch.cat([query, response]).unsqueeze(0)
                attention_mask = torch.ones_like(full_input)
                
                # Get reward from reward model
                reward_output = self.reward_model(
                    input_ids=full_input,
                    attention_mask=attention_mask,
                )
                
                if hasattr(reward_output, "logits"):
                    logits = reward_output.logits  # shape: (batch_size, num_labels)
                    
                    # Handle different reward model types
                    if logits.shape[-1] == 1:
                        # Single-output reward model (regression)
                        reward = logits.squeeze(-1).squeeze(0)
                        if reward.ndim == 0:
                            reward_value = reward.item()
                        else:
                            reward_value = reward[-1].item() if reward.ndim > 0 else reward.item()
                    elif logits.shape[-1] == 2 and self.reward_model_pos_label_idx is not None:
                        # Binary classification model (e.g., sentiment classifier)
                        # Extract positive class probability as reward
                        probs = torch.softmax(logits, dim=-1)
                        reward = probs[0, self.reward_model_pos_label_idx]
                        reward_value = reward.item()
                    elif logits.shape[-1] == 2:
                        # Binary classification without pos_label_idx specified
                        # Use the second class (index 1) as positive by default
                        probs = torch.softmax(logits, dim=-1)
                        reward = probs[0, 1]
                        reward_value = reward.item()
                    else:
                        # Multi-class or unknown format, use last logit
                        reward = logits.squeeze(0)
                        if reward.ndim == 0:
                            reward_value = reward.item()
                        else:
                            reward_value = reward[-1].item()
                elif isinstance(reward_output, torch.Tensor):
                    reward = reward_output.squeeze(-1).squeeze(0)
                    if reward.ndim == 0:
                        reward_value = reward.item()
                    else:
                        reward_value = reward[-1].item()
                else:
                    raise ValueError(f"Unexpected reward model output type: {type(reward_output)}")
                    
                rewards.append(torch.tensor(reward_value, device=self.current_device))
                
        return rewards

    def compute_advantages_and_returns(
        self,
        rewards: List[torch.FloatTensor],
        values: List[torch.FloatTensor],
        response_lengths: List[int],
        lastgaelam: Optional[float] = None,
    ) -> Tuple[List[torch.FloatTensor], List[torch.FloatTensor]]:
        """
        Compute advantages and returns using GAE (Generalized Advantage Estimation).
        
        Args:
            rewards: List of reward tensors
            values: List of value tensors (from value head)
            response_lengths: List of response lengths
            lastgaelam: Last GAE lambda value (for continuing trajectories)
            
        Returns:
            advantages: List of advantage tensors
            returns: List of return tensors
        """
        advantages = []
        returns = []
        gamma = self.config.gamma
        lam = self.config.lam
        
        lastgaelam = 0.0 if lastgaelam is None else lastgaelam
        
        for reward, value, length in zip(rewards, values, response_lengths):
            # For simplicity, we use scalar rewards and values per response
            # In practice, you might want per-token rewards/values
            if isinstance(reward, torch.Tensor):
                reward_val = reward.item() if reward.numel() == 1 else reward[-1].item()
            else:
                reward_val = reward
                
            if isinstance(value, torch.Tensor):
                if value.ndim > 0:
                    value_vals = value.cpu().numpy()
                    last_value = value_vals[-1] if len(value_vals) > 0 else 0.0
                else:
                    last_value = value.item()
            else:
                last_value = value
                
            # Simple advantage computation for scalar rewards
            # In practice, you'd compute GAE over the full trajectory
            advantage = reward_val - last_value + gamma * lastgaelam
            return_val = reward_val + gamma * last_value
            
            advantages.append(torch.tensor(advantage))
            returns.append(torch.tensor(return_val))
            
        return advantages, returns

    def _step(
        self,
        batch: PPORLBatch,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute PPO loss for a batch of data.
        
        Args:
            batch: PPORLBatch containing queries, responses, logprobs, values, rewards
            
        Returns:
            loss: Total PPO loss
            stats: Dictionary of statistics
        """
        # Prepare input tensors
        queries = [q.to(self.current_device) for q in batch.query_tensors]
        responses = [r.to(self.current_device) for r in batch.response_tensors]
        old_logprobs = [lp.to(self.current_device) for lp in batch.logprobs]
        old_values = [v.to(self.current_device) for v in batch.values]
        rewards = [r.to(self.current_device) for r in batch.rewards]
        attention_masks = [am.to(self.current_device) for am in batch.attention_mask]
        
        # Pad sequences
        max_len = max([q.shape[0] + r.shape[0] for q, r in zip(queries, responses)])
        input_ids_list = []
        attention_mask_list = []
        
        for q, r in zip(queries, responses):
            # Ensure both query and response are on the same device
            q = q.to(self.current_device)
            r = r.to(self.current_device)
            full_seq = torch.cat([q, r])
            pad_len = max_len - full_seq.shape[0]
            if pad_len > 0:
                full_seq = torch.cat([full_seq, torch.full((pad_len,), self.tokenizer.pad_token_id, device=self.current_device)])
                attn_mask = torch.cat([torch.ones(q.shape[0] + r.shape[0], device=self.current_device), 
                                      torch.zeros(pad_len, device=self.current_device)])
            else:
                attn_mask = torch.ones(full_seq.shape[0], device=self.current_device)
            input_ids_list.append(full_seq)
            attention_mask_list.append(attn_mask)
            
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        
        # Forward pass to get new logprobs and values
        logits, _, values = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Compute new logprobs for responses only
        new_logprobs_list = []
        new_values_list = []
        
        for i, (q, r, old_lp, old_val) in enumerate(zip(queries, responses, old_logprobs, old_values)):
            q_len = q.shape[0]
            r_len = r.shape[0]
            
            # Get logprobs for response tokens
            response_logits = logits[i, q_len-1:q_len-1+r_len, :]
            response_tokens = r
            response_logprobs = logprobs_from_logits(
                response_logits.unsqueeze(0),
                response_tokens.unsqueeze(0),
            ).squeeze(0)
            
            # Sum over sequence length to get sequence-level logprob
            seq_logprob = response_logprobs.sum()
            new_logprobs_list.append(seq_logprob)
            
            # Get value at the start of response
            response_value = values[i, q_len-1]  # Value at first response token
            new_values_list.append(response_value)
            
        new_logprobs = torch.stack(new_logprobs_list)
        new_values = torch.stack(new_values_list)
        old_logprobs_tensor = torch.stack([lp.sum() if lp.ndim > 0 else lp for lp in old_logprobs])
        old_values_tensor = torch.stack([v[-1] if v.ndim > 0 else v for v in old_values])
        rewards_tensor = torch.stack([r if isinstance(r, torch.Tensor) else torch.tensor(r, device=self.current_device) 
                                     for r in rewards]).to(self.current_device)
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages_and_returns(
            rewards, old_values, [r.shape[0] for r in responses]
        )
        advantages_tensor = torch.stack(advantages).to(self.current_device)
        returns_tensor = torch.stack(returns).to(self.current_device)
        
        # Normalize advantages if configured
        if self.config.whiten_rewards:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
            
        # Compute PPO loss
        # Policy loss
        ratio = torch.exp(new_logprobs - old_logprobs_tensor)
        pg_losses = -advantages_tensor * ratio
        pg_losses_clipped = -advantages_tensor * torch.clamp(
            ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange
        )
        pg_loss = torch.max(pg_losses, pg_losses_clipped).mean()
        
        # Value loss
        vpred_clipped = old_values_tensor + torch.clamp(
            new_values - old_values_tensor, -self.config.cliprange_value, self.config.cliprange_value
        )
        vf_losses1 = (new_values - returns_tensor) ** 2
        vf_losses2 = (vpred_clipped - returns_tensor) ** 2
        vf_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()
        
        # KL penalty - compute KL divergence relative to reference model
        with torch.no_grad():
            # Get reference model logprobs
            ref_logits = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(ref_logits, "logits"):
                ref_logits = ref_logits.logits
            elif isinstance(ref_logits, tuple):
                ref_logits = ref_logits[0]
            
            # Compute reference logprobs for responses only
            ref_logprobs_list = []
            for i, (q, r) in enumerate(zip(queries, responses)):
                q_len = q.shape[0]
                r_len = r.shape[0]
                
                # Get logprobs for response tokens
                response_logits = ref_logits[i, q_len-1:q_len-1+r_len, :]
                response_tokens = r
                response_logprobs = logprobs_from_logits(
                    response_logits.unsqueeze(0),
                    response_tokens.unsqueeze(0),
                ).squeeze(0)
                
                # Sum over sequence length to get sequence-level logprob
                seq_logprob = response_logprobs.sum()
                ref_logprobs_list.append(seq_logprob)
            
            ref_logprobs = torch.stack(ref_logprobs_list)
            
            # Compute KL divergence as difference between new policy and reference model
            kl = new_logprobs - ref_logprobs
            kl_penalty = (self.kl_ctl.value * kl).mean()
            
        # Total loss - add KL penalty (positive) to penalize deviation from reference
        total_loss = pg_loss + self.config.vf_coef * vf_loss + kl_penalty
        
        # Update KL controller
        if self.config.adap_kl_ctrl:
            self.kl_ctl.update(kl.mean().item(), n_steps=1)
            
        # Compute stats
        clipfrac = ((ratio - 1.0).abs() > self.config.cliprange).float().mean()
        
        stats = dict(
            loss=dict(
                total_loss=total_loss.detach(),
                pg_loss=pg_loss.detach(),
                vf_loss=vf_loss.detach(),
                kl_penalty=kl_penalty.detach(),
            ),
            policy=dict(
                clipfrac=clipfrac.detach(),
                ratio_mean=ratio.mean().detach(),
                ratio_std=ratio.std().detach(),
                kl=kl.mean().detach(),
                rewards_mean=rewards_tensor.mean().detach(),
                advantages_mean=advantages_tensor.mean().detach(),
                returns_mean=returns_tensor.mean().detach(),
            ),
        )
        
        return total_loss, flatten_dict(stats)
    
    def step(
        self,
        queries: List[torch.LongTensor],
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform one PPO training step.
        
        Args:
            queries: List of query tensors
            generation_kwargs: Additional generation arguments
            
        Returns:
            Dictionary of training statistics
        """
        self.model.train()
        
        # Step 1: Generate responses
        response_tensors = []
        full_tensors = []
        
        for query in queries:
            response_tensor, full_tensor = self.generate(query, generation_kwargs)
            response_tensors.append(response_tensor)
            full_tensors.append(full_tensor)
            
        # Step 2: Compute logprobs and values with current model
        logprobs_list = []
        values_list = []
        attention_masks = []
        
        for query, response in zip(queries, response_tensors):
            # Ensure both query and response are on the same device
            query = query.to(self.current_device)
            response = response.to(self.current_device)
            full_input = torch.cat([query, response]).unsqueeze(0)
            attention_mask = torch.ones_like(full_input)
            attention_masks.append(attention_mask.squeeze(0))
            
            logits, _, values = self.model(input_ids=full_input, attention_mask=attention_mask)
            
            # Compute logprobs for response tokens
            # Note: logits are shifted, so response tokens start at q_len-1
            q_len = query.shape[0]
            r_len = response.shape[0]
            if q_len + r_len > logits.shape[1]:
                r_len = logits.shape[1] - q_len + 1
                response = response[:r_len]
                
            response_logits = logits[0, q_len-1:q_len-1+r_len, :]
            response_tokens = response.unsqueeze(0)
            response_logprobs = logprobs_from_logits(
                response_logits.unsqueeze(0),
                response_tokens,
            ).squeeze(0)
            
            logprobs_list.append(response_logprobs)
            
            # Get values for response tokens
            if q_len + r_len <= values.shape[1]:
                values_list.append(values[0, q_len-1:q_len-1+r_len])
            else:
                values_list.append(values[0, q_len-1:])
            
        # Step 3: Compute rewards
        rewards = self.compute_rewards(queries, response_tensors)
        
        # Step 4: Create batch and compute loss
        # Detach old logprobs and values (they should not have gradients)
        old_logprobs_detached = [lp.detach() for lp in logprobs_list]
        old_values_detached = [v.detach() for v in values_list]
        
        batch = PPORLBatch(
            query_tensors=queries,
            response_tensors=response_tensors,
            logprobs=old_logprobs_detached,
            values=old_values_detached,
            rewards=rewards,
            attention_mask=attention_masks,
        )
        
        # Multiple PPO epochs
        all_stats = []
        for ppo_epoch in range(self.config.ppo_epochs):
            loss, stats = self._step(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                
            all_stats.append(stats)
            self.current_step += 1
            
        # Average stats over PPO epochs
        avg_stats = {}
        for key in all_stats[0].keys():
            avg_stats[key] = sum([s[key] for s in all_stats]) / len(all_stats)
            
        return avg_stats

    def log_stats(
        self,
        stats: dict,
        batch: dict,
        rewards: List[torch.FloatTensor],
        columns_to_log: List[str] = ["query", "response"],
    ):
        """
        Log training statistics.
        """
        if self.accelerator.is_main_process:
            logs = {}

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

            for k, v in logs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    logs[k] = v.float()

            self.accelerator.log(
                logs,
                step=self.current_step if self.config.log_with == "tensorboard" else None,
            )
    
    @torch.no_grad()
    def evaluate(self, eval_dataloader):
        """
        Evaluate the model on evaluation data.
        """
        self.model.eval()
        
        total_reward = 0.0
        total_examples = 0
        
        for batch in eval_dataloader:
            if isinstance(batch["query"], list):
                queries = []
                for q in batch["query"]:
                    query_tokens = self.tokenizer(
                        q, 
                        padding=False, 
                        truncation=True, 
                        max_length=128, 
                        return_tensors='pt'
                    ).input_ids.squeeze(0)
                    queries.append(query_tokens)
            else:
                queries = [q.squeeze(0) for q in batch["query"]]
                
            # Generate responses
            responses = []
            for query in queries:
                response, _ = self.generate(query)
                responses.append(response)
                
            # Compute rewards
            rewards = self.compute_rewards(queries, responses)
            
            total_reward += sum([r.item() if isinstance(r, torch.Tensor) else r for r in rewards])
            total_examples += len(rewards)
            
        if total_examples == 0:
            mean_reward = 0.0
        else:
            mean_reward = total_reward / total_examples
            
        metrics = {
            "eval/mean_reward": mean_reward,
        }
        return metrics

