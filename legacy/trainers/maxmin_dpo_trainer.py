"""
MaxMin-DPO Trainer with EM Algorithm for Reward Learning

Implements Algorithm 2 (EM for reward learning) and MaxMin-DPO training
as described in "MaxMin Approach to Align with Diverse Human Preferences"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)
from torch.optim import AdamW
from typing import Optional, Union, Dict, List, Tuple, Any
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import logging

from trainers.dpo_trainer import DPOTrainer
from trainers.dpo_config import DPOConfig
from trainers.utils import logprobs_from_logits, set_seed
from dataclasses import dataclass
import typing

logger = logging.getLogger(__name__)

# Define PreTrainedModelWrapper type (same as in dpo_trainer.py)
PreTrainedModelWrapper = typing.Union[nn.Module, nn.DataParallel]


@dataclass
class MaxMinDPOConfig(DPOConfig):
    """Configuration for MaxMin-DPO training."""
    
    num_clusters: int = 2
    """Number of user clusters/subpopulations"""
    
    maxmin_weight: float = 0.1
    """Weight for MaxMin objective in combined loss"""
    
    # EM algorithm parameters
    em_convergence_threshold: float = 1e-4
    """Convergence threshold for EM algorithm"""
    
    em_max_iterations: int = 50
    """Maximum number of EM iterations"""
    
    reward_model_name: Optional[str] = None
    """Base model name for reward models (if None, will use model_name)"""
    
    reward_learning_rate: float = 1e-5
    """Learning rate for reward model training in M-step"""
    
    reward_num_epochs: int = 3
    """Number of epochs for reward model training in M-step"""
    
    reward_batch_size: int = 8
    """Batch size for reward model training"""


class RewardModel(nn.Module):
    """Simple reward model wrapper for sequence classification models."""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
    def forward(self, input_ids, attention_mask=None):
        """Forward pass that returns scalar rewards."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # For sequence classification with num_labels=1, logits is [batch_size, 1]
        if outputs.logits.dim() == 2 and outputs.logits.size(1) == 1:
            return outputs.logits.squeeze(-1)  # [batch_size]
        return outputs.logits  # Fallback
    
    def compute_reward(self, x: str, y: str) -> float:
        """Compute reward for a (prompt, response) pair."""
        text = f"{x}{y}"  # Simple concatenation, can be customized
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        # Ensure input_ids is on the correct device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            reward = self.forward(**inputs)
        return reward.item() if reward.numel() == 1 else reward.mean().item()


def compute_reward_probability(reward_model: RewardModel, x: str, y1: str, y2: str, device: str) -> float:
    """
    Compute w(φ_u, x, y1, y2) = exp(r_φ_u(y1,x)) / (exp(r_φ_u(y1,x)) + exp(r_φ_u(y2,x)))
    """
    reward_model.eval()
    with torch.no_grad():
        r_y1 = reward_model.compute_reward(x, y1)
        r_y2 = reward_model.compute_reward(x, y2)
        
        # Use log-sum-exp trick for numerical stability
        exp_r_y1 = np.exp(r_y1)
        exp_r_y2 = np.exp(r_y2)
        w = exp_r_y1 / (exp_r_y1 + exp_r_y2)
    return w


class EMRewardLearner:
    """
    Algorithm 2: Learning Rewards with EM Algorithm
    
    Implements the EM algorithm to learn reward models for different user clusters.
    """
    
    def __init__(
        self,
        num_clusters: int,
        reward_model_name: str,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cuda",
        convergence_threshold: float = 1e-4,
        max_iterations: int = 50,
        reward_learning_rate: float = 1e-5,
        reward_num_epochs: int = 3,
    ):
        self.num_clusters = num_clusters
        self.reward_model_name = reward_model_name
        self.tokenizer = tokenizer
        self.device = device
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.reward_learning_rate = reward_learning_rate
        self.reward_num_epochs = reward_num_epochs
        
        # Initialize reward models for each cluster
        # IMPORTANT: Reward models need their own tokenizer, not the policy tokenizer!
        # Using policy tokenizer with reward model can cause vocab size mismatch
        try:
            reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
            if reward_tokenizer.pad_token is None:
                reward_tokenizer.pad_token = reward_tokenizer.eos_token or "[PAD]"
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for reward model {reward_model_name}: {e}. Using policy tokenizer.")
            reward_tokenizer = tokenizer
        
        self.reward_models = []
        for u in range(num_clusters):
            config = AutoConfig.from_pretrained(reward_model_name)
            config.num_labels = 1
            model = AutoModelForSequenceClassification.from_pretrained(
                reward_model_name, config=config
            )
            model.to(device)
            reward_model = RewardModel(model, reward_tokenizer)  # Use reward_tokenizer, not policy tokenizer
            self.reward_models.append(reward_model)
        
        # Cluster assignments: user_id -> cluster_id
        self.user_assignments = {}
    
    def e_step(self, preference_data: List[Dict]) -> Dict[int, int]:
        """
        E-step: Hard cluster assignment for each user.
        
        For each user h, assign to cluster u that maximizes:
        Π_{(x,y1,y2,h) ∈ D} w(φ_u, x, y1, y2)
        
        Args:
            preference_data: List of dicts with keys: 'user_id', 'prompt', 'chosen', 'rejected'
            
        Returns:
            Dictionary mapping user_id to cluster_id
        """
        # Group data by user
        user_data = {}
        for item in preference_data:
            user_id = item.get('user_id', item.get('annotator', 0))  # Support multiple field names
            if user_id not in user_data:
                user_data[user_id] = []
            user_data[user_id].append(item)
        
        # Assign each user to the best cluster
        assignments = {}
        for user_id, items in tqdm(user_data.items(), desc="E-step: Assigning users to clusters"):
            best_cluster = 0
            best_score = -float('inf')
            
            for u in range(self.num_clusters):
                # Compute product of w(φ_u, x, y1, y2) for all preference pairs
                log_likelihood = 0.0
                for item in items:
                    x = item['prompt']
                    y1 = item['chosen']
                    y2 = item['rejected']
                    
                    try:
                        w = compute_reward_probability(self.reward_models[u], x, y1, y2, self.device)
                        # Use log to avoid numerical underflow
                        log_likelihood += np.log(w + 1e-10)
                    except Exception as e:
                        logger.warning(f"Error computing reward probability: {e}")
                        continue
                
                if log_likelihood > best_score:
                    best_score = log_likelihood
                    best_cluster = u
            
            assignments[user_id] = best_cluster
        
        self.user_assignments = assignments
        return assignments
    
    def m_step(self, preference_data: List[Dict], assignments: Dict[int, int]):
        """
        M-step: Update each reward model φ_u by minimizing negative log-likelihood
        on the assigned users' data.
        """
        # Group data by cluster
        cluster_data = {u: [] for u in range(self.num_clusters)}
        for item in preference_data:
            user_id = item.get('user_id', item.get('annotator', 0))
            if user_id in assignments:
                cluster_id = assignments[user_id]
                cluster_data[cluster_id].append(item)
        
        # Train each reward model on its assigned data
        for u in range(self.num_clusters):
            if len(cluster_data[u]) == 0:
                logger.warning(f"Cluster {u} has no assigned data, skipping update")
                continue
            
            logger.info(f"M-step: Training reward model for cluster {u} on {len(cluster_data[u])} samples")
            self._train_reward_model(self.reward_models[u], cluster_data[u])
    
    def _train_reward_model(self, reward_model: RewardModel, data: List[Dict], batch_size: int = 8):
        """Train a single reward model on preference data using ranking loss."""
        reward_model.train()
        optimizer = AdamW(reward_model.model.parameters(), lr=self.reward_learning_rate)
        
        # Simple training loop with batching
        for epoch in range(self.reward_num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Process in batches
            for batch_start in range(0, len(data), batch_size):
                batch_end = min(batch_start + batch_size, len(data))
                batch_data = data[batch_start:batch_end]
                
                # Collect texts
                chosen_texts = []
                rejected_texts = []
                for item in batch_data:
                    x = item['prompt']
                    y_chosen = item['chosen']
                    y_rejected = item['rejected']
                    chosen_texts.append(f"{x}{y_chosen}")
                    rejected_texts.append(f"{x}{y_rejected}")
                
                # Tokenize batch with proper padding
                # Use max_length padding to ensure consistent tensor sizes
                chosen_inputs = reward_model.tokenizer(
                    chosen_texts, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512, 
                    padding="max_length",
                    return_attention_mask=True
                )
                rejected_inputs = reward_model.tokenizer(
                    rejected_texts, 
                    return_tensors="pt", 
                    truncation=True,
                    max_length=512, 
                    padding="max_length",
                    return_attention_mask=True
                )
                
                # Clamp input_ids to valid vocab range to avoid index errors
                vocab_size = reward_model.model.config.vocab_size
                chosen_inputs['input_ids'] = torch.clamp(chosen_inputs['input_ids'], 0, vocab_size - 1)
                rejected_inputs['input_ids'] = torch.clamp(rejected_inputs['input_ids'], 0, vocab_size - 1)
                
                # Move to device and ensure proper dtype
                device = next(reward_model.model.parameters()).device
                chosen_inputs = {k: v.to(device) for k, v in chosen_inputs.items()}
                rejected_inputs = {k: v.to(device) for k, v in rejected_inputs.items()}
                
                # Forward pass
                try:
                    reward_chosen = reward_model(**chosen_inputs)
                    reward_rejected = reward_model(**rejected_inputs)
                except RuntimeError as e:
                    logger.error(f"Error in forward pass: {e}")
                    logger.error(f"Chosen input_ids shape: {chosen_inputs['input_ids'].shape}")
                    logger.error(f"Rejected input_ids shape: {rejected_inputs['input_ids'].shape}")
                    logger.error(f"Chosen input_ids max: {chosen_inputs['input_ids'].max()}, vocab size: {reward_model.model.config.vocab_size}")
                    raise
                
                # Ranking loss: -log sigmoid(reward_chosen - reward_rejected)
                loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"  Epoch {epoch+1}/{self.reward_num_epochs}, Loss: {avg_loss:.4f}")
        
        reward_model.eval()
    
    def fit(self, preference_data: List[Dict]) -> List[RewardModel]:
        """
        Run EM algorithm to learn reward models.
        
        Returns:
            List of trained reward models (one per cluster)
        """
        logger.info(f"Starting EM algorithm with {self.num_clusters} clusters")
        
        prev_assignments = None
        for iteration in range(self.max_iterations):
            logger.info(f"\n=== EM Iteration {iteration + 1}/{self.max_iterations} ===")
            
            # E-step: Assign users to clusters
            assignments = self.e_step(preference_data)
            
            # Check convergence
            if prev_assignments is not None:
                # Count how many assignments changed
                num_changes = sum(
                    1 for user_id in assignments 
                    if user_id in prev_assignments and assignments[user_id] != prev_assignments[user_id]
                )
                total_users = len(assignments)
                change_ratio = num_changes / total_users if total_users > 0 else 0.0
                
                logger.info(f"Assignment change ratio: {change_ratio:.4f}")
                
                if change_ratio < self.convergence_threshold:
                    logger.info("Converged!")
                    break
            
            prev_assignments = assignments.copy()
            
            # M-step: Update reward models
            self.m_step(preference_data, assignments)
        
        logger.info("EM algorithm completed")
        return self.reward_models


class MaxMinDPOTrainer(DPOTrainer):
    """
    MaxMin-DPO Trainer
    
    Extends DPO trainer to use MaxMin objective across multiple user clusters.
    Uses reward models learned via EM algorithm to identify clusters.
    """
    
    def __init__(
        self,
        config: DPOConfig = None,
        model: PreTrainedModelWrapper = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        tokenizer: PreTrainedTokenizerBase = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        reward_models: Optional[List[RewardModel]] = None,
        user_assignments: Optional[Dict[int, int]] = None,
        num_clusters: int = 2,
        **kwargs
    ):
        super().__init__(
            config=config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            **kwargs
        )
        
        self.reward_models = reward_models or []
        self.user_assignments = user_assignments or {}
        self.num_clusters = num_clusters
        
        # Cluster utilities (will be computed during training)
        self.cluster_utilities = {u: [] for u in range(num_clusters)}
    
    def _compute_cluster_utilities(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute utilities for each cluster.
        
        Utility for cluster u: F_{r_φ_u}(π) = E_{x~P, y~π(.|x)}[r_φ_u(y,x)]
        
        For DPO, we approximate this using the log probability difference on the batch,
        which represents the policy's preference alignment with each cluster.
        """
        # Get logprobs from model (similar to DPO)
        input_ids_w = torch.cat((queries, responses_w), dim=1)
        input_ids_l = torch.cat((queries, responses_l), dim=1)
        pad_id = self.tokenizer.pad_token_id
        
        def process_input_ids(input_ids):
            attention_mask = (input_ids != pad_id).long()
            input_data = {"input_ids": input_ids, "attention_mask": attention_mask}
            logits, _, _ = self.model(**input_data)
            with torch.no_grad():
                ref_logits, _, _ = self.ref_model(**input_data)
                ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], input_ids[:, 1:])
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            attn_mask_shifted = attention_mask[:, 1:]
            return logprobs, ref_logprobs, attn_mask_shifted
        
        logprobs_w, ref_logprobs_w, attn_shift_w = process_input_ids(input_ids_w)
        logprobs_l, ref_logprobs_l, attn_shift_l = process_input_ids(input_ids_l)
        
        q_len = queries.size(1)
        mask_w = attn_shift_w.clone()
        mask_l = attn_shift_l.clone()
        if q_len > 1:
            mask_w[:, :q_len - 1] = 0
            mask_l[:, :q_len - 1] = 0
        
        seq_logprob_w = (logprobs_w * mask_w).sum(dim=1)
        seq_logprob_l = (logprobs_l * mask_l).sum(dim=1)
        seq_ref_logprob_w = (ref_logprobs_w * mask_w).sum(dim=1)
        seq_ref_logprob_l = (ref_logprobs_l * mask_l).sum(dim=1)
        
        # Compute utility per sample: log prob difference (represents alignment)
        utilities = (seq_logprob_w - seq_logprob_l) - (seq_ref_logprob_w - seq_ref_logprob_l)
        
        # Group utilities by cluster based on user assignments
        cluster_utilities = {u: [] for u in range(self.num_clusters)}
        
        if user_ids is not None:
            user_ids_cpu = user_ids.cpu().numpy() if isinstance(user_ids, torch.Tensor) else user_ids
            for i, user_id in enumerate(user_ids_cpu):
                cluster_id = self.user_assignments.get(int(user_id), 0)
                cluster_utilities[cluster_id].append(utilities[i])
        else:
            # If no user_ids, assign all samples to cluster 0
            for i in range(utilities.size(0)):
                cluster_utilities[0].append(utilities[i])
        
        # Convert to tensors
        result = {}
        for u in range(self.num_clusters):
            if len(cluster_utilities[u]) > 0:
                result[u] = torch.stack(cluster_utilities[u])
            else:
                result[u] = torch.tensor([0.0], device=queries.device, requires_grad=False)
        
        return result
    
    def _step(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
        preference_mask: Optional[torch.BoolTensor] = None,
        user_ids: Optional[torch.Tensor] = None,
    ):
        """
        MaxMin-DPO step: maximize minimum utility across clusters.
        
        The objective is: π* = arg max_π min_{u ∈ U} F_{r_φ_u}(π) - βD_KL[π || π_ref]
        """
        # Compute standard DPO loss first
        dpo_loss, stats = super()._step(
            queries=queries,
            responses_w=responses_w,
            responses_l=responses_l,
            preference_mask=preference_mask,
        )
        
        # Compute cluster utilities for MaxMin objective
        cluster_utilities = self._compute_cluster_utilities(
            queries, responses_w, responses_l, user_ids
        )
        
        # Add MaxMin objective: maximize minimum utility across clusters
        if len(cluster_utilities) > 0 and all(len(util) > 0 for util in cluster_utilities.values()):
            # Compute mean utility per cluster
            cluster_means = {u: torch.mean(util) for u, util in cluster_utilities.items() if len(util) > 0}
            
            if len(cluster_means) > 0:
                # Find minimum utility across clusters
                utility_values = list(cluster_means.values())
                min_utility = torch.stack(utility_values).min()
                
                # MaxMin objective: maximize the minimum utility
                # We add -min_utility to the loss (negative because we want to maximize)
                maxmin_weight = getattr(self.config, 'maxmin_weight', 0.1)
                maxmin_loss = -maxmin_weight * min_utility
                
                # Combine DPO loss with MaxMin objective
                combined_loss = dpo_loss + maxmin_loss
                
                # Update stats
                stats['maxmin/min_cluster_utility'] = min_utility.detach()
                stats['maxmin/maxmin_loss'] = maxmin_loss.detach()
                
                # Log per-cluster utilities
                for u, mean_util in cluster_means.items():
                    stats[f'maxmin/cluster_{u}_utility'] = mean_util.detach()
                
                return combined_loss, stats
        
        return dpo_loss, stats
    
    def step(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
        preference_mask: Optional[torch.BoolTensor] = None,
        user_ids: Optional[torch.Tensor] = None,
    ):
        """
        MaxMin-DPO step with user_ids support.
        """
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
            user_ids_ = user_ids[i : i + sub_bs] if user_ids is not None else None

            loss, stats = self._step(
                queries=queries_,
                responses_w=responses_w_,
                responses_l=responses_l_,
                preference_mask=preference_mask_,
                user_ids=user_ids_,
            )
            
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.current_step += 1
            
            all_stats.append(stats)
        
        # Aggregate stats
        aggregated_stats = {}
        for key in all_stats[0].keys():
            values = [s[key] for s in all_stats if key in s]
            if values:
                if isinstance(values[0], torch.Tensor):
                    aggregated_stats[key] = torch.stack(values).mean()
                else:
                    aggregated_stats[key] = np.mean(values)
        
        return aggregated_stats

