"""
EM-DPO Trainer: Using EM Algorithm for Reward Learning, Standard DPO for Training

This trainer uses EM algorithm to learn reward models and clusters,
but then uses standard DPO (not MaxMin-DPO) for training.
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
class EMDPOConfig(DPOConfig):
    """Configuration for EM-DPO training."""
    
    num_clusters: int = 2
    """Number of user clusters/subpopulations"""
    
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


class EMDPOTrainer(DPOTrainer):
    """
    EM-DPO Trainer
    
    Uses EM algorithm to learn reward models and clusters, but then
    uses standard DPO (not MaxMin-DPO) for training.
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
        
        # Log cluster information
        if user_assignments:
            for cluster_id in range(num_clusters):
                count = sum(1 for uid, cid in user_assignments.items() if cid == cluster_id)
                logger.info(f"Cluster {cluster_id}: {count} users")

