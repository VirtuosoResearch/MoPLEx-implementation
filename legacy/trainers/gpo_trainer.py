"""
GPO Trainer: Group Preference Optimization Trainer.
Implements GPM training with bilinear scoring and cross-entropy loss.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM
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
from dataclasses import dataclass, field
from typing import Literal

from trainers.dpo_config import DPOConfig, flatten_dict
from trainers.modeling_base import PreTrainedModelWrapper
from trainers.utils import (
    RunningMoments,
    AdaptiveKLController,
    FixedKLController,
    masked_mean,
    flatten_dict as flatten_dict_util,
    set_seed,
    is_torch_greater_2_0,
)


# ========== Configuration ==========
@dataclass
class GPOConfig(DPOConfig):
    """Configuration class for GPO (Group Preference Optimization) Trainer."""
    # GPO-specific hyperparameters
    embedding_dim: int = 16
    """Embedding dimension (2k). The actual embedding will be 2k-dimensional, where k = embedding_dim // 2"""
    beta: float = 1.0
    """Temperature scaling hyperparameter β for the sigmoid in loss function. Start with 1.0"""
    use_order_bias_augmentation: bool = True
    """Whether to use order bias augmentation: 50% probability to swap (yw, yl) and flip label"""
    use_l2_normalization: bool = True
    """Whether to apply L2 normalization to embeddings (essential for stability)"""
    use_gating_network: bool = False
    """Whether to use optional gating network G_λ(x) that weights each 2D subspace"""
    gating_hidden_size: int = 128
    """Hidden size for gating network (if used)"""
    pooling_type: Literal["last_token", "mean"] = "last_token"
    """How to extract hidden states: 'last_token' or 'mean' pooling"""
    
    def __post_init__(self):
        super().__post_init__()
        if self.embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even (2k), got {self.embedding_dim}")
        if self.beta <= 0:
            raise ValueError("beta must be positive")


# ========== Model Components ==========
def create_antisymmetric_matrix(k: int, device: torch.device = None, dtype: torch.dtype = None):
    """Create the antisymmetric matrix R_> as block-diagonal matrix."""
    R = torch.zeros(2 * k, 2 * k, device=device, dtype=dtype)
    block = torch.tensor([[0, -1], [1, 0]], device=device, dtype=dtype)
    for i in range(k):
        R[2*i:2*i+2, 2*i:2*i+2] = block
    return R


class GPMEmbeddingHead(nn.Module):
    """Embedding head that maps hidden states to 2k-dimensional embeddings."""
    def __init__(self, config, embedding_dim: int, pooling_type: str = "last_token", use_l2_normalization: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pooling_type = pooling_type
        self.use_l2_normalization = use_l2_normalization
        hidden_size = config.word_embed_proj_dim if hasattr(config, "word_embed_proj_dim") else config.hidden_size
        self.linear = nn.Linear(hidden_size, embedding_dim)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if self.pooling_type == "last_token":
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.size(0)
                embeddings = hidden_states[torch.arange(batch_size, device=hidden_states.device), lengths]
            else:
                embeddings = hidden_states[:, -1, :]
        else:  # mean pooling
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                masked_states = hidden_states * mask_expanded
                sum_states = masked_states.sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1)
                embeddings = sum_states / (sum_mask + 1e-8)
            else:
                embeddings = hidden_states.mean(dim=1)
        embeddings = self.linear(embeddings)
        if self.use_l2_normalization:
            embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalization (optional)
        return embeddings


class OptionalGatingNetwork(nn.Module):
    """Optional gating network G_λ(x) that weights each 2D subspace (D(x))."""
    def __init__(self, config, k: int, hidden_size: int = 128):
        super().__init__()
        self.k = k
        input_hidden_size = config.word_embed_proj_dim if hasattr(config, "word_embed_proj_dim") else config.hidden_size
        self.gating_network = nn.Sequential(
            nn.Linear(input_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, k),
            nn.Softplus(),  # Ensure positive weights (eigenvalue scale gate)
        )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if hidden_states.dim() == 3:
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                masked_states = hidden_states * mask_expanded
                sum_states = masked_states.sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1)
                hidden_states = sum_states / (sum_mask + 1e-8)
            else:
                hidden_states = hidden_states.mean(dim=1)
        # Get D(x) weights for each 2D subspace
        D_weights = self.gating_network(hidden_states)  # (batch_size, k)
        return D_weights


class GPMScoring(nn.Module):
    """Bilinear scoring function using antisymmetric matrix R_> and optional D(x) gating."""
    def __init__(self, k: int, use_gating: bool = False, gating_network: Optional[nn.Module] = None):
        super().__init__()
        self.k = k
        self.embedding_dim = 2 * k
        self.use_gating = use_gating
        self.gating_network = gating_network
        R = create_antisymmetric_matrix(k)
        self.register_buffer('R', R)
    
    def forward(self, v_w: torch.Tensor, v_l: torch.Tensor, D_weights: Optional[torch.Tensor] = None):
        """
        Compute preference score.
        
        If D(x) gating is used: s = v_w^T D(x) R_ D(x) v_l
        Otherwise: s = v_w^T R_ v_l
        """
        if self.use_gating and D_weights is not None:
            # Apply D(x) scaling to each 2D subspace
            # D_weights: (batch_size, k)
            # Scale each 2D block of embeddings
            batch_size = v_w.size(0)
            v_w_scaled = v_w.clone()
            v_l_scaled = v_l.clone()
            
            for i in range(self.k):
                # Scale the 2D subspace: [v_w[:, 2*i], v_w[:, 2*i+1]]
                scale = D_weights[:, i].unsqueeze(-1)  # (batch_size, 1)
                v_w_scaled[:, 2*i:2*i+2] = v_w[:, 2*i:2*i+2] * scale
                v_l_scaled[:, 2*i:2*i+2] = v_l[:, 2*i:2*i+2] * scale
            
            # Compute bilinear score with scaled embeddings
            Rv_w_scaled = torch.matmul(v_w_scaled, self.R.T)
            scores = (Rv_w_scaled * v_l_scaled).sum(dim=1)
        else:
            # Standard bilinear score without gating
            Rv_w = torch.matmul(v_w, self.R.T)
            scores = (Rv_w * v_l).sum(dim=1)
        return scores


class AutoModelForCausalLMWithGPM(PreTrainedModelWrapper):
    """Model wrapper that adds GPM embedding head and scoring for GPO."""
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    
    def __init__(self, pretrained_model, embedding_dim: int = 16, pooling_type: str = "last_token",
                 use_gating: bool = False, gating_hidden_size: int = 128, use_l2_normalization: bool = True):
        super().__init__(pretrained_model)
        k = embedding_dim // 2
        if 2 * k != embedding_dim:
            raise ValueError(f"embedding_dim must be even (2k), got {embedding_dim}")
        self.embedding_head = GPMEmbeddingHead(
            self.pretrained_model.config, embedding_dim, pooling_type, use_l2_normalization
        )
        # Initialize gating network if needed
        self.use_gating = use_gating
        gating_network = None
        if use_gating:
            gating_network = OptionalGatingNetwork(self.pretrained_model.config, k, gating_hidden_size)
        self.gating_network = gating_network
        self.scoring = GPMScoring(k=k, use_gating=use_gating, gating_network=gating_network)
        self.embedding_dim = embedding_dim
        self.k = k
        self.is_peft_model = hasattr(self.pretrained_model, "peft_config") or hasattr(self.pretrained_model, "base_model")
    
    def _has_lm_head(self):
        if any(hasattr(self.pretrained_model, attr) for attr in self.lm_head_namings):
            return True
        if hasattr(self.pretrained_model, "base_model") and hasattr(self.pretrained_model.base_model, "model"):
            if any(hasattr(self.pretrained_model.base_model.model, attr) for attr in self.lm_head_namings):
                return True
        for name, _ in self.pretrained_model.named_modules():
            if any(attr in name for attr in self.lm_head_namings):
                return True
        return False
    
    def forward(self, input_ids=None, attention_mask=None, return_embeddings: bool = False, return_gating: bool = False, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        base_model_output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = getattr(base_model_output, 'loss', None)
        embeddings = None
        gating_weights = None
        if return_embeddings:
            embeddings = self.embedding_head(last_hidden_state, attention_mask)
        if return_gating and self.use_gating and self.gating_network is not None:
            # Get D(x) weights from prompt (use first part of sequence or mean pooling)
            # For simplicity, use mean pooling over all tokens
            gating_weights = self.gating_network(last_hidden_state, attention_mask)
        if return_embeddings:
            if return_gating:
                return (lm_logits, loss, embeddings, gating_weights)
            else:
                return (lm_logits, loss, embeddings)
        else:
            return (lm_logits, loss)
    
    def compute_preference_score(self, v_w: torch.Tensor, v_l: torch.Tensor, D_weights: Optional[torch.Tensor] = None):
        return self.scoring(v_w, v_l, D_weights)
    
    def state_dict(self, *args, **kwargs):
        """Returns the state dictionary of the model."""
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            # if it is a peft model, only save the embedding_head, scoring, and gating_network
            pretrained_model_state_dict = {}
        
        embedding_head_state_dict = self.embedding_head.state_dict(*args, **kwargs)
        for k, v in embedding_head_state_dict.items():
            pretrained_model_state_dict[f"embedding_head.{k}"] = v
        
        scoring_state_dict = self.scoring.state_dict(*args, **kwargs)
        for k, v in scoring_state_dict.items():
            pretrained_model_state_dict[f"scoring.{k}"] = v
        
        if self.use_gating and self.gating_network is not None:
            gating_state_dict = self.gating_network.state_dict(*args, **kwargs)
            for k, v in gating_state_dict.items():
                pretrained_model_state_dict[f"gating_network.{k}"] = v
        
        return pretrained_model_state_dict


# ========== Trainer ==========
class GPOTrainer:
    """GPO Trainer implementing GPM with bilinear scoring and cross-entropy loss."""
    
    def __init__(self, config: GPOConfig = None, model: PreTrainedModelWrapper = None,
                 tokenizer: PreTrainedTokenizerBase = None, dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None, data_collator: Optional[typing.Callable] = None,
                 lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 additional_config_kwargs: Optional[dict] = None):
        self.config = config
        set_seed(self.config.seed)
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            deepspeed_plugin=None,
            **config.accelerator_kwargs,
        )
        self.model = model
        self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        if self.is_encoder_decoder:
            raise ValueError("GPO does not support encoder-decoder models.")
        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        config.is_encoder_decoder = self.is_encoder_decoder
        config.is_peft_model = self.is_peft_model
        
        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        current_config = dict(trl_gpo_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict()
        current_config.update(flatten_dict_util(additional_config_kwargs or {}))
        self.accelerator.init_trackers(config.tracker_project_name, config=current_config, init_kwargs=config.tracker_kwargs)
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        if optimizer is None:
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.learning_rate)
        else:
            self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        (self.model, self.optimizer, self.data_collator, self.lr_scheduler) = self.accelerator.prepare(
            self.model, self.optimizer, self.data_collator, self.lr_scheduler)
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"
        self.current_step = 0
        self.current_device = self.accelerator.device
        self.running = RunningMoments(self.accelerator)
    
    def _apply_order_bias_augmentation(self, v_w: torch.Tensor, v_l: torch.Tensor, labels: torch.Tensor):
        """Apply order bias augmentation: 50% probability to swap (v_w, v_l) and flip label."""
        batch_size = v_w.size(0)
        swap_mask = torch.rand(batch_size, device=v_w.device) < 0.5
        v_w_aug = torch.where(swap_mask.unsqueeze(-1), v_l, v_w)
        v_l_aug = torch.where(swap_mask.unsqueeze(-1), v_w, v_l)
        labels_aug = torch.where(swap_mask, -labels, labels)
        return v_w_aug, v_l_aug, labels_aug
    
    def _step(self, queries: torch.LongTensor, responses_w: torch.LongTensor, responses_l: torch.LongTensor,
              preference_mask: Optional[torch.BoolTensor] = None, apply_augmentation: bool = True) -> Tuple[torch.Tensor, dict]:
        pad_id = self.tokenizer.pad_token_id
        device = queries.device
        input_ids_w = torch.cat((queries, responses_w), dim=1)
        input_ids_l = torch.cat((queries, responses_l), dim=1)
        
        def get_embeddings_and_gating(input_ids):
            attention_mask = (input_ids != pad_id).long()
            model_output = self.model(
                **{"input_ids": input_ids, "attention_mask": attention_mask},
                return_embeddings=True,
                return_gating=self.config.use_gating_network
            )
            embeddings = model_output[2]
            gating_weights = model_output[3] if len(model_output) > 3 and self.config.use_gating_network else None
            return embeddings, attention_mask, gating_weights
        
        v_w, attn_mask_w, D_w = get_embeddings_and_gating(input_ids_w)
        v_l, attn_mask_l, D_l = get_embeddings_and_gating(input_ids_l)
        
        # Labels: +1 for y_w > y_l, -1 for y_l > y_w (after augmentation)
        labels = torch.ones(v_w.size(0), device=device)
        
        if apply_augmentation and self.config.use_order_bias_augmentation:
            v_w, v_l, labels = self._apply_order_bias_augmentation(v_w, v_l, labels)
            # Also swap gating weights if applicable
            if D_w is not None and D_l is not None:
                swap_mask = (labels == -1)  # Samples that were swapped
                D_w_orig, D_l_orig = D_w.clone(), D_l.clone()
                D_w = torch.where(swap_mask.unsqueeze(-1), D_l_orig, D_w)
                D_l = torch.where(swap_mask.unsqueeze(-1), D_w_orig, D_l)
        
        # Compute raw preference scores (before applying labels)
        # Use D_w for both (assuming prompt is the same)
        D_weights = D_w if D_w is not None else None
        raw_scores = self.model.compute_preference_score(v_w, v_l, D_weights)
        
        # For loss: apply label to get correct direction
        # If label is -1 (swapped), we want s(y_l > y_w) = -s(y_w > y_l)
        scores_for_loss = raw_scores * labels
        scaled_scores = scores_for_loss / self.config.beta
        loss = -F.logsigmoid(scaled_scores)
        
        if preference_mask is not None:
            sample_mask = preference_mask.to(device).float()
        else:
            sample_mask = torch.ones(loss.size(0), device=device)
        
        denom = sample_mask.sum()
        total_loss = (loss * sample_mask).sum() / denom if denom.item() > 0 else loss.mean()
        
        # Fix accuracy calculation: use raw_scores, not scores after multiplying by labels
        # pred = +1 if raw_score > 0 (y_w > y_l), -1 if raw_score < 0 (y_l > y_w)
        # label = +1 means y_w should win, -1 means y_l should win
        predictions = (raw_scores > 0).float() * 2 - 1  # Convert to +1/-1
        correct = (predictions * labels > 0).float()  # Same sign means correct
        accuracy = (correct * sample_mask).sum() / (sample_mask.sum() + 1e-8)
        
        stats = dict(
            loss=dict(gpo_loss=total_loss.detach()),
            scores=dict(
                preference_scores_mean=raw_scores.mean().detach(),
                preference_scores_std=raw_scores.std().detach(),
                accuracy=accuracy.detach()
            ),
            embeddings=dict(embedding_norm_w=v_w.norm(dim=1).mean().detach(), embedding_norm_l=v_l.norm(dim=1).mean().detach()),
        )
        return total_loss, flatten_dict_util(stats)
    
    def step(self, queries: torch.LongTensor, responses_w: torch.LongTensor, responses_l: torch.LongTensor,
             preference_mask: Optional[torch.BoolTensor] = None):
        assert queries.ndim == 2 and responses_w.ndim == 2 and responses_l.ndim == 2
        self.model.train()
        # Fix batch size logic: use actual batch size from input, not config
        actual_bs = queries.size(0)
        sub_bs = self.config.mini_batch_size
        assert actual_bs % sub_bs == 0 or actual_bs < sub_bs, f"actual_bs ({actual_bs}) must be divisible by sub_bs ({sub_bs})"
        all_stats = []
        for i in range(0, actual_bs, sub_bs):
            end_idx = min(i + sub_bs, actual_bs)  # Handle last incomplete batch
            if i >= actual_bs:
                break
            queries_ = queries[i : end_idx]
            responses_w_ = responses_w[i : end_idx]
            responses_l_ = responses_l[i : end_idx]
            preference_mask_ = preference_mask[i : end_idx] if preference_mask is not None else None
            loss, stats = self._step(queries=queries_, responses_w=responses_w_, responses_l=responses_l_,
                                     preference_mask=preference_mask_, apply_augmentation=True)
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.current_step += 1
            all_stats.append(stats)
        
        if len(all_stats) > 1:
            avg_stats = {}
            for key in all_stats[0].keys():
                if isinstance(all_stats[0][key], (int, float, torch.Tensor)):
                    if isinstance(all_stats[0][key], torch.Tensor):
                        avg_stats[key] = torch.stack([s[key] for s in all_stats]).mean()
                    else:
                        avg_stats[key] = np.mean([s[key] for s in all_stats])
                else:
                    avg_stats[key] = all_stats[0][key]
            return avg_stats
        else:
            return all_stats[0]
    
    @torch.no_grad()
    def evaluate(self, eval_dataloader):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_examples = 0
        for batch in eval_dataloader:
            preference_mask = batch.get("preference_mask", None)
            if isinstance(batch["query"], list) or isinstance(batch["query"], tuple):
                queries = self.tokenizer(batch["query"], padding=True, truncation=True, max_length=128, return_tensors="pt").input_ids.to(self.current_device)
                all_pref = batch["response_w"] + batch["response_l"]
                tokenized = self.tokenizer(all_pref, padding=True, truncation=True, max_length=256, return_tensors="pt").input_ids.to(self.current_device)
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
            loss, stats = self._step(queries=queries, responses_w=responses_w, responses_l=responses_l,
                                     preference_mask=preference_mask, apply_augmentation=False)
            batch_size = queries.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
            batch_acc = stats["scores/accuracy"].item()
            total_acc += batch_acc * batch_size
        
        mean_loss = total_loss / total_examples if total_examples > 0 else 0.0
        mean_acc = total_acc / total_examples if total_examples > 0 else 0.0
        return {"eval/gpo_loss": mean_loss, "eval/accuracy": mean_acc}
    
    def end_of_epoch_step(self, epoch: int):
        pass
    
    def log_stats(self, stats: dict, batch: dict, rewards: List[torch.FloatTensor],
                  columns_to_log: List[str] = ["query", "response"]):
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
            self.accelerator.log(logs, step=self.current_step if self.config.log_with == "tensorboard" else None)
