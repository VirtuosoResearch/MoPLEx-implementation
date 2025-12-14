import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
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

class ApproxDPOTrainer():
    def __init__(
        self,
        config: DPOConfig = None,
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

        # Accelerator setup
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
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
            raise ValueError("ApproxDPOTrainer does not support encoder-decoder models.")

        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        config.is_encoder_decoder = self.is_encoder_decoder
        config.is_peft_model = self.is_peft_model

        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=config.to_dict(),
            init_kwargs=config.tracker_kwargs,
        )
        self.tokenizer = tokenizer

        self.dataset = dataset
        self._signature_columns = None
        if self.dataset is not None:
            self.dataloader = self.prepare_dataloader(self.dataset, data_collator)
        else:
            self.dataloader = None

        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        if optimizer is None:
            self.optimizer = Adam(self.model_params, lr=self.config.learning_rate)
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(self.config.init_kl_coef, self.config.target, self.config.horizon)
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)

        (
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        )
        self.ref_model = self.accelerator.prepare(self.ref_model)
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"
        self.current_step = 0
        self.current_device = self.accelerator.device
        self.running = RunningMoments(self.accelerator)
        
        # Storage for precomputed gradients and b values at reference model (θ0)
        # According to paper A.1: g = ∇_θ h_θ0, b = -h_θ0
        self.precomputed_gradients = None  # List of gradient vectors for each sample
        self.precomputed_b_values = None   # List of b values for each sample
        
        # Random projection matrix for dimension reduction (Johnson-Lindenstrauss)
        self.projection_matrix = None  # [num_params, projection_dim] if projection_dim is not None
        self.projection_dim = config.projection_dim if hasattr(config, 'projection_dim') else None

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            self._signature_columns += ["label", "query", "response"]

    def _remove_unused_columns(self, dataset: "Dataset"):
        if not self.config.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"],
                columns=columns,
                format_kwargs=dataset.format["format_kwargs"],
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def prepare_dataloader(self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None):
        if isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.dataloader_batch_size or self.config.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    def _generate_projection_matrix(self, num_params: int, projection_dim: int, seed: Optional[int] = None):
        """
        Generate a random projection matrix for Johnson-Lindenstrauss dimension reduction.
        
        Args:
            num_params: Original dimension (number of model parameters)
            projection_dim: Target dimension after projection
            seed: Random seed for reproducibility
            
        Returns:
            projection_matrix: [num_params, projection_dim] numpy array
        """
        if seed is not None:
            np.random.seed(seed)
        # Generate ±1 random matrix and normalize
        # This is the standard Johnson-Lindenstrauss projection
        projection_matrix = (2 * np.random.randint(2, size=(num_params, projection_dim)) - 1).astype(np.float32)
        projection_matrix *= 1.0 / np.sqrt(projection_dim)
        return projection_matrix
    
    def _ensure_projection_matrix(self):
        """
        Ensure projection matrix is generated if projection is enabled.
        Should be called after model is set up.
        """
        if self.projection_dim is None:
            return
        
        if self.projection_matrix is None:
            # Count number of trainable parameters
            model_params = [p for p in self.model.parameters() if p.requires_grad]
            num_params = sum(p.numel() for p in model_params)
            
            print(f"Generating random projection matrix: {num_params} -> {self.projection_dim}")
            self.projection_matrix = self._generate_projection_matrix(
                num_params=num_params,
                projection_dim=self.projection_dim,
                seed=self.config.seed
            )
            print(f"Projection matrix shape: {self.projection_matrix.shape}")

    def precompute_gradients_and_b(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
        sample_indices: Optional[torch.LongTensor] = None,
    ):
        """
        Precompute gradients g and b values at reference model (θ0) for each sample.
        According to paper A.1:
        - g = ∇_θ h_θ0(x, y1, y2) = β(∇_θ log π_θ0(y_w|x) - ∇_θ log π_θ0(y_l|x))
        - b = -h_θ0(x, y1, y2) = -β(log π_θ0(y_w|x) - log π_θ0(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))
        
        Args:
            queries: Query tensors [B, Lq]
            responses_w: Winner response tensors [B, Lw]
            responses_l: Loser response tensors [B, Ll]
            sample_indices: Optional indices to map samples to storage positions
            
        Returns:
            gradients: List of gradient vectors [B, num_params]
            b_values: List of b values [B]
        """
        # Temporarily set model to reference model state for gradient computation
        original_state = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Save reference model state
        ref_state_dict = {name: param.clone() for name, param in self.ref_model.named_parameters()}
        
        # Set model to reference model state (θ0)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in ref_state_dict:
                    param.data.copy_(ref_state_dict[name].data)
        
        input_ids_w = torch.cat((queries, responses_w), dim=1)
        input_ids_l = torch.cat((queries, responses_l), dim=1)
        pad_id = self.tokenizer.pad_token_id
        
        batch_size = queries.size(0)
        beta = self.config.temperature
        
        # Compute logprobs at θ0
        def process_at_theta0(input_ids):
            attention_mask = (input_ids != pad_id).long()
            input_data = {"input_ids": input_ids, "attention_mask": attention_mask}
            logits, _, _ = self.model(**input_data)
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            attn_mask_shifted = attention_mask[:, 1:]
            return logprobs, attn_mask_shifted
        
        logprobs_w_theta0, attn_shift_w = process_at_theta0(input_ids_w)
        logprobs_l_theta0, attn_shift_l = process_at_theta0(input_ids_l)
        
        # Also compute at ref_model for b calculation
        with torch.no_grad():
            old_logits_w, _, _ = self.ref_model(**{"input_ids": input_ids_w, "attention_mask": (input_ids_w != pad_id).long()})
            old_logits_l, _, _ = self.ref_model(**{"input_ids": input_ids_l, "attention_mask": (input_ids_l != pad_id).long()})
            old_logprobs_w = logprobs_from_logits(old_logits_w[:, :-1, :], input_ids_w[:, 1:])
            old_logprobs_l = logprobs_from_logits(old_logits_l[:, :-1, :], input_ids_l[:, 1:])
        
        q_len = queries.size(1)
        mask_w = attn_shift_w.clone()
        mask_l = attn_shift_l.clone()
        if q_len > 1:
            mask_w[:, :q_len - 1] = 0
            mask_l[:, :q_len - 1] = 0
        
        # Compute sequence-level logprobs at θ0
        seq_logprob_w_theta0 = (logprobs_w_theta0 * mask_w).sum(dim=1)  # [B]
        seq_logprob_l_theta0 = (logprobs_l_theta0 * mask_l).sum(dim=1)  # [B]
        seq_old_logprob_w = (old_logprobs_w * mask_w).sum(dim=1)  # [B]
        seq_old_logprob_l = (old_logprobs_l * mask_l).sum(dim=1)  # [B]
        
        # Compute h_θ0 and b
        # h_θ0 = β(log π_θ0(y_w|x) - log π_θ0(y_l|x)) - β(log π_ref(y_w|x) - log π_ref(y_l|x))
        # Since model is set to θ0 (ref_model), and ref_model is also θ0:
        # h_θ0 = β(log π_θ0(y_w|x) - log π_θ0(y_l|x)) - β(log π_θ0(y_w|x) - log π_θ0(y_l|x)) = 0
        # So b = -h_θ0 = 0
        # However, we compute it explicitly for correctness
        h_theta0 = beta * ((seq_logprob_w_theta0 - seq_logprob_l_theta0) - (seq_old_logprob_w - seq_old_logprob_l))  # [B]
        b_values = -h_theta0.detach().cpu().numpy()  # [B]
        
        # Compute gradients g = β(∇_θ log π_θ0(y_w|x) - ∇_θ log π_θ0(y_l|x))
        model_params = [p for p in self.model.parameters() if p.requires_grad]
        gradients_list = []
        
        for i in range(batch_size):
            # Compute gradient for sample i
            grads_w_i = torch.autograd.grad(
                outputs=seq_logprob_w_theta0[i],
                inputs=model_params,
                create_graph=False,
                retain_graph=True,
                allow_unused=True
            )
            grads_l_i = torch.autograd.grad(
                outputs=seq_logprob_l_theta0[i],
                inputs=model_params,
                create_graph=False,
                retain_graph=True,
                allow_unused=True
            )
            
            # Flatten and concatenate gradients
            # IMPORTANT: Include ALL parameters that require_grad, even if gradient is None
            # This ensures dimension consistency with training time
            # If gradient is None, use zero gradient (parameter doesn't contribute to loss)
            grad_diff = []
            for param, gw, gl in zip(model_params, grads_w_i, grads_l_i):
                if gw is not None and gl is not None:
                    grad_diff.append((gw - gl).detach().cpu().flatten())
                elif gw is None and gl is None:
                    # Parameter has no gradient (not in computation graph), use zero
                    # This maintains dimension consistency with training time
                    zero_grad = torch.zeros_like(param).cpu().flatten()
                    grad_diff.append(zero_grad)
                else:
                    # One gradient is None but the other isn't - this shouldn't happen
                    raise ValueError(f"Inconsistent gradients for parameter: {param.shape}")
            
            if grad_diff:
                g_i = beta * torch.cat(grad_diff).numpy()  # [num_params]
            else:
                g_i = np.array([])
            
            # Apply random projection if enabled
            if self.projection_dim is not None:
                if self.projection_matrix is None:
                    self._ensure_projection_matrix()
                if len(g_i) > 0:
                    # Project: g_projected = projection_matrix^T @ g
                    g_i = self.projection_matrix.T @ g_i  # [projection_dim]
            
            gradients_list.append(g_i)
        
        # Restore original model state
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in original_state:
                    param.data.copy_(original_state[name].data)
        
        return gradients_list, b_values

    def save_precomputed_gradients_and_b(self, save_path: str):
        """Save precomputed gradients and b values to disk."""
        if self.precomputed_gradients is None or self.precomputed_b_values is None:
            raise ValueError("No precomputed gradients and b values to save. Call precompute_gradients_and_b first.")
        
        data = {
            'gradients': self.precomputed_gradients,
            'b_values': self.precomputed_b_values,
        }
        
        # Save projection matrix if it exists
        if self.projection_matrix is not None:
            data['projection_matrix'] = self.projection_matrix
            data['projection_dim'] = self.projection_dim
        
        torch.save(data, save_path, _use_new_zipfile_serialization=False)
        print(f"Saved precomputed gradients and b values to {save_path}")
        if self.projection_matrix is not None:
            print(f"  Projection matrix included: {self.projection_matrix.shape}")

    def load_precomputed_gradients_and_b(self, load_path: str):
        """Load precomputed gradients and b values from disk."""
        # Use weights_only=False to allow loading numpy arrays (PyTorch 2.6+ compatibility)
        data = torch.load(load_path, map_location='cpu', weights_only=False)
        self.precomputed_gradients = data['gradients']
        self.precomputed_b_values = data['b_values']
        
        # Load projection matrix if it exists
        if 'projection_matrix' in data:
            self.projection_matrix = data['projection_matrix']
            if 'projection_dim' in data:
                self.projection_dim = data['projection_dim']
            print(f"  Loaded projection matrix: {self.projection_matrix.shape}")
        elif self.projection_dim is not None:
            # Projection is enabled but not found in saved file
            # This might be okay if we're loading old files, but warn the user
            print(f"  Warning: Projection is enabled (dim={self.projection_dim}) but projection matrix not found in saved file.")
            print(f"  Will generate new projection matrix if needed.")
        
        print(f"Loaded precomputed gradients and b values from {load_path}")
        print(f"  Number of samples: {len(self.precomputed_gradients)}")
        if len(self.precomputed_gradients) > 0:
            grad_dim = len(self.precomputed_gradients[0])
            print(f"  Gradient dimension: {grad_dim}")

    def _step(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
        return_stats: bool = False,
        preference_mask: Optional[torch.BoolTensor] = None,
        compute_first_order: bool = True,
        sample_indices: Optional[torch.LongTensor] = None,
    ):  
        input_ids_w = torch.cat((queries, responses_w), dim=1)  # [B, Lq+Lw]
        input_ids_l = torch.cat((queries, responses_l), dim=1)  # [B, Lq+Ll]
        pad_id = self.tokenizer.pad_token_id

        def process_input_ids(input_ids):
            # attention_mask: pad=0, others=1
            attention_mask = (input_ids != pad_id).long()  # [B, L]
            input_data = {"input_ids": input_ids, "attention_mask": attention_mask}

            logits, _, _ = self.model(**input_data)
            with torch.no_grad():
                old_logits, _, _ = self.ref_model(**input_data)
                old_logprobs = logprobs_from_logits(old_logits[:, :-1, :], input_ids[:, 1:])  # [B, L-1]
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])  # [B, L-1]
            attn_mask_shifted = attention_mask[:, 1:]  # [B, L-1]
            entropy = entropy_from_logits(logits)

            return logprobs, old_logprobs, attn_mask_shifted, entropy

        logprobs_w, old_logprobs_w, attn_shift_w, entropy_w = process_input_ids(input_ids_w)
        logprobs_l, old_logprobs_l, attn_shift_l, entropy_l = process_input_ids(input_ids_l)

        q_len = queries.size(1)
        mask_w = attn_shift_w.clone()  # [B, L-1]
        mask_l = attn_shift_l.clone()
        if q_len > 1:
            mask_w[:, :q_len - 1] = 0
            mask_l[:, :q_len - 1] = 0

        if preference_mask is not None:
            sample_mask = preference_mask.to(logprobs_w.device).float()  # [B]
        else:
            sample_mask = torch.ones(logprobs_w.size(0), device=logprobs_w.device)

        # Compute sequence-level logprobs
        seq_logprob_w = (logprobs_w * mask_w).sum(dim=1)          # [B]
        seq_logprob_l = (logprobs_l * mask_l).sum(dim=1)          # [B]
        seq_old_logprob_w = (old_logprobs_w * mask_w).sum(dim=1)  # [B]
        seq_old_logprob_l = (old_logprobs_l * mask_l).sum(dim=1)  # [B]

        # Compute sequence-level logratios
        pi_logratios_seq = seq_logprob_w - seq_logprob_l               # [B]
        ref_logratios_seq = seq_old_logprob_w - seq_old_logprob_l      # [B]

        # According to paper A.1, we approximate h_θ(x, y1, y2) using logistic regression:
        # l_hat(x, y1, y2, z) = log(1 + exp(b - z * g^T (θ - θ0)))
        # where:
        # - g = ∇_θ h_θ0(x, y1, y2) = β(∇_θ log π_θ0(y_w|x) - ∇_θ log π_θ0(y_l|x)) [precomputed]
        # - b = -h_θ0(x, y1, y2) [precomputed]
        # - z = +1 for y_w preferred, -1 for y_l preferred
        
        batch_size = seq_logprob_w.size(0)
        beta = self.config.temperature  # β in the paper
        
        # MUST use precomputed gradients - no online computation allowed
        if self.precomputed_gradients is None or self.precomputed_b_values is None:
            raise ValueError(
                "ApproxDPOTrainer requires precomputed gradients and b values. "
                "Please call load_precomputed_gradients_and_b() or precompute_gradients_and_b() first."
            )
        if len(self.precomputed_gradients) == 0:
            raise ValueError("Precomputed gradients list is empty. Cannot proceed without precomputed gradients.")
        
        # Use precomputed gradients and b values for logistic regression
        if compute_first_order:
            # Use precomputed gradients and b values for logistic regression
            # l_hat(x, y1, y2, z) = log(1 + exp(b - z * g^T (θ - θ0)))
            # For DPO, z = +1 (y_w is preferred), so: l_hat = log(1 + exp(b - g^T (θ - θ0)))
            
            model_params = [p for p in self.model.parameters() if p.requires_grad]
            ref_model_params = []
            model_param_iter = iter(self.model.parameters())
            ref_model_param_iter = iter(self.ref_model.parameters())
            for p in model_param_iter:
                p_ref = next(ref_model_param_iter)
                if p.requires_grad:
                    ref_model_params.append(p_ref)
            
            # Compute (θ - θ0) for all parameters
            # IMPORTANT: Use the same parameter filtering logic as in precomputation
            # We need to match exactly which parameters were included in precomputation
            # In precomputation, we only include parameters that have non-None gradients
            # So here we should only include parameters that would have gradients
            theta_diff_flat_list = []
            for p, p_theta0 in zip(model_params, ref_model_params):
                # Include all parameters that require grad (same filtering as model_params)
                # The gradient computation will determine which ones actually contribute
                theta_diff_flat_list.append((p - p_theta0).detach().cpu().flatten())
            theta_diff_flat = torch.cat(theta_diff_flat_list).numpy()  # [num_params]
            
            # Apply random projection if enabled
            if self.projection_dim is not None:
                if self.projection_matrix is None:
                    self._ensure_projection_matrix()
                # Project: theta_diff_projected = projection_matrix^T @ theta_diff
                theta_diff_flat = self.projection_matrix.T @ theta_diff_flat  # [projection_dim]
            
            # Compute logistic regression loss for each sample
            # According to paper A.1: l_hat(x, y1, y2, z) = log(1 + exp(b - z * g^T (θ - θ0)))
            # For DPO, z = +1 (y_w is preferred), so: l_hat = log(1 + exp(b - g^T (θ - θ0)))
            loss_vec = []
            for i in range(batch_size):
                # Use sample index if provided, otherwise use batch index
                sample_idx = sample_indices[i].item() if sample_indices is not None and i < len(sample_indices) else i
                
                if sample_idx >= len(self.precomputed_gradients) or sample_idx >= len(self.precomputed_b_values):
                    raise ValueError(
                        f"Sample index {sample_idx} out of range. "
                        f"Precomputed gradients has {len(self.precomputed_gradients)} samples, "
                        f"b_values has {len(self.precomputed_b_values)} samples."
                    )
                
                g_i = self.precomputed_gradients[sample_idx]  # [num_params]
                b_i = self.precomputed_b_values[sample_idx]    # scalar
                
                # Ensure dimensions match
                if len(g_i) != len(theta_diff_flat):
                    # Provide more detailed error message
                    model_param_count = sum(p.numel() for p in model_params if p.requires_grad)
                    raise ValueError(
                        f"Gradient dimension mismatch:\n"
                        f"  Precomputed gradient dimension: {len(g_i)}\n"
                        f"  Current model parameter dimension: {len(theta_diff_flat)}\n"
                        f"  Total trainable parameters: {model_param_count}\n"
                        f"  Difference: {abs(len(g_i) - len(theta_diff_flat))}\n"
                        f"This usually happens when:\n"
                        f"  1. The model architecture changed between precomputation and training\n"
                        f"  2. Tokenizer vocabulary size changed (affects embedding layer)\n"
                        f"  3. LoRA configuration changed\n"
                        f"Please recompute gradients with the same model configuration used for training."
                    )
                
                # Compute g^T (θ - θ0)
                g_dot_theta_diff = np.dot(g_i, theta_diff_flat)  # scalar
                
                # l_hat = log(1 + exp(b - g^T (θ - θ0)))
                # This is the approximate DPO loss according to paper A.1
                loss_i = np.log(1 + np.exp(b_i - g_dot_theta_diff))
                loss_vec.append(loss_i)
            
            # Convert to tensor - these are already loss values from logistic regression
            dpo_loss_vec = torch.tensor(loss_vec, device=self.current_device, dtype=torch.float32)
        else:
            # When compute_first_order=False (e.g., in evaluation mode), still use precomputed gradients
            # but compute standard DPO loss for evaluation
            # Use standard DPO: h_θ = β(log π_θ(y_w|x) - log π_θ(y_l|x)) - β(log π_ref(y_w|x) - log π_ref(y_l|x))
            approx_logits = beta * (pi_logratios_seq - ref_logratios_seq)  # [B]

            if self.config.ipo_loss:
                dpo_loss_vec = (approx_logits - 1.0 / (2 * self.config.temperature)) ** 2  # [B]
            else:
                dpo_loss_vec = -F.logsigmoid(approx_logits)  # [B]

        # Apply preference_mask for weighted average
        denom = sample_mask.sum()
        if denom.item() == 0:
            dpo_loss = dpo_loss_vec.mean()
        else:
            dpo_loss = (dpo_loss_vec * sample_mask).sum() / denom

        if return_stats:
            delta_w = seq_logprob_w - seq_old_logprob_w  # [B]
            delta_l = seq_logprob_l - seq_old_logprob_l  # [B]
            rewards_chosen = self.config.temperature * delta_w.detach()
            rewards_rejected = self.config.temperature * delta_l.detach()
            reward_margin = rewards_chosen - rewards_rejected

            stats = dict(
                loss=dict(dpo_loss=dpo_loss.detach()),
                policy=dict(
                    entropy=torch.cat((entropy_w, entropy_l), dim=0).detach(),
                    rewards_chosen=rewards_chosen.mean().detach(),
                    rewards_rejected=rewards_rejected.mean().detach(),
                    reward_margin=reward_margin.mean().detach(),
                    logprobs_w=seq_logprob_w.mean().detach(),
                    logprobs_l=seq_logprob_l.mean().detach(),
                    pi_logratios=pi_logratios_seq.mean().detach(),
                    ref_logratios=ref_logratios_seq.mean().detach(),
                    dpo_logit_mean=approx_logits.mean().detach(),
                    classifier_accuracy=(reward_margin > 0).float().mean().detach(),
                )
            )
            return dpo_loss, flatten_dict(stats)
        else:
            return dpo_loss

    def step(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
        preference_mask: Optional[torch.BoolTensor] = None,
        sample_indices: Optional[torch.LongTensor] = None,
    ):
        self.model.train()
        bs = self.config.batch_size
        sub_bs = self.config.mini_batch_size
        assert bs % sub_bs == 0
        for i in tqdm.tqdm(range(0, bs, sub_bs), desc="Training with Minibatches", leave=False):
            queries_ = queries[i:i + sub_bs]
            responses_w_ = responses_w[i:i + sub_bs]
            responses_l_ = responses_l[i:i + sub_bs]
            preference_mask_ = preference_mask[i:i + sub_bs] if preference_mask is not None else None
            sample_indices_ = sample_indices[i:i + sub_bs] if sample_indices is not None else None

            loss, stats = self._step(
                queries=queries_,
                responses_w=responses_w_,
                responses_l=responses_l_,
                return_stats=True,
                preference_mask=preference_mask_,
                sample_indices=sample_indices_,
            )

            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.current_step += 1
        return stats

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
    
    @torch.no_grad()
    def evaluate(self, eval_dataloader):
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
                return_stats=True,
                compute_first_order=False,  # Skip first-order term in evaluation mode
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
            "eval/dpo_loss": mean_loss,
            "eval/classification_accuracy": mean_acc,
        }
        return metrics

    def end_of_epoch_step(self, epoch: int):
        """Performs tasks at the end of an epoch, like saving models."""
        print(f"Executing end-of-epoch tasks for epoch: {epoch}")

        # save p* (params + grads) for ApproxDPO
        if (
            hasattr(self.config, 'save_pstar_at_epoch') and
            self.config.save_pstar_at_epoch >= 0 and
            epoch == self.config.save_pstar_at_epoch and
            not getattr(self, 'has_saved_pstar', False) and
            self.accelerator.is_main_process
        ):
            self.has_saved_pstar = True
            print(f"Saving p* (params + grads) at epoch {epoch} to {self.config.pstar_save_path} ...")
            self.save_pstar()

        # save p (params only)
        if (
            hasattr(self.config, 'save_p_at_epoch') and
            self.config.save_p_at_epoch >= 0 and
            epoch == self.config.save_p_at_epoch and
            not getattr(self, 'has_saved_p', False) and
            self.accelerator.is_main_process
        ):
            self.has_saved_p = True
            print(f"Saving p (params only) at epoch {epoch} to {self.config.p_save_path} ...")
            self.save_p()

    def save_pstar(self):
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            params_vector = torch.cat([p.detach().to(device).flatten() for p in self.model.parameters()])
        torch.save({"params": params_vector}, self.config.pstar_save_path)
        print(f"Saved p* to {self.config.pstar_save_path}")
        self.model.train()

    def save_p(self):
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            params_vector = torch.cat([p.detach().to(device).flatten() for p in self.model.parameters()])
        torch.save({"params": params_vector}, self.config.p_save_path)
        print(f"Saved p to {self.config.p_save_path}")
        self.model.train()
