"""
Compare DPO loss on test set between trained model and estimated model.

This script:
1. Loads a trained DPO model
2. Loads precomputed gradients and estimates theta using logistic regression
3. Computes DPO loss on test set for both models
4. Reports relative error
"""

import os
import sys
import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trainers.network_utils import AutoModelForCausalLMWithValueHead
from trainers.dpo_trainer import DPOTrainer
from trainers.dpo_config import DPOConfig
from trainers.data_loader import load_data, load_model
from src.logistic_regression import (
    solve_logistic_regression,
    load_precomputed_gradients_b,
)
from trainers.kernel_estimation import unproject_theta


def compute_dpo_loss_on_testset(
    trained_model_path: str,
    precompute_file: str,
    base_model_name: str,
    preference_dataset_path: str,
    beta: float = 0.05,
    cache_dir: str = "cache",
    verbose: bool = True,
):
    """
    Compute DPO loss on test set for trained model and estimated model.
    
    Args:
        trained_model_path: Path to the trained DPO model checkpoint
        precompute_file: Filename of precomputed gradients (in pre_compute directory)
        base_model_name: Name of the base model (e.g., "Qwen/Qwen3-0.6B")
        preference_dataset_path: Path to preference dataset
        beta: Temperature parameter (beta)
        cache_dir: Cache directory
        verbose: Whether to print progress
        
    Returns:
        Dictionary with loss statistics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to absolute path
    trained_model_path = os.path.abspath(trained_model_path)
    
    # Check if it's a LoRA checkpoint
    adapter_config_path = os.path.join(trained_model_path, "adapter_config.json")
    is_lora_checkpoint = os.path.exists(adapter_config_path)
    
    # Load tokenizer
    if is_lora_checkpoint:
        try:
            tokenizer = AutoTokenizer.from_pretrained(trained_model_path, cache_dir=cache_dir, local_files_only=True)
            if verbose:
                print(f"Loaded tokenizer from checkpoint (vocab_size={len(tokenizer)})")
        except Exception as e:
            if verbose:
                print(f"Could not load tokenizer from checkpoint: {e}")
                print("Using base model tokenizer instead...")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)
            tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
            tokenizer.padding_side = "left"
            tokenizer.truncation_side = "left"
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
    
    # Load trained model
    if verbose:
        print(f"Loading trained model from: {trained_model_path}")
    
    if is_lora_checkpoint:
        if verbose:
            print("  Detected LoRA checkpoint, loading base model and LoRA adapter...")
        trained_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        trained_model.resize_token_embeddings(len(tokenizer))
        trained_model = PeftModel.from_pretrained(trained_model, trained_model_path)
        trained_model = trained_model.merge_and_unload()
        if verbose:
            print("  LoRA adapter loaded and merged")
    else:
        if verbose:
            print("  Loading full model checkpoint...")
        trained_model = AutoModelForCausalLM.from_pretrained(
            trained_model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            local_files_only=True,
        )
    
    trained_model = trained_model.to(device)
    trained_model.eval()
    
    # Load precomputed gradients and estimate theta
    if verbose:
        print(f"\nLoading precomputed gradients from: {precompute_file}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    gradients_np, b_values_np, z_values_np, projection_matrix, projection_dim = load_precomputed_gradients_b(
        precompute_file
    )
    
    if verbose:
        print(f"Loaded {len(gradients_np)} gradients")
        if projection_matrix is not None:
            print(f"Projection: {projection_matrix.shape[0]} -> {projection_matrix.shape[1]}")
        else:
            print("No projection used")
    
    # Estimate theta using logistic regression
    if verbose:
        print("\nEstimating theta using logistic regression...")
    theta_projected = solve_logistic_regression(
        gradients=gradients_np,
        b_values=b_values_np,
        z=z_values_np,
        max_iters=1000,
        lr=0.1,
        tol=1e-6,
        verbose=verbose,
    )
    
    if projection_matrix is not None:
        if verbose:
            print("\nUnprojecting theta to full parameter space...")
        theta_estimated_flat = unproject_theta(theta_projected, projection_matrix)
    else:
        theta_estimated_flat = theta_projected
    
    if verbose:
        print(f"Theta (full) shape: {theta_estimated_flat.shape}")
        print(f"Theta (full) norm: {np.linalg.norm(theta_estimated_flat):.6e}")
    
    # Load test dataset
    if verbose:
        print(f"\nLoading test dataset from: {preference_dataset_path}")
    data_args = argparse.Namespace(
        preference_dataset_path=preference_dataset_path,
        preference_dataset_subset=None,
        preference_dataset_split=None,
        downsample_ratio=0.05,  # Downsample test set to 0.05
        batch_size=1,  # Use batch_size=1 for simplicity (can be increased for efficiency)
        mini_batch_size=1,
        seed=42,
        cache_dir=cache_dir,
        batched=True,
        num_proc=32,
    )
    
    _, all_eval_dataloaders = load_data(data_args)
    test_dataloader = all_eval_dataloaders.get("eval_pref", None)
    
    if test_dataloader is None:
        raise ValueError("No eval_pref dataloader found in dataset")
    
    if verbose:
        print(f"Test set size: {len(test_dataloader.dataset)} samples")
    
    # Load base/reference model for DPO loss computation
    if verbose:
        print("\nLoading reference model for DPO loss computation...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    ref_model.resize_token_embeddings(len(tokenizer))
    ref_model = ref_model.to(device)
    ref_model.eval()
    
    # Compute DPO loss for trained model on test set
    if verbose:
        print("\nComputing DPO loss for trained model on test set...")
    
    trained_losses = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Trained model", disable=not verbose):
            if isinstance(batch["query"], list) or isinstance(batch["query"], tuple):
                queries = tokenizer(
                    batch["query"],
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).input_ids.to(device)
                
                all_pref = batch["response_w"] + batch["response_l"]
                tokenized = tokenizer(
                    all_pref,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).input_ids.to(device)
                
                n_w = len(batch["response_w"])
                responses_w = tokenized[:n_w]
                responses_l = tokenized[n_w:]
            else:
                queries = batch["query"].to(device)
                responses_w = batch["response_w"].to(device)
                responses_l = batch["response_l"].to(device)
            
            # Compute logprobs using DPOTrainer style (need both trained and reference model)
            from trainers.utils import logprobs_from_logits
            
            input_ids_w = torch.cat([queries, responses_w], dim=1)
            input_ids_l = torch.cat([queries, responses_l], dim=1)
            pad_id = tokenizer.pad_token_id
            
            def compute_seq_logprob(model, input_ids):
                attention_mask = (input_ids != pad_id).long()
                with torch.no_grad():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
                    attn_mask_shifted = attention_mask[:, 1:]
                    q_len = queries.size(1)
                    mask = attn_mask_shifted.clone()
                    if q_len > 1:
                        mask[:, :q_len-1] = 0
                    seq_logprob = (logprobs * mask).sum(dim=1)  # [batch_size]
                    return seq_logprob  # Return tensor, not scalar
            
            seq_logprob_w = compute_seq_logprob(trained_model, input_ids_w)  # [batch_size]
            seq_logprob_l = compute_seq_logprob(trained_model, input_ids_l)  # [batch_size]
            seq_ref_logprob_w = compute_seq_logprob(ref_model, input_ids_w)  # [batch_size]
            seq_ref_logprob_l = compute_seq_logprob(ref_model, input_ids_l)  # [batch_size]
            
            # DPO loss: -log(sigmoid(beta * ((logprob_w - logprob_l) - (ref_logprob_w - ref_logprob_l))))
            pi_logratio = seq_logprob_w - seq_logprob_l  # [batch_size]
            ref_logratio = seq_ref_logprob_w - seq_ref_logprob_l  # [batch_size]
            dpo_logit = beta * (pi_logratio - ref_logratio)  # [batch_size]
            losses = -torch.log(torch.sigmoid(dpo_logit))  # [batch_size]
            
            # Append losses for each sample in the batch
            for loss_val in losses.cpu().numpy():
                trained_losses.append(float(loss_val))
    
    trained_losses = np.array(trained_losses)
    trained_mean_loss = float(np.mean(trained_losses))
    
    if verbose:
        print(f"Trained model - Mean loss: {trained_mean_loss:.6f}")
        print(f"Trained model - Std loss: {np.std(trained_losses):.6f}")
    
    # Compute DPO loss for estimated model on test set
    # For estimated model, we need to compute gradients on test set and use logistic regression form
    if verbose:
        print("\nComputing DPO loss for estimated model on test set...")
        print("Note: Estimated model uses logistic regression form: loss = log(1 + exp(b - z * g^T theta))")
    
    # Apply estimated theta to model and do standard inference on test set
    # This is much faster than computing gradients for each test sample
    if verbose:
        print("\nApplying estimated theta to model for inference...")
    
    from trainers.data_loader import load_model
    
    # Load model with same structure as precomputation (LoRA if used)
    use_lora_for_estimated = is_lora_checkpoint
    estimated_model_wrapper = load_model(tokenizer, argparse.Namespace(
        pretrained_dir=base_model_name,
        cache_dir=cache_dir,
        use_lora=use_lora_for_estimated,
        lora_r=8,
        lora_alpha=32.0,
        lora_dropout=0.05,
        lora_target_modules="",
    ))
    estimated_model_wrapper = estimated_model_wrapper.to(device)
    estimated_model_wrapper.eval()
    
    # Apply theta to model parameters
    # theta_estimated_flat is the parameter delta (theta - theta_0)
    # We need to add it to reference model parameters
    # IMPORTANT: estimated_model_wrapper has ValueHead, and theta includes value head params
    # For LoRA models, only LoRA parameters have requires_grad=True (set automatically by PeftModel)
    # We must NOT set all parameters to requires_grad=True, as that would change which parameters are trainable
    
    # Get reference parameters: create reference model wrapper to get the reference parameters
    from trainers.utils import create_reference_model
    ref_model_wrapper = create_reference_model(estimated_model_wrapper)
    ref_model_wrapper = ref_model_wrapper.to(device)
    ref_model_wrapper.eval()
    
    # Match parameters by name to ensure correct ordering
    # Get trainable parameters from estimated model (only those with requires_grad=True)
    # This matches the logic in precomputation: model_params = [p for p in self.model.parameters() if p.requires_grad]
    model_param_dict = dict(estimated_model_wrapper.named_parameters())
    ref_param_dict = dict(ref_model_wrapper.named_parameters())
    
    model_params = []
    ref_params_for_trainable = []
    for name, param in estimated_model_wrapper.named_parameters():
        if param.requires_grad:  # Only include parameters that have requires_grad=True (LoRA params for LoRA models)
            model_params.append(param)
            ref_params_for_trainable.append(ref_param_dict[name])
    
    if len(model_params) != len(ref_params_for_trainable):
        raise ValueError(f"Parameter count mismatch: model has {len(model_params)} trainable params, ref has {len(ref_params_for_trainable)}")
    
    # Verify dimension matches theta
    total_params = sum(p.numel() for p in model_params)
    if total_params != len(theta_estimated_flat):
        raise ValueError(f"Parameter dimension mismatch: model has {total_params} trainable params, but theta has {len(theta_estimated_flat)}. "
                        f"This suggests the model structure doesn't match precomputation.")
    
    # Reshape theta_estimated_flat back to parameter shapes and apply
    theta_idx = 0
    for param, ref_param in zip(model_params, ref_params_for_trainable):
        param_shape = param.shape
        param_numel = param.numel()
        if theta_idx + param_numel > len(theta_estimated_flat):
            raise ValueError(f"Theta dimension mismatch: need {theta_idx + param_numel}, but only have {len(theta_estimated_flat)}")
        theta_slice = theta_estimated_flat[theta_idx:theta_idx + param_numel]
        theta_param = torch.from_numpy(theta_slice.reshape(param_shape)).float().to(device)
        # Apply: param = ref_param + theta
        param.data.copy_(ref_param.data + theta_param)
        theta_idx += param_numel
    
    if theta_idx != len(theta_estimated_flat):
        raise ValueError(f"Theta dimension mismatch: used {theta_idx}, but theta has {len(theta_estimated_flat)}")
    
    if verbose:
        print(f"Applied theta to {len(model_params)} parameter groups")
    
    # Now do standard inference on test set using the estimated model
    estimated_losses = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Estimated model", disable=not verbose):
            if isinstance(batch["query"], list) or isinstance(batch["query"], tuple):
                queries = tokenizer(
                    batch["query"],
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).input_ids.to(device)
                
                all_pref = batch["response_w"] + batch["response_l"]
                tokenized = tokenizer(
                    all_pref,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).input_ids.to(device)
                
                n_w = len(batch["response_w"])
                responses_w = tokenized[:n_w]
                responses_l = tokenized[n_w:]
            else:
                queries = batch["query"].to(device)
                responses_w = batch["response_w"].to(device)
                responses_l = batch["response_l"].to(device)
            
            # Compute logprobs using standard DPO loss computation
            from trainers.utils import logprobs_from_logits
            
            input_ids_w = torch.cat([queries, responses_w], dim=1)
            input_ids_l = torch.cat([queries, responses_l], dim=1)
            pad_id = tokenizer.pad_token_id
            
            def compute_seq_logprob(model, input_ids, is_value_head_wrapper=False):
                attention_mask = (input_ids != pad_id).long()
                with torch.no_grad():
                    if is_value_head_wrapper:
                        # For AutoModelForCausalLMWithValueHead, forward returns (logits, loss, value)
                        logits, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                    else:
                        # For AutoModelForCausalLM, forward returns CausalLMOutput
                        output = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = output.logits
                    logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
                    attn_mask_shifted = attention_mask[:, 1:]
                    q_len = queries.size(1)
                    mask = attn_mask_shifted.clone()
                    if q_len > 1:
                        mask[:, :q_len-1] = 0
                    seq_logprob = (logprobs * mask).sum(dim=1)  # [batch_size]
                    return seq_logprob  # Return tensor, not scalar
            
            seq_logprob_w = compute_seq_logprob(estimated_model_wrapper, input_ids_w, is_value_head_wrapper=True)  # [batch_size]
            seq_logprob_l = compute_seq_logprob(estimated_model_wrapper, input_ids_l, is_value_head_wrapper=True)  # [batch_size]
            seq_ref_logprob_w = compute_seq_logprob(ref_model, input_ids_w, is_value_head_wrapper=False)  # [batch_size]
            seq_ref_logprob_l = compute_seq_logprob(ref_model, input_ids_l, is_value_head_wrapper=False)  # [batch_size]
            
            # DPO loss: -log(sigmoid(beta * ((logprob_w - logprob_l) - (ref_logprob_w - ref_logprob_l))))
            pi_logratio = seq_logprob_w - seq_logprob_l  # [batch_size]
            ref_logratio = seq_ref_logprob_w - seq_ref_logprob_l  # [batch_size]
            dpo_logit = beta * (pi_logratio - ref_logratio)  # [batch_size]
            losses = -torch.log(torch.sigmoid(dpo_logit))  # [batch_size]
            
            # Append losses for each sample in the batch
            for loss_val in losses.cpu().numpy():
                estimated_losses.append(float(loss_val))
    
    estimated_losses = np.array(estimated_losses)
    estimated_mean_loss = float(np.mean(estimated_losses))
    
    if verbose:
        print(f"Estimated model - Mean loss: {estimated_mean_loss:.6f}")
        print(f"Estimated model - Std loss: {np.std(estimated_losses):.6f}")
    
    # Compute relative error
    absolute_error = abs(estimated_mean_loss - trained_mean_loss)
    relative_error = absolute_error / (abs(trained_mean_loss) + 1e-10) * 100.0
    
    results = {
        'trained_mean_loss': trained_mean_loss,
        'trained_std_loss': float(np.std(trained_losses)),
        'estimated_mean_loss': estimated_mean_loss,
        'estimated_std_loss': float(np.std(estimated_losses)),
        'absolute_error': absolute_error,
        'relative_error_percent': relative_error,
        'num_test_samples': len(trained_losses),
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("Loss Comparison on Test Set:")
        print(f"{'='*60}")
        print(f"Trained model:")
        print(f"  Mean loss: {results['trained_mean_loss']:.6f}")
        print(f"  Std loss: {results['trained_std_loss']:.6f}")
        print(f"\nEstimated model:")
        print(f"  Mean loss: {results['estimated_mean_loss']:.6f}")
        print(f"  Std loss: {results['estimated_std_loss']:.6f}")
        print(f"\nError:")
        print(f"  Absolute error: {results['absolute_error']:.6f}")
        print(f"  Relative error: {results['relative_error_percent']:.2f}%")
        print(f"\nNumber of test samples: {results['num_test_samples']}")
        print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compare DPO loss on test set between trained and estimated models'
    )
    parser.add_argument('--trained_model_path', type=str, required=True,
                       help='Path to trained DPO model checkpoint')
    parser.add_argument('--precompute_file', type=str, required=True,
                       help='Filename of precomputed gradients (in pre_compute directory)')
    parser.add_argument('--base_model_name', type=str, required=True,
                       help='Name of base model (e.g., Qwen/Qwen3-0.6B)')
    parser.add_argument('--preference_dataset_path', type=str, required=True,
                       help='Path to preference dataset')
    parser.add_argument('--beta', type=float, default=0.05,
                       help='Temperature parameter (beta)')
    parser.add_argument('--cache_dir', type=str, default='cache',
                       help='Cache directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')
    
    args = parser.parse_args()
    
    results = compute_dpo_loss_on_testset(
        trained_model_path=args.trained_model_path,
        precompute_file=args.precompute_file,
        base_model_name=args.base_model_name,
        preference_dataset_path=args.preference_dataset_path,
        beta=args.beta,
        cache_dir=args.cache_dir,
        verbose=args.verbose,
    )
    
    return results


if __name__ == "__main__":
    main()
