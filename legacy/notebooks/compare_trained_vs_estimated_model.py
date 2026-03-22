"""
Compare the distance between a trained DPO model and an estimated model using ApproxDPO.

This script:
1. Loads a trained DPO model (trained on IMDB conciseness data)
2. Loads precomputed gradients for the same training data
3. Estimates theta using logistic regression (ApproxDPO method)
4. Computes the parameter distance between the trained model and estimated model
"""

import os
import sys
import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trainers.network_utils import AutoModelForCausalLMWithValueHead
from trainers.approx_dpo_trainer import ApproxDPOTrainer
from trainers.dpo_config import DPOConfig
from trainers.data_loader import load_model
from notebooks.logistic_regression import (
    solve_logistic_regression,
    load_precomputed_gradients_b,
)
from trainers.kernel_estimation import unproject_theta


def compute_model_distance(
    trained_model_path: str,
    precompute_file: str,
    base_model_name: str,
    cache_dir: str = "cache",
    verbose: bool = True,
):
    """
    Compute the distance between a trained model and an estimated model.
    
    Args:
        trained_model_path: Path to the trained DPO model checkpoint
        precompute_file: Filename of precomputed gradients (in pre_compute directory)
        base_model_name: Name of the base model (e.g., "Qwen/Qwen3-0.6B")
        cache_dir: Cache directory
        verbose: Whether to print progress
        
    Returns:
        Dictionary with distance metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if it's a LoRA checkpoint
    adapter_config_path = os.path.join(trained_model_path, "adapter_config.json")
    is_lora_checkpoint = os.path.exists(adapter_config_path)
    
    # Load tokenizer - prefer from checkpoint if available, otherwise use base model
    if is_lora_checkpoint:
        # Try loading tokenizer from checkpoint first (should have the same vocab as training)
        try:
            tokenizer = AutoTokenizer.from_pretrained(trained_model_path, cache_dir=cache_dir)
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
    
    # Load base/reference model
    if verbose:
        print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
        device_map="cpu",  # Load to CPU first
    )
    # Resize embeddings to match tokenizer (same as training)
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load trained model (LoRA checkpoint)
    if verbose:
        print(f"Loading trained model from: {trained_model_path}")
    
    if is_lora_checkpoint:
        # It's a LoRA checkpoint, need to load base model first, then LoRA
        if verbose:
            print("  Detected LoRA checkpoint, loading base model and LoRA adapter...")
        trained_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        # Resize embeddings to match the tokenizer used in training
        trained_model.resize_token_embeddings(len(tokenizer))
        # Load LoRA adapter
        trained_model = PeftModel.from_pretrained(trained_model, trained_model_path)
        # Merge LoRA weights into base model for comparison
        trained_model = trained_model.merge_and_unload()
        if verbose:
            print("  LoRA adapter loaded and merged")
    else:
        # It's a full model checkpoint
        if verbose:
            print("  Loading full model checkpoint...")
        trained_model = AutoModelForCausalLM.from_pretrained(
            trained_model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
    
    # Move models to CPU for distance computation (to save memory)
    base_model = base_model.to("cpu")
    trained_model = trained_model.to("cpu")
    
    # Compute parameter difference: Δ_trained = θ_trained - θ_ref
    if verbose:
        print("Computing parameter difference (trained - base)...")
    delta_trained_list = []
    base_params_list = []
    trained_params_list = []
    
    for p_base, p_trained in zip(base_model.parameters(), trained_model.parameters()):
        if p_base.requires_grad:
            delta = (p_trained - p_base).detach().cpu().flatten()
            delta_trained_list.append(delta)
            base_params_list.append(p_base.detach().cpu().flatten())
            trained_params_list.append(p_trained.detach().cpu().flatten())
    
    delta_trained_flat = torch.cat(delta_trained_list).numpy()  # [num_params]
    base_params_flat = torch.cat(base_params_list).numpy()  # [num_params]
    trained_params_flat = torch.cat(trained_params_list).numpy()  # [num_params]
    
    # Compute L2 norm of base model (for vectors, use ord=2, not 'fro')
    base_norm = np.linalg.norm(base_params_flat, ord=2)
    
    # Compute L2 norm of trained model difference
    delta_trained_norm = np.linalg.norm(delta_trained_flat, ord=2)
    relative_delta_trained = delta_trained_norm / base_norm if base_norm > 0 else 0.0
    
    if verbose:
        print(f"Base model norm (||θ_ref||_F): {base_norm:.6e}")
        print(f"Trained model difference norm (||θ_trained - θ_ref||_F): {delta_trained_norm:.6e}")
        print(f"Relative difference (trained): {relative_delta_trained:.6e}")
    
    # Load precomputed gradients
    if verbose:
        print(f"\nLoading precomputed gradients from: {precompute_file}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    precompute_path = os.path.join(script_dir, "..", "pre_compute", precompute_file)
    
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
    
    if verbose:
        print(f"Theta (projected) shape: {theta_projected.shape}")
        print(f"Theta (projected) norm: {np.linalg.norm(theta_projected):.6e}")
    
    # Unproject theta to full parameter space
    if projection_matrix is not None:
        if verbose:
            print("\nUnprojecting theta to full parameter space...")
        theta_estimated_flat = unproject_theta(theta_projected, projection_matrix)
    else:
        theta_estimated_flat = theta_projected
    
    if verbose:
        print(f"Theta (full) shape: {theta_estimated_flat.shape}")
        print(f"Theta (full) norm: {np.linalg.norm(theta_estimated_flat):.6e}")
    
    # Compare dimensions
    if delta_trained_flat.shape[0] != theta_estimated_flat.shape[0]:
        print(f"\nWARNING: Dimension mismatch!")
        print(f"  Trained model difference: {delta_trained_flat.shape[0]} parameters")
        print(f"  Estimated theta: {theta_estimated_flat.shape[0]} parameters")
        print(f"\nThis might happen if:")
        print(f"  - The models have different architectures")
        print(f"  - Only trainable parameters were included in gradient computation")
        print(f"  - LoRA was used in training but not in gradient computation (or vice versa)")
        print(f"\nComputing distance on the minimum of the two dimensions...")
        min_dim = min(delta_trained_flat.shape[0], theta_estimated_flat.shape[0])
        delta_trained_flat = delta_trained_flat[:min_dim]
        theta_estimated_flat = theta_estimated_flat[:min_dim]
    
    # Compute distance between trained and estimated parameter changes
    if verbose:
        print("\nComputing distance between trained and estimated models...")
    diff = delta_trained_flat - theta_estimated_flat
    diff_norm = np.linalg.norm(diff, ord=2)  # L2 norm for vectors
    relative_diff = diff_norm / base_norm if base_norm > 0 else 0.0
    
    # Compute cosine similarity
    dot_product = np.dot(delta_trained_flat, theta_estimated_flat)
    norm_trained = np.linalg.norm(delta_trained_flat)
    norm_estimated = np.linalg.norm(theta_estimated_flat)
    cosine_sim = dot_product / (norm_trained * norm_estimated + 1e-10)
    
    results = {
        'base_model_norm': float(base_norm),
        'trained_delta_norm': float(delta_trained_norm),
        'relative_trained_delta': float(relative_delta_trained),
        'estimated_theta_norm': float(np.linalg.norm(theta_estimated_flat)),
        'relative_estimated_theta': float(np.linalg.norm(theta_estimated_flat) / base_norm) if base_norm > 0 else 0.0,
        'difference_norm': float(diff_norm),
        'relative_difference': float(relative_diff),
        'cosine_similarity': float(cosine_sim),
        'num_params_trained': int(delta_trained_flat.shape[0]),
        'num_params_estimated': int(theta_estimated_flat.shape[0]),
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("Distance Metrics:")
        print(f"{'='*60}")
        print(f"Base model norm (||θ_ref||_F): {results['base_model_norm']:.6e}")
        print(f"\nTrained model:")
        print(f"  ||θ_trained - θ_ref||_F: {results['trained_delta_norm']:.6e}")
        print(f"  Relative: {results['relative_trained_delta']:.6e}")
        print(f"\nEstimated model:")
        print(f"  ||θ_estimated||_F: {results['estimated_theta_norm']:.6e}")
        print(f"  Relative: {results['relative_estimated_theta']:.6e}")
        print(f"\nDifference:")
        print(f"  ||(θ_trained - θ_ref) - θ_estimated||_F: {results['difference_norm']:.6e}")
        print(f"  Relative: {results['relative_difference']:.6e}")
        print(f"\nCosine similarity: {results['cosine_similarity']:.6f}")
        print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compare distance between trained and estimated models'
    )
    parser.add_argument(
        '--trained_model_path',
        type=str,
        required=True,
        help='Path to trained DPO model checkpoint'
    )
    parser.add_argument(
        '--precompute_file',
        type=str,
        required=True,
        help='Filename of precomputed gradients (in pre_compute directory)'
    )
    parser.add_argument(
        '--base_model_name',
        type=str,
        required=True,
        help='Name of the base model (e.g., Qwen/Qwen3-0.6B)'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='cache',
        help='Cache directory'
    )
    
    args = parser.parse_args()
    
    results = compute_model_distance(
        trained_model_path=args.trained_model_path,
        precompute_file=args.precompute_file,
        base_model_name=args.base_model_name,
        cache_dir=args.cache_dir,
        verbose=True,
    )
    
    return results


if __name__ == "__main__":
    main()
