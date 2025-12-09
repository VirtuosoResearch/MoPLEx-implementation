"""
Script to compare approximation error between ApproxDPO and standard DPO.
Computes the difference in loss values for the same data.
"""

import os
import sys
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock tyro if not available (for DPOConfig)
try:
    import tyro
except ImportError:
    # Create a mock tyro module that supports generic syntax like Suppress[int]
    class MockSuppress:
        def __init__(self, *args, **kwargs):
            pass
        
        @classmethod
        def __class_getitem__(cls, item):
            # Support generic syntax: Suppress[int] -> returns the class itself
            return cls
    
    class MockConf:
        Suppress = MockSuppress
    
    class MockTyro:
        conf = MockConf
    
    import types
    tyro_module = types.ModuleType('tyro')
    tyro_module.conf = MockConf
    tyro_module.conf.Suppress = MockSuppress
    sys.modules['tyro'] = tyro_module
    sys.modules['tyro.conf'] = MockConf

from trainers.dpo_trainer import DPOTrainer
from trainers.approx_dpo_trainer import ApproxDPOTrainer
from trainers.dpo_config import DPOConfig
from trainers.data_loader import load_data, load_model
from trainers.utils import set_seed

def compute_approximation_error(
    model_name: str,
    preference_dataset_path: str,
    batch_size: int = 4,
    mini_batch_size: int = 1,
    temperature: float = 0.05,
    num_samples: int = 10,
    use_lora: bool = False,
    seed: int = 42,
    cache_dir: str = "cache",
):
    """
    Compute approximation error between ApproxDPO and standard DPO.
    
    Args:
        model_name: Name of the model to use
        preference_dataset_path: Path to preference dataset
        batch_size: Batch size for evaluation
        mini_batch_size: Mini batch size
        temperature: Temperature parameter
        num_samples: Number of samples to evaluate
        use_lora: Whether to use LoRA
        seed: Random seed
        cache_dir: Cache directory
        
    Returns:
        Dictionary with error statistics
    """
    set_seed(seed)
    
    # Setup accelerator
    accelerator = Accelerator(
        project_config=ProjectConfiguration(project_dir="./cache"),
    )
    
    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    model = load_model(tokenizer, argparse.Namespace(
        pretrained_dir=model_name,
        use_lora=use_lora,
        lora_r=8,
        lora_alpha=32.0,
        lora_dropout=0.05,
        lora_target_modules="",
    ))
    
    # Create config
    config = DPOConfig(
        model_name=model_name,
        gradient_accumulation_steps=1,
        learning_rate=1e-7,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        temperature=temperature,
        tracker_project_name="approx_error_comparison",
        log_with=None,  # Disable logging
        seed=seed,
    )
    
    # Create trainers
    # Note: Each trainer creates its own reference model, so we can use the same base model
    # But we need separate instances to avoid interference
    dpo_trainer = DPOTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
    )
    
    # Load a fresh model instance for approx trainer to avoid interference
    model_approx = load_model(tokenizer, argparse.Namespace(
        pretrained_dir=model_name,
        use_lora=use_lora,
        lora_r=8,
        lora_alpha=32.0,
        lora_dropout=0.05,
        lora_target_modules="",
    ))
    
    approx_dpo_trainer = ApproxDPOTrainer(
        model=model_approx,
        config=config,
        tokenizer=tokenizer,
    )
    
    # Load data
    print(f"Loading dataset: {preference_dataset_path}")
    args = argparse.Namespace(
        preference_dataset_path=preference_dataset_path,
        preference_dataset_subset=None,
        preference_dataset_split=None,
        downsample_ratio=1.0,
        batch_size=batch_size,
        cache_dir=cache_dir,
    )
    
    _, all_eval_dataloaders = load_data(args)
    eval_dataloader = all_eval_dataloaders.get("eval_pref", None)
    
    if eval_dataloader is None:
        raise ValueError("No eval_pref dataloader found")
    
    # Note: Trainers already prepare models in __init__, so we don't need to prepare again
    # But we need to ensure models are on the correct device
    device = accelerator.device
    dpo_trainer.current_device = device
    approx_dpo_trainer.current_device = device
    
    # Collect errors
    errors = []
    dpo_losses = []
    approx_losses = []
    
    print(f"Computing approximation error on {num_samples} samples...")
    
    for i, batch in enumerate(tqdm(eval_dataloader, total=min(num_samples, len(eval_dataloader)))):
        if i >= num_samples:
            break
            
        # Process batch
        preference_mask = batch.get("preference_mask", None)
        
        if isinstance(batch["query"], list) or isinstance(batch["query"], tuple):
            queries = tokenizer(
                batch["query"],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)
            
            all_pref = batch["response_w"] + batch["response_l"]
            tokenized = tokenizer(
                all_pref,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)
            
            n_w = len(batch["response_w"])
            responses_w = tokenized[:n_w]
            responses_l = tokenized[n_w:]
            
            if preference_mask is not None:
                preference_mask = torch.as_tensor(preference_mask, device=accelerator.device)
        else:
            queries = batch["query"].to(accelerator.device)
            responses_w = batch["response_w"].to(accelerator.device)
            responses_l = batch["response_l"].to(accelerator.device)
            if preference_mask is not None:
                preference_mask = preference_mask.to(accelerator.device)
        
        # Compute standard DPO loss
        dpo_trainer.model.eval()
        dpo_trainer.ref_model.eval()
        with torch.no_grad():
            dpo_loss, dpo_stats = dpo_trainer._step(
                queries=queries,
                responses_w=responses_w,
                responses_l=responses_l,
                preference_mask=preference_mask,
            )
            dpo_loss_val = dpo_loss.item()
            dpo_losses.append(dpo_loss_val)
        
        # Compute ApproxDPO loss
        # Note: ApproxDPO needs gradients for first-order term, but we detach the final loss
        approx_dpo_trainer.model.eval()
        approx_dpo_trainer.ref_model.eval()
        # We need gradients for first-order computation, but we'll detach the final result
        approx_loss, approx_stats = approx_dpo_trainer._step(
            queries=queries,
            responses_w=responses_w,
            responses_l=responses_l,
            preference_mask=preference_mask,
            compute_first_order=True,
            return_stats=True,
        )
        approx_loss_val = approx_loss.detach().item()
        approx_losses.append(approx_loss_val)
        
        # Clear any gradients that might have accumulated
        if approx_dpo_trainer.model.training:
            approx_dpo_trainer.model.zero_grad()
        
        # Compute error
        error = abs(dpo_loss_val - approx_loss_val)
        relative_error = error / (abs(dpo_loss_val) + 1e-8) * 100  # Percentage
        
        errors.append({
            'sample_idx': i,
            'dpo_loss': dpo_loss_val,
            'approx_loss': approx_loss_val,
            'absolute_error': error,
            'relative_error_pct': relative_error,
        })
    
    # Compute statistics
    errors_array = np.array([e['absolute_error'] for e in errors])
    relative_errors_array = np.array([e['relative_error_pct'] for e in errors])
    
    stats = {
        'model': model_name,
        'dataset': preference_dataset_path,
        'num_samples': len(errors),
        'mean_absolute_error': float(np.mean(errors_array)),
        'std_absolute_error': float(np.std(errors_array)),
        'max_absolute_error': float(np.max(errors_array)),
        'min_absolute_error': float(np.min(errors_array)),
        'mean_relative_error_pct': float(np.mean(relative_errors_array)),
        'std_relative_error_pct': float(np.std(relative_errors_array)),
        'mean_dpo_loss': float(np.mean(dpo_losses)),
        'mean_approx_loss': float(np.mean(approx_losses)),
    }
    
    return stats, errors


def main():
    parser = argparse.ArgumentParser(description='Compare approximation error between ApproxDPO and DPO')
    parser.add_argument('--model_names', type=str, nargs='+', 
                       default=['meta-llama/Llama-3.2-1B'],
                       help='List of model names to evaluate')
    parser.add_argument('--preference_dataset_path', type=str,
                       default='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_minlength',
                       help='Path to preference dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--mini_batch_size', type=int, default=1, help='Mini batch size')
    parser.add_argument('--temperature', type=float, default=0.05, help='Temperature parameter')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to evaluate')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Cache directory')
    
    args = parser.parse_args()
    
    all_stats = []
    
    for model_name in args.model_names:
        print(f"\n{'='*80}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*80}\n")
        
        try:
            stats, errors = compute_approximation_error(
                model_name=model_name,
                preference_dataset_path=args.preference_dataset_path,
                batch_size=args.batch_size,
                mini_batch_size=args.mini_batch_size,
                temperature=args.temperature,
                num_samples=args.num_samples,
                use_lora=args.use_lora,
                seed=args.seed,
                cache_dir=args.cache_dir,
            )
            
            all_stats.append(stats)
            
            print(f"\nResults for {model_name}:")
            print(f"  Mean Absolute Error: {stats['mean_absolute_error']:.6f}")
            print(f"  Mean Relative Error: {stats['mean_relative_error_pct']:.2f}%")
            print(f"  Mean DPO Loss: {stats['mean_dpo_loss']:.6f}")
            print(f"  Mean Approx Loss: {stats['mean_approx_loss']:.6f}")
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary table
    if all_stats:
        print(f"\n{'='*80}")
        print("Summary Table:")
        print(f"{'='*80}")
        
        # Print header
        print(f"{'Model':<40} {'Samples':<10} {'Mean Abs Error':<18} {'Mean Rel Error %':<18} {'Mean DPO Loss':<18} {'Mean Approx Loss':<18}")
        print("-" * 120)
        
        # Print each model's stats
        for stats in all_stats:
            print(f"{stats['model']:<40} "
                  f"{stats['num_samples']:<10} "
                  f"{stats['mean_absolute_error']:<18.6f} "
                  f"{stats['mean_relative_error_pct']:<18.2f} "
                  f"{stats['mean_dpo_loss']:<18.6f} "
                  f"{stats['mean_approx_loss']:<18.6f}")
        
        print(f"\n{'='*80}")
        print("Detailed Statistics:")
        print(f"{'='*80}")
        
        for stats in all_stats:
            print(f"\nModel: {stats['model']}")
            print(f"  Dataset: {stats['dataset']}")
            print(f"  Number of samples: {stats['num_samples']}")
            print(f"  Mean Absolute Error: {stats['mean_absolute_error']:.6f} ± {stats['std_absolute_error']:.6f}")
            print(f"  Max Absolute Error: {stats['max_absolute_error']:.6f}")
            print(f"  Min Absolute Error: {stats['min_absolute_error']:.6f}")
            print(f"  Mean Relative Error: {stats['mean_relative_error_pct']:.2f}% ± {stats['std_relative_error_pct']:.2f}%")
            print(f"  Mean DPO Loss: {stats['mean_dpo_loss']:.6f}")
            print(f"  Mean Approx Loss: {stats['mean_approx_loss']:.6f}")
    else:
        print("\nNo results to display. All models failed to evaluate.")


if __name__ == "__main__":
    main()
