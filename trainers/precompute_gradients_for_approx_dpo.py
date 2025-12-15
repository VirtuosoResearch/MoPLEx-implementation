"""
Script to precompute gradients and b values at reference model for ApproxDPO.
According to paper A.1, we need to compute:
- g = ∇_θ h_θ0(x, y1, y2) = β(∇_θ log π_θ0(y_w|x) - ∇_θ log π_θ0(y_l|x))
- b = -h_θ0(x, y1, y2)
"""

import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock tyro if not available
try:
    import tyro
except ImportError:
    class MockSuppress:
        @classmethod
        def __class_getitem__(cls, item):
            return cls
    class MockConf:
        Suppress = MockSuppress
    import types
    tyro_module = types.ModuleType('tyro')
    tyro_module.conf = MockConf
    sys.modules['tyro'] = tyro_module
    sys.modules['tyro.conf'] = MockConf

from transformers import AutoTokenizer
from trainers.approx_dpo_trainer import ApproxDPOTrainer
from trainers.dpo_config import DPOConfig
from trainers.data_loader import load_data, load_model
from trainers.utils import set_seed
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

def main(args):
    set_seed(args.seed)
    output_path = os.path.join("pre_compute", f"{args.model_name.split('/')[-1]}_{args.projection_dim}_{args.downsample_ratio}.pt")
    # Setup accelerator
    accelerator = Accelerator(
        project_config=ProjectConfiguration(project_dir="./cache"),
    )
    
    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    model = load_model(tokenizer, argparse.Namespace(
        pretrained_dir=args.model_name,
        cache_dir=args.cache_dir,
        use_lora=args.use_lora,
        lora_r=8,
        lora_alpha=32.0,
        lora_dropout=0.05,
        lora_target_modules="",
    ))
    
    # Create config
    config = DPOConfig(
        model_name=args.model_name,
        gradient_accumulation_steps=1,
        learning_rate=1e-7,
        batch_size=args.batch_size,
        mini_batch_size=args.batch_size,
        temperature=args.temperature,
        tracker_project_name="precompute_gradients",
        log_with=None,
        seed=args.seed,
        projection_dim=args.projection_dim,
    )
    
    # Create trainer
    trainer = ApproxDPOTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
    )
    
    # Prepare with accelerator
    trainer.model, trainer.ref_model = accelerator.prepare(
        trainer.model, trainer.ref_model
    )
    trainer.current_device = accelerator.device
    
    # Load data
    print(f"Loading dataset: {args.preference_dataset_path}")
    data_args = argparse.Namespace(
        preference_dataset_path=args.preference_dataset_path,
        preference_dataset_subset=None,
        preference_dataset_split=None,
        preference_num_samples=None,
        downsample_ratio=args.downsample_ratio,
        batch_size=args.batch_size,
        mini_batch_size=args.batch_size,
        seed=args.seed,
        cache_dir=args.cache_dir,
        batched=True,
        num_proc=32,
    )
    
    pref_dataset_dataloader, all_eval_dataloaders = load_data(data_args)
    
    # Use training dataset for precomputation (same as training)
    # Ensure projection matrix is generated if projection is enabled
    if config.projection_dim is not None:
        trainer._ensure_projection_matrix()
        print(f"Projection enabled: {trainer.projection_matrix.shape[0]} -> {trainer.projection_matrix.shape[1]}")
    
    # Precompute gradients and (optionally) z values
    # Note: b = 0 (since model == ref_model at θ0), so we don't compute or store it
    print("Precomputing gradients at reference model (theta_0) on training dataset...")
    all_gradients = []
    all_z_values = []  # z \in {+1, -1}. For standard DPO, z = +1 (winner preferred).
    
    num_samples = 0
    for batch_idx, batch in enumerate(tqdm(pref_dataset_dataloader, desc="Precomputing")):
        if args.max_samples is not None and num_samples >= args.max_samples:
            break
            
        # Process batch
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
        else:
            queries = batch["query"].to(accelerator.device)
            responses_w = batch["response_w"].to(accelerator.device)
            responses_l = batch["response_l"].to(accelerator.device)
        
        # Precompute for this batch (only gradients, b = 0)
        batch_gradients = trainer.precompute_gradients_and_b(
            queries=queries,
            responses_w=responses_w,
            responses_l=responses_l,
        )
        
        all_gradients.extend(batch_gradients)
        # For standard DPO, each (query, response_w, response_l) has label z = +1
        all_z_values.extend([1.0] * len(batch_gradients))
        num_samples += len(batch_gradients)
        
        if args.max_samples is not None and num_samples >= args.max_samples:
            break
    
    # Store in trainer
    trainer.precomputed_gradients = all_gradients
    trainer.precomputed_z_values = np.array(all_z_values, dtype=np.float32)
    
    # Save to disk
    trainer.save_precomputed_gradients_and_b(output_path)
    
    print(f"\nPrecomputation complete!")
    print(f"  Total samples: {len(all_gradients)}")
    if len(all_gradients) > 0:
        print(f"  Gradient dimension: {len(all_gradients[0])}")
        print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Precompute gradients and b values for ApproxDPO')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B',
                       help='Model name')
    parser.add_argument('--preference_dataset_path', type=str,
                       default='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_minlength',
                       help='Path to preference dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for precomputation')
    parser.add_argument('--temperature', type=float, default=0.05, help='Temperature parameter')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Cache directory')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to precompute (None for all)')
    parser.add_argument('--projection_dim', type=int, default=None,
                       help='Dimension of the projected space for gradients (Johnson-Lindenstrauss). If None, no projection is applied.')
    parser.add_argument('--downsample_ratio', type=float, default=1.0,
                       help='Downsample ratio for the dataset. If 1.0, use the full dataset')
    args = parser.parse_args()
    main(args)
