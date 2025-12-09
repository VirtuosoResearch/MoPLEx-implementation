#!/usr/bin/env python3
"""
Compare trained models: ApproxDPO vs Standard DPO
Computes relative error in logprobs and DPO loss on test data.
"""

import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trainers.utils import logprobs_from_logits
from trainers.dpo_config import DPOConfig
from trainers.dpo_trainer import DPOTrainer
from trainers.approx_dpo_trainer import ApproxDPOTrainer

PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'
EOS_TOKEN = '<|endoftext|>'


def robust_load_dataset(dataset_path, split):
    """Load and process dataset."""
    pref_dataset = load_dataset(dataset_path, split=split)
    remove_columns = ['output', 'text', 'alpaca_text', 'y_ref', 'y_1', 'y_2', 'y_w', 'y_w_alpaca', 'y_l', 'y_l_alpaca', 'y_w_score', 'y_l_score', 'score_diff', 'prompt', 'alpaca_prompt']

    def process_dataset(batch):
        new_batch = {}
        new_batch['query'] = batch['prompt']
        new_batch['text_w'] = batch['y_w'] 
        new_batch['text_l'] = batch['y_l']
        new_batch['response_w'] = [x.split(ASSISTANT_TOKEN)[-1] if ASSISTANT_TOKEN in x else x for x in batch['y_w']]
        new_batch['response_l'] = [x.split(ASSISTANT_TOKEN)[-1] if ASSISTANT_TOKEN in x else x for x in batch['y_l']]
        return new_batch
    
    pref_dataset = pref_dataset.map(
        process_dataset,
        batched=True,
        num_proc=1,
    )
    return pref_dataset


def compute_logprobs(model, tokenizer, query, response, device):
    """Compute average log probability of response given query."""
    # Tokenize query and response
    query_ids = tokenizer(query, return_tensors="pt", add_special_tokens=False).input_ids[0]
    response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids[0]
    input_ids = torch.cat([query_ids, response_ids], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Compute logprobs for response tokens
    response_start = query_ids.shape[0]
    log_probs = []
    for i in range(response_start, input_ids.shape[1]):
        token_id = input_ids[0, i]
        prev_logits = logits[0, i - 1]
        log_prob = torch.log_softmax(prev_logits, dim=-1)[token_id]
        log_probs.append(log_prob.item())
    
    if len(log_probs) == 0:
        return 0.0
    return sum(log_probs) / len(log_probs)


def compute_dpo_loss(model, tokenizer, query, response_w, response_l, beta, device):
    """Compute DPO loss for a single example."""
    logprob_w = compute_logprobs(model, tokenizer, query, response_w, device)
    logprob_l = compute_logprobs(model, tokenizer, query, response_l, device)
    
    # DPO loss: -log(sigmoid(beta * (logprob_w - logprob_l)))
    logratio = beta * (logprob_w - logprob_l)
    loss = -torch.log(torch.sigmoid(torch.tensor(logratio))).item()
    return loss, logprob_w, logprob_l


def compare_models(
    approx_model_path,
    standard_model_path,
    dataset_path,
    split="test",
    beta=0.05,
    max_samples=None,
    base_model_name=None,
):
    """Compare two trained models on the same dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer (use base model if provided, otherwise use one of the model paths)
    if base_model_name:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(approx_model_path, use_fast=True)
    
    # Load models (handle LoRA adapters if present)
    print(f"Loading approximate model from: {approx_model_path}")
    try:
        from peft import PeftModel
        if base_model_name and os.path.exists(os.path.join(approx_model_path, "adapter_config.json")):
            # Try loading as LoRA adapter
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            approx_model = PeftModel.from_pretrained(base_model, approx_model_path)
            print("  Loaded as LoRA adapter")
        else:
            approx_model = AutoModelForCausalLM.from_pretrained(approx_model_path)
            print("  Loaded as full model")
    except (ImportError, Exception) as e:
        # Fallback if peft is not available or loading fails
        approx_model = AutoModelForCausalLM.from_pretrained(approx_model_path)
        print(f"  Loaded as full model (fallback: {e})")
    approx_model = approx_model.to(device)
    approx_model.eval()
    
    print(f"Loading standard model from: {standard_model_path}")
    try:
        from peft import PeftModel
        if base_model_name and os.path.exists(os.path.join(standard_model_path, "adapter_config.json")):
            # Try loading as LoRA adapter
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            standard_model = PeftModel.from_pretrained(base_model, standard_model_path)
            print("  Loaded as LoRA adapter")
        else:
            standard_model = AutoModelForCausalLM.from_pretrained(standard_model_path)
            print("  Loaded as full model")
    except (ImportError, Exception) as e:
        # Fallback if peft is not available or loading fails
        standard_model = AutoModelForCausalLM.from_pretrained(standard_model_path)
        print(f"  Loaded as full model (fallback: {e})")
    standard_model = standard_model.to(device)
    standard_model.eval()
    
    # Load dataset
    print(f"Loading dataset: {dataset_path} (split: {split})")
    dataset = robust_load_dataset(dataset_path, split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    # Collect statistics
    approx_logprobs_w = []
    approx_logprobs_l = []
    standard_logprobs_w = []
    standard_logprobs_l = []
    approx_losses = []
    standard_losses = []
    
    for example in tqdm(dataset, desc="Computing logprobs"):
        query = example['query']
        response_w = example['response_w']
        response_l = example['response_l']
        
        # Compute for approximate model
        approx_logprob_w = compute_logprobs(approx_model, tokenizer, query, response_w, device)
        approx_logprob_l = compute_logprobs(approx_model, tokenizer, query, response_l, device)
        approx_loss, _, _ = compute_dpo_loss(approx_model, tokenizer, query, response_w, response_l, beta, device)
        
        # Compute for standard model
        standard_logprob_w = compute_logprobs(standard_model, tokenizer, query, response_w, device)
        standard_logprob_l = compute_logprobs(standard_model, tokenizer, query, response_l, device)
        standard_loss, _, _ = compute_dpo_loss(standard_model, tokenizer, query, response_w, response_l, beta, device)
        
        approx_logprobs_w.append(approx_logprob_w)
        approx_logprobs_l.append(approx_logprob_l)
        standard_logprobs_w.append(standard_logprob_w)
        standard_logprobs_l.append(standard_logprob_l)
        approx_losses.append(approx_loss)
        standard_losses.append(standard_loss)
    
    # Convert to numpy arrays
    approx_logprobs_w = np.array(approx_logprobs_w)
    approx_logprobs_l = np.array(approx_logprobs_l)
    standard_logprobs_w = np.array(standard_logprobs_w)
    standard_logprobs_l = np.array(standard_logprobs_l)
    approx_losses = np.array(approx_losses)
    standard_losses = np.array(standard_losses)
    
    # Compute differences
    logprob_w_diff = approx_logprobs_w - standard_logprobs_w
    logprob_l_diff = approx_logprobs_l - standard_logprobs_l
    logratio_diff = (approx_logprobs_w - approx_logprobs_l) - (standard_logprobs_w - standard_logprobs_l)
    loss_diff = approx_losses - standard_losses
    
    # Compute relative errors
    def safe_relative_error(numerator, denominator, name):
        """Compute relative error safely handling zeros."""
        abs_denom = np.abs(denominator)
        # Use a small epsilon to avoid division by zero
        epsilon = 1e-8
        relative_error = np.abs(numerator) / (abs_denom + epsilon)
        # For cases where denominator is very small, use absolute error
        mask = abs_denom < epsilon
        relative_error[mask] = np.abs(numerator[mask])
        return relative_error
    
    rel_error_logprob_w = safe_relative_error(logprob_w_diff, standard_logprobs_w, "logprob_w")
    rel_error_logprob_l = safe_relative_error(logprob_l_diff, standard_logprobs_l, "logprob_l")
    rel_error_logratio = safe_relative_error(logratio_diff, standard_logprobs_w - standard_logprobs_l, "logratio")
    rel_error_loss = safe_relative_error(loss_diff, standard_losses, "loss")
    
    # Print results
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(f"\nDataset: {dataset_path} ({split})")
    print(f"Number of samples: {len(dataset)}")
    print(f"Beta (temperature): {beta}")
    
    print("\n" + "-"*80)
    print("LOGPROB STATISTICS")
    print("-"*80)
    print(f"\n{'Metric':<30} {'Approx Model':<20} {'Standard Model':<20} {'Difference':<20} {'Rel Error %':<15}")
    print("-"*105)
    
    print(f"{'Mean logprob_w':<30} {np.mean(approx_logprobs_w):<20.6f} {np.mean(standard_logprobs_w):<20.6f} {np.mean(logprob_w_diff):<20.6f} {np.mean(rel_error_logprob_w)*100:<15.2f}")
    print(f"{'Mean logprob_l':<30} {np.mean(approx_logprobs_l):<20.6f} {np.mean(standard_logprobs_l):<20.6f} {np.mean(logprob_l_diff):<20.6f} {np.mean(rel_error_logprob_l)*100:<15.2f}")
    print(f"{'Mean logratio (w-l)':<30} {np.mean(approx_logprobs_w - approx_logprobs_l):<20.6f} {np.mean(standard_logprobs_w - standard_logprobs_l):<20.6f} {np.mean(logratio_diff):<20.6f} {np.mean(rel_error_logratio)*100:<15.2f}")
    
    print("\n" + "-"*80)
    print("DPO LOSS STATISTICS")
    print("-"*80)
    print(f"\n{'Metric':<30} {'Approx Model':<20} {'Standard Model':<20} {'Difference':<20} {'Rel Error %':<15}")
    print("-"*105)
    print(f"{'Mean DPO loss':<30} {np.mean(approx_losses):<20.6f} {np.mean(standard_losses):<20.6f} {np.mean(loss_diff):<20.6f} {np.mean(rel_error_loss)*100:<15.2f}")
    print(f"{'Std DPO loss':<30} {np.std(approx_losses):<20.6f} {np.std(standard_losses):<20.6f} {np.std(loss_diff):<20.6f} {np.std(rel_error_loss)*100:<15.2f}")
    
    print("\n" + "-"*80)
    print("DETAILED RELATIVE ERROR STATISTICS")
    print("-"*80)
    print(f"\n{'Metric':<30} {'Mean Rel Error %':<20} {'Median Rel Error %':<20} {'Max Rel Error %':<20} {'Min Rel Error %':<20}")
    print("-"*110)
    print(f"{'logprob_w':<30} {np.mean(rel_error_logprob_w)*100:<20.2f} {np.median(rel_error_logprob_w)*100:<20.2f} {np.max(rel_error_logprob_w)*100:<20.2f} {np.min(rel_error_logprob_w)*100:<20.2f}")
    print(f"{'logprob_l':<30} {np.mean(rel_error_logprob_l)*100:<20.2f} {np.median(rel_error_logprob_l)*100:<20.2f} {np.max(rel_error_logprob_l)*100:<20.2f} {np.min(rel_error_logprob_l)*100:<20.2f}")
    print(f"{'logratio (w-l)':<30} {np.mean(rel_error_logratio)*100:<20.2f} {np.median(rel_error_logratio)*100:<20.2f} {np.max(rel_error_logratio)*100:<20.2f} {np.min(rel_error_logratio)*100:<20.2f}")
    print(f"{'DPO loss':<30} {np.mean(rel_error_loss)*100:<20.2f} {np.median(rel_error_loss)*100:<20.2f} {np.max(rel_error_loss)*100:<20.2f} {np.min(rel_error_loss)*100:<20.2f}")
    
    print("\n" + "="*80)
    
    return {
        'approx_logprobs_w': approx_logprobs_w,
        'approx_logprobs_l': approx_logprobs_l,
        'standard_logprobs_w': standard_logprobs_w,
        'standard_logprobs_l': standard_logprobs_l,
        'approx_losses': approx_losses,
        'standard_losses': standard_losses,
        'rel_error_logprob_w': rel_error_logprob_w,
        'rel_error_logprob_l': rel_error_logprob_l,
        'rel_error_logratio': rel_error_logratio,
        'rel_error_loss': rel_error_loss,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare trained ApproxDPO and Standard DPO models")
    parser.add_argument("--approx_model_path", type=str, required=True,
                       help="Path to trained ApproxDPO model")
    parser.add_argument("--standard_model_path", type=str, required=True,
                       help="Path to trained Standard DPO model")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to evaluation dataset")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to use")
    parser.add_argument("--beta", type=float, default=0.05,
                       help="Temperature/beta parameter for DPO")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--base_model_name", type=str, default=None,
                       help="Base model name for tokenizer (if different from model paths)")
    
    args = parser.parse_args()
    
    compare_models(
        approx_model_path=args.approx_model_path,
        standard_model_path=args.standard_model_path,
        dataset_path=args.dataset_path,
        split=args.split,
        beta=args.beta,
        max_samples=args.max_samples,
        base_model_name=args.base_model_name,
    )


if __name__ == "__main__":
    main()
