"""
Train DPO on single-criteria and mixed-criteria datasets and evaluate accuracy.

This script:
1. Trains DPO on single-criteria datasets (one for each criterion)
2. Trains DPO on mixed-criteria dataset
3. Evaluates accuracy on test sets
"""

import os
import sys
import argparse
import torch
import numpy as np
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trainers.network_utils import AutoModelForCausalLMWithValueHead
from trainers.dpo_trainer import DPOTrainer
from trainers.dpo_config import DPOConfig
from trainers.data_loader import load_data, generation_kwargs
import accelerate

PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'
EOS_TOKEN = '<|endoftext|>'


def compute_accuracy(model, tokenizer, dataset, device, batch_size=8):
    """
    Compute accuracy: percentage of times the model prefers y_w over y_l.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
            batch = dataset[i:i+batch_size]
            
            prompts = batch["prompt"]
            y_w_list = batch["y_w"]
            y_l_list = batch["y_l"]
            
            # Tokenize query
            query_tensors = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)
            
            # Use original responses without splitting (matching training logic)
            response_w_tensors = tokenizer(
                y_w_list,
                padding=True,
                truncation=True,
                max_length=64 + generation_kwargs['max_new_tokens'],
                return_tensors="pt"
            ).to(device)
            
            response_l_tensors = tokenizer(
                y_l_list,
                padding=True,
                truncation=True,
                max_length=64 + generation_kwargs['max_new_tokens'],
                return_tensors="pt"
            ).to(device)
            
            # Compute log probabilities
            input_ids_w = torch.cat((query_tensors.input_ids, response_w_tensors.input_ids), dim=1)
            input_ids_l = torch.cat((query_tensors.input_ids, response_l_tensors.input_ids), dim=1)
            
            def get_logprob(input_ids):
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
                # Get model output - handle both wrapped and unwrapped models
                if hasattr(model, 'pretrained_model'):
                    outputs = model.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
                # Compute log probs
                from trainers.utils import logprobs_from_logits
                logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
                attn_mask_shifted = attention_mask[:, 1:]
                
                # Sum over sequence (excluding prompt)
                q_len = query_tensors.input_ids.size(1)
                mask = attn_mask_shifted.clone()
                if q_len > 1:
                    mask[:, :q_len-1] = 0
                
                seq_logprob = (logprobs * mask).sum(dim=1)
                return seq_logprob
            
            logprob_w = get_logprob(input_ids_w)
            logprob_l = get_logprob(input_ids_l)
            
            # Model prefers y_w if logprob_w > logprob_l
            predictions = (logprob_w > logprob_l).cpu().numpy()
            correct += predictions.sum()
            total += len(predictions)
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def compute_accuracy_by_criterion(model, tokenizer, dataset, device, batch_size=8):
    """
    Compute accuracy broken down by criterion (for mixed_criteria dataset).
    Returns overall accuracy and per-criterion accuracies.
    """
    if "criterion_used" not in dataset.column_names:
        return None, None
    
    model.eval()
    correct_by_criterion = {}
    total_by_criterion = {}
    overall_correct = 0
    overall_total = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating by criterion"):
            batch = dataset[i:i+batch_size]
            
            prompts = batch["prompt"]
            y_w_list = batch["y_w"]
            y_l_list = batch["y_l"]
            criteria = batch["criterion_used"]
            
            # Tokenize query
            query_tensors = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)
            
            response_w_tensors = tokenizer(
                y_w_list,
                padding=True,
                truncation=True,
                max_length=64 + generation_kwargs['max_new_tokens'],
                return_tensors="pt"
            ).to(device)
            
            response_l_tensors = tokenizer(
                y_l_list,
                padding=True,
                truncation=True,
                max_length=64 + generation_kwargs['max_new_tokens'],
                return_tensors="pt"
            ).to(device)
            
            input_ids_w = torch.cat((query_tensors.input_ids, response_w_tensors.input_ids), dim=1)
            input_ids_l = torch.cat((query_tensors.input_ids, response_l_tensors.input_ids), dim=1)
            
            def get_logprob(input_ids):
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
                if hasattr(model, 'pretrained_model'):
                    outputs = model.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
                from trainers.utils import logprobs_from_logits
                logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
                attn_mask_shifted = attention_mask[:, 1:]
                
                q_len = query_tensors.input_ids.size(1)
                mask = attn_mask_shifted.clone()
                if q_len > 1:
                    mask[:, :q_len-1] = 0
                
                seq_logprob = (logprobs * mask).sum(dim=1)
                return seq_logprob
            
            logprob_w = get_logprob(input_ids_w)
            logprob_l = get_logprob(input_ids_l)
            
            predictions = (logprob_w > logprob_l).cpu().numpy()
            
            for j, criterion in enumerate(criteria):
                if criterion not in correct_by_criterion:
                    correct_by_criterion[criterion] = 0
                    total_by_criterion[criterion] = 0
                
                if predictions[j]:
                    correct_by_criterion[criterion] += 1
                total_by_criterion[criterion] += 1
                overall_total += 1
                if predictions[j]:
                    overall_correct += 1
    
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
    per_criterion_accuracy = {
        crit: correct_by_criterion[crit] / total_by_criterion[crit] 
        if total_by_criterion[crit] > 0 else 0.0
        for crit in correct_by_criterion
    }
    
    return overall_accuracy, per_criterion_accuracy


def train_and_evaluate(
    dataset_path: str,
    model_name: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-6,
    device: str = "cuda",
    seed: int = 42,
    downsample_ratio: float = 0.25
):
    """
    Train DPO and evaluate accuracy.
    """
    print(f"\n{'='*60}")
    print(f"Training on: {dataset_path}")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")
    
    # Load dataset
    dataset_dict = load_from_disk(dataset_path)
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict.get("validation", None)
    test_dataset = dataset_dict["test"]
    
    # Downsample datasets
    if downsample_ratio < 1.0:
        train_size = int(len(train_dataset) * downsample_ratio)
        test_size = int(len(test_dataset) * downsample_ratio)
        
        train_dataset = train_dataset.shuffle(seed=seed).select(range(train_size))
        test_dataset = test_dataset.shuffle(seed=seed + 1).select(range(test_size))
        
        if val_dataset is not None:
            val_size = int(len(val_dataset) * downsample_ratio)
            val_dataset = val_dataset.shuffle(seed=seed + 2).select(range(val_size))
    
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    if val_dataset is not None:
        print(f"Val size: {len(val_dataset)}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = AutoModelForCausalLMWithValueHead(model)
    
    if device == "cpu":
        model = model.to(device)
    
    # Create DPO config
    config = DPOConfig(
        model_name=model_name,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=batch_size,
        ppo_epochs=1,
        tracker_project_name="dpo_criteria_experiment",
        use_score_scaling=False,
        use_score_norm=False,
        temperature=1.0,
        use_tpu=False,
        ipo_loss=False,
        project_kwargs={'project_dir': output_dir},
        tracker_kwargs={"wandb": {"name": os.path.basename(dataset_path)}},
        log_with=None,  # Disable wandb for this experiment
        seed=seed,
    )
    
    # Create trainer
    trainer = DPOTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
    )
    
    # Create dataloader
    from torch.utils.data import DataLoader
    def collate_fn(batch):
        return batch
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Evaluate before training
    print("\nEvaluating before training...")
    acc_before = compute_accuracy(model, tokenizer, test_dataset, device, batch_size=batch_size)
    print(f"Accuracy before training: {acc_before:.4f}")
    
    # For mixed_criteria dataset, also compute accuracy by criterion
    if "criterion_used" in test_dataset.column_names:
        print("\nAnalyzing accuracy by criterion (before training)...")
        overall_acc, per_crit_acc = compute_accuracy_by_criterion(model, tokenizer, test_dataset, device, batch_size=batch_size)
        if per_crit_acc:
            print(f"Overall accuracy: {overall_acc:.4f}")
            for crit, acc in sorted(per_crit_acc.items()):
                print(f"  {crit}: {acc:.4f} (n={sum(1 for x in test_dataset['criterion_used'] if x == crit)})")
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for batch in tqdm(train_dataloader, desc=f"Training epoch {epoch+1}"):
            # Process batch - map field names to match dpo.py format
            # Dataset uses "prompt", "y_w", "y_l", but we need "query", "response_w", "response_l"
            pref_batch = {
                "query": [item["prompt"] for item in batch],
                "response_w": [item["y_w"] for item in batch],
                "response_l": [item["y_l"] for item in batch],
            }
            
            # Tokenize query
            pref_query = tokenizer(
                pref_batch["query"],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).input_ids
            pref_query_tensors = accelerate.utils.send_to_device(pref_query, trainer.accelerator.device)
            
            # Tokenize responses together to ensure same length (matching dpo.py logic)
            all_pref = pref_batch["response_w"] + pref_batch["response_l"]
            tokenized = tokenizer(
                all_pref,
                padding=True,
                truncation=True,
                max_length=64 + generation_kwargs['max_new_tokens'],
                return_tensors="pt"
            ).input_ids
            
            pref_response_w_tensors = tokenized[:len(pref_batch["response_w"])]
            pref_response_w_tensors = accelerate.utils.send_to_device(pref_response_w_tensors, trainer.accelerator.device)
            pref_response_l_tensors = tokenized[len(pref_batch["response_w"]):]
            pref_response_l_tensors = accelerate.utils.send_to_device(pref_response_l_tensors, trainer.accelerator.device)
            
            # Train step
            stats = trainer.step(
                queries=pref_query_tensors,
                responses_w=pref_response_w_tensors,
                responses_l=pref_response_l_tensors,
            )
        
        # Evaluate after each epoch
        print(f"\nEvaluating after epoch {epoch + 1}...")
        acc_after = compute_accuracy(model, tokenizer, test_dataset, device, batch_size=batch_size)
        print(f"Accuracy after epoch {epoch + 1}: {acc_after:.4f}")
        
        # For mixed_criteria dataset, also compute accuracy by criterion
        if "criterion_used" in test_dataset.column_names:
            print(f"\nAnalyzing accuracy by criterion (after epoch {epoch + 1})...")
            overall_acc, per_crit_acc = compute_accuracy_by_criterion(model, tokenizer, test_dataset, device, batch_size=batch_size)
            if per_crit_acc:
                print(f"Overall accuracy: {overall_acc:.4f}")
                for crit, acc in sorted(per_crit_acc.items()):
                    print(f"  {crit}: {acc:.4f}")
    
    return {
        "dataset": os.path.basename(dataset_path),
        "accuracy_before": acc_before,
        "accuracy_after": acc_after,
        "improvement": acc_after - acc_before,
    }


def main():
    parser = argparse.ArgumentParser(description="Train DPO on criteria-based datasets")
    parser.add_argument("--datasets_dir", type=str, default="synthetic_datasets", 
                       help="Directory containing synthetic datasets")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B",
                       help="Base model name")
    parser.add_argument("--output_dir", type=str, default="dpo_criteria_results",
                       help="Output directory for results")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--downsample_ratio", type=float, default=0.25, help="Downsample ratio for train/val/test (default: 0.25)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    results = []
    
    # Train on single-criteria datasets
    criteria = ["helpfulness", "truthfulness", "instruction_following", "honesty"]
    for criterion in criteria:
        dataset_path = os.path.join(args.datasets_dir, f"single_{criterion}")
        if os.path.exists(dataset_path):
            result = train_and_evaluate(
                dataset_path=dataset_path,
                model_name=args.model_name,
                output_dir=args.output_dir,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
                seed=args.seed,
                downsample_ratio=args.downsample_ratio,
            )
            results.append(result)
        else:
            print(f"Warning: Dataset not found: {dataset_path}")
    
    # Train on mixed-criteria dataset
    mixed_dataset_path = os.path.join(args.datasets_dir, "mixed_criteria")
    if os.path.exists(mixed_dataset_path):
        result = train_and_evaluate(
            dataset_path=mixed_dataset_path,
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=args.seed,
            downsample_ratio=args.downsample_ratio,
        )
        results.append(result)
    else:
        print(f"Warning: Dataset not found: {mixed_dataset_path}")
    
    # Save results
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for result in results:
        print(f"\n{result['dataset']}:")
        print(f"  Accuracy before: {result['accuracy_before']:.4f}")
        print(f"  Accuracy after:  {result['accuracy_after']:.4f}")
        print(f"  Improvement:     {result['improvement']:.4f}")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

