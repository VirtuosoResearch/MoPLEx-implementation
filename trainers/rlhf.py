import os
# Set global temporary directory to ./cache/tmp so that all tmp files go there
TMP_ROOT = os.path.abspath("./cache/tmp")
os.makedirs(TMP_ROOT, exist_ok=True)
for var in ["TMPDIR", "TEMP", "TMP"]:
    os.environ[var] = TMP_ROOT
os.environ["WANDB__SERVICE_WAIT"] = "10000"
os.environ["WANDB_INIT_TIMEOUT"] = "10000"
os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_DIR"] = "./cache"
os.environ["WANDB_DATA_DIR"] = "./cache"
os.environ["WANDB_CACHE_DIR"] = "./cache"
os.environ["WANDB_TEMP"] = TMP_ROOT

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trainers.rlhf_trainer import RLHFTrainer
from trainers.dpo_config import DPOConfig
from trainers.model_value_head import AutoModelForCausalLMWithValueHead

import torch
import accelerate
import gc
import datetime
import numpy as np
import tempfile
from tqdm import tqdm
import wandb
from trainers.utils import (
    logprobs_from_logits,
    entropy_from_logits,
    save_model
)
import argparse
from trainers.data_loader import get_dataset, load_data, generation_kwargs, load_model

PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'
EOS_TOKEN = '<|endoftext|>'


def main(args):
    output_dir = os.path.join(args.output_dir, args.wandb_project, args.run_name)
    if args.use_lora: output_dir += "_lora"
    model_name = (args.pretrained_dir).split("/")[-1]
    print('Output dir:', output_dir, '\nModel name:', model_name)

    # Load dataset (for queries only, responses will be generated)
    query_dataset_dataloader, all_eval_dataloaders = load_data(args)

    unique_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f") + '-' + str(np.random.randint(100000))
    wandb_output_dir = tempfile.mkdtemp(dir=args.cache_dir)
    
    config = DPOConfig(
        model_name=args.pretrained_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lam=args.gae_lambda,
        cliprange=args.clip_range,
        cliprange_value=args.clip_range,
        batch_size=args.batch_size,
        dataloader_batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.inner_iteration_steps,
        tracker_project_name=args.wandb_project,
        gamma=getattr(args, "gamma", 1.0),
        vf_coef=getattr(args, "vf_coef", 0.1),
        whiten_rewards=getattr(args, "whiten_rewards", False),
        max_grad_norm=getattr(args, "max_grad_norm", None),
        project_kwargs={
            'project_dir': output_dir,
        },
        tracker_kwargs={
            "wandb": {
                "entity": getattr(args, "wandb_entity", "michaelzona"), 
                "name": args.run_name,
                "id": unique_str, 
                "dir": wandb_output_dir,
            }
        },
        log_with='wandb',
        seed=args.seed,
        adap_kl_ctrl=getattr(args, "adap_kl_ctrl", True),
        init_kl_coef=getattr(args, "init_kl_coef", 0.2),
        target=getattr(args, "target_kl", 6.0),
        horizon=getattr(args, "kl_horizon", 10000),
    )
    
    # Set max_length and max_new_tokens as attributes (not DPOConfig parameters)
    config.max_length = getattr(args, "max_length", 512)
    config.max_new_tokens = getattr(args, "max_new_tokens", 256)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    # Load policy model (with value head)
    model = load_model(tokenizer, args)
    
    # Load reward model if provided
    reward_model = None
    reward_model_pos_label_idx = None
    if hasattr(args, "reward_model_path") and args.reward_model_path:
        print(f"Loading reward model from {args.reward_model_path}")
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        reward_model.eval()
        
        # Determine positive label index for binary classification models
        # This is useful for sentiment models like lvwerra/distilbert-imdb
        if hasattr(reward_model.config, 'id2label') and reward_model.config.id2label:
            id2label = reward_model.config.id2label
            for idx, label in id2label.items():
                if "POS" in label.upper():
                    reward_model_pos_label_idx = int(idx)
                    break
            # If not found and it's binary classification, default to index 1
            if reward_model_pos_label_idx is None and len(id2label) == 2:
                reward_model_pos_label_idx = 1
        
        print(f"Reward model loaded (num_labels={reward_model.config.num_labels}, pos_label_idx={reward_model_pos_label_idx})")
    
    # Create trainer
    trainer = RLHFTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        reward_model=reward_model,
        reward_model_pos_label_idx=reward_model_pos_label_idx,
        additional_config_kwargs=vars(args),
    )
    
    def empty_cache():
        gc.collect()
        if args.use_tpu:
            return
        torch.cuda.empty_cache()
        gc.collect()

    def empty_cache_decorator(func):
        def func_wrapper(*args, **kwargs):
            empty_cache()
            return func(*args, **kwargs)
        return func_wrapper

    @empty_cache_decorator
    @torch.no_grad()
    def process_query_batch(query_batch):
        # Process query batch for RLHF
        if isinstance(query_batch["query"], list):
            queries = []
            for q in query_batch["query"]:
                # Tokenize query
                query_tokens = tokenizer(
                    q, 
                    padding=False, 
                    truncation=True, 
                    max_length=128, 
                    return_tensors='pt'
                ).input_ids.squeeze(0)
                queries.append(query_tokens)
        else:
            # Already tokenized
            queries = [q.squeeze(0) for q in query_batch["query"]]
            
        return queries

    print("Starting RLHF training")
    total_iterations = 0
    columns_to_log = ["query", "response"]

    for epoch in tqdm(range(args.num_train_epochs), desc="Epochs"):
        for sub_iteration, query_batch in tqdm(enumerate(query_dataset_dataloader), desc="Batches", total=len(query_dataset_dataloader)):
            empty_cache()
            stats = {}
            
            # Process queries
            queries = process_query_batch(query_batch)
            
            # Trainer step (generates responses, computes rewards, updates model)
            train_stats = trainer.step(queries=queries, generation_kwargs=generation_kwargs)
            for key in train_stats:
                stats[key] = train_stats[key]
            
            # Prepare output batch for logging
            output_batch = {
                "query": query_batch.get("query", [""] * len(queries)),
                "response": [""] * len(queries),  # Responses are generated internally
            }
            
            # Get rewards for logging (from last generated batch)
            rewards = train_stats.get("policy/rewards_mean", torch.tensor(0.0))
            if isinstance(rewards, torch.Tensor):
                rewards = [rewards.item()] * len(queries)
            else:
                rewards = [rewards] * len(queries)

            stats['epoch'] = epoch + sub_iteration / len(query_dataset_dataloader)
            stats['total_iterations'] = total_iterations
            stats['gradient_steps'] = total_iterations * args.inner_iteration_steps

            total_iterations += 1
            trainer.log_stats(
                stats=stats,
                batch=output_batch,
                rewards=rewards,
                columns_to_log=columns_to_log
            )
        
        # Evaluation step
        if "eval_pref" in all_eval_dataloaders:
            eval_metrics = trainer.evaluate(all_eval_dataloaders["eval_pref"])
            print(f"[Eval] evaluation : {eval_metrics}")
            trainer.log_stats(
                stats=eval_metrics,
                batch={"query": [], "response": []},
                rewards=torch.zeros(1),
                columns_to_log=columns_to_log,
            )
            
        if epoch % 10 == 0:
            save_model(trainer, output_dir, model, tokenizer, model_name + f"_epoch_{epoch}", epoch, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--pretrained_dir", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--reward_model_path", type=str, default=None, help="Path to reward model")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="", help="comma-separated module names to apply LoRA to; if empty, auto-detect by model_type")
    
    # Dataset arguments
    parser.add_argument("--preference_dataset_path", type=str, required=True, help="Path to preference dataset")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes")
    parser.add_argument("--batched", action="store_true", default=True, help="Use batched processing")
    parser.add_argument("--downsample_ratio", type=float, default=1.0, help="Downsample ratio for dataset")
    parser.add_argument("--preference_num_samples", type=int, default=-1, help="Number of samples to use (-1 for all)")
    parser.add_argument("--num_samples_test", type=int, default=None, help="Number of test samples")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory")
    parser.add_argument("--wandb_project", type=str, default="rlhf", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="michaelzona", help="W&B entity")
    parser.add_argument("--run_name", type=str, required=True, help="Run name")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=4, help="Mini batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--inner_iteration_steps", type=int, default=4, help="PPO epochs")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--vf_coef", type=float, default=0.1, help="Value function coefficient")
    parser.add_argument("--whiten_rewards", action="store_true", help="Whiten rewards")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Max gradient norm")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens to generate")
    
    # KL control
    parser.add_argument("--adap_kl_ctrl", action="store_true", default=True, help="Use adaptive KL control")
    parser.add_argument("--init_kl_coef", type=float, default=0.2, help="Initial KL coefficient")
    parser.add_argument("--target_kl", type=float, default=6.0, help="Target KL")
    parser.add_argument("--kl_horizon", type=float, default=10000, help="KL horizon")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_tpu", action="store_true", help="Use TPU")
    
    args = parser.parse_args()
    main(args)

