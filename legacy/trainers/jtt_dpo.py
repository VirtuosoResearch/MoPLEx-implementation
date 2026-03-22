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
from transformers import AutoTokenizer
from trainers.dpo_trainer import DPOTrainer
from trainers.jtt_dpo_trainer import JTTDPOTrainer
from trainers.dpo_config import DPOConfig

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
    save_model
)
import argparse
from trainers.data_loader import load_data, generation_kwargs, load_model

PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'
EOS_TOKEN = '<|endoftext|>'


def identify_error_examples(model, tokenizer, dataset, config, device, max_steps=100):
    """
    Stage 1: Identify error examples using a limited-capacity model.
    
    Error examples are those where the model prefers rejected over chosen:
    log π_θ(y_w|x) < log π_θ(y_l|x)
    
    Returns:
        Set of indices of error examples
    """
    print(f"\n=== Stage 1: Identifying error examples (training for {max_steps} steps) ===")
    
    # Create a trainer for identification
    id_config = DPOConfig(
        model_name=config.model_name,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        mini_batch_size=config.mini_batch_size,
        temperature=config.temperature,
        use_tpu=config.use_tpu,
        ipo_loss=config.ipo_loss,
        seed=config.seed,
    )
    
    # Create identification trainer (will train the model for limited steps)
    id_trainer = DPOTrainer(
        model=model,
        config=id_config,
        tokenizer=tokenizer,
    )
    
    # Create dataloader for identification training
    from torch.utils.data import DataLoader
    id_dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    
    # Train identification model for limited steps
    id_trainer.model.train()
    steps_trained = 0
    
    for epoch in range(1000):  # Large range, will break when max_steps reached
        if steps_trained >= max_steps:
            break
        
        for pref_batch in id_dataloader:
            if steps_trained >= max_steps:
                break
            
            # Skip empty batches
            if len(pref_batch["query"]) == 0:
                continue
            
            # Process batch
            pref_query = tokenizer(pref_batch["query"], padding='max_length' if config.use_tpu else True, truncation=True, max_length=128, return_tensors='pt').input_ids
            pref_query_tensors = accelerate.utils.send_to_device(pref_query, device)
            
            # Validate batch size after tokenization
            if pref_query_tensors.shape[0] == 0:
                continue
            
            all_pref = pref_batch["response_w"] + pref_batch["response_l"]
            tokenized = tokenizer(all_pref, padding='max_length' if config.use_tpu else True, truncation=True, max_length=64 + generation_kwargs['max_new_tokens'], return_tensors='pt').input_ids
            
            # Validate tokenized batch size
            if tokenized.shape[0] == 0:
                continue
            
            pref_response_w_tensors = tokenized[:len(pref_batch["response_w"])]
            pref_response_w_tensors = accelerate.utils.send_to_device(pref_response_w_tensors, device)
            pref_response_l_tensors = tokenized[len(pref_batch["response_w"]):]
            pref_response_l_tensors = accelerate.utils.send_to_device(pref_response_l_tensors, device)
            
            # Validate response tensors batch sizes
            if pref_response_w_tensors.shape[0] == 0 or pref_response_l_tensors.shape[0] == 0:
                continue
            
            # Ensure all tensors have the same batch size
            if pref_query_tensors.shape[0] != pref_response_w_tensors.shape[0] or pref_query_tensors.shape[0] != pref_response_l_tensors.shape[0]:
                continue
            
            # Train step
            id_trainer.step(
                queries=pref_query_tensors,
                responses_w=pref_response_w_tensors,
                responses_l=pref_response_l_tensors,
            )
            
            steps_trained += 1
    
    print(f"Identification model trained for {steps_trained} steps")
    
    # Now identify error examples
    print("Identifying error examples...")
    id_trainer.model.eval()
    error_indices = set()
    
    # Process dataset in batches
    with torch.no_grad():
        batch_size = config.batch_size
        for i in tqdm(range(0, len(dataset), batch_size), desc="Identifying errors"):
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            
            # Skip empty batches
            if len(batch) == 0:
                continue
            
            # Extract fields from dataset as lists
            # HuggingFace datasets: batch["column"] returns a list of all values in that column
            try:
                batch_queries = list(batch["query"])
                batch_response_w = list(batch["response_w"])
                batch_response_l = list(batch["response_l"])
            except (TypeError, KeyError) as e:
                # Fallback: extract field by field
                batch_queries = [batch[j]["query"] for j in range(len(batch))]
                batch_response_w = [batch[j]["response_w"] for j in range(len(batch))]
                batch_response_l = [batch[j]["response_l"] for j in range(len(batch))]
            
            if len(batch_queries) == 0:
                continue
            
            pref_query = tokenizer(batch_queries, padding='max_length' if config.use_tpu else True, truncation=True, max_length=128, return_tensors='pt').input_ids
            pref_query_tensors = accelerate.utils.send_to_device(pref_query, device)
            
            # Validate batch size after tokenization
            if pref_query_tensors.shape[0] == 0:
                continue
            
            all_pref = batch_response_w + batch_response_l
            tokenized = tokenizer(all_pref, padding='max_length' if config.use_tpu else True, truncation=True, max_length=64 + generation_kwargs['max_new_tokens'], return_tensors='pt').input_ids
            
            # Validate tokenized batch size
            if tokenized.shape[0] == 0:
                continue
            
            pref_response_w_tensors = tokenized[:len(batch_response_w)]
            pref_response_w_tensors = accelerate.utils.send_to_device(pref_response_w_tensors, device)
            pref_response_l_tensors = tokenized[len(batch_response_w):]
            pref_response_l_tensors = accelerate.utils.send_to_device(pref_response_l_tensors, device)
            
            # Validate response tensors batch sizes
            if pref_response_w_tensors.shape[0] == 0 or pref_response_l_tensors.shape[0] == 0:
                continue
            
            # Ensure all tensors have the same batch size
            if pref_query_tensors.shape[0] != pref_response_w_tensors.shape[0] or pref_query_tensors.shape[0] != pref_response_l_tensors.shape[0]:
                continue
            
            # Compute log probabilities
            input_ids_w = torch.cat((pref_query_tensors, pref_response_w_tensors), dim=1)
            input_ids_l = torch.cat((pref_query_tensors, pref_response_l_tensors), dim=1)
            pad_id = tokenizer.pad_token_id
            
            def process_input_ids(input_ids):
                attention_mask = (input_ids != pad_id).long()
                input_data = {"input_ids": input_ids, "attention_mask": attention_mask}
                logits, _, _ = id_trainer.model(**input_data)
                logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
                attn_mask_shifted = attention_mask[:, 1:]
                return logprobs, attn_mask_shifted
            
            logprobs_w, attn_shift_w = process_input_ids(input_ids_w)
            logprobs_l, attn_shift_l = process_input_ids(input_ids_l)
            
            q_len = pref_query_tensors.size(1)
            mask_w = attn_shift_w.clone()
            mask_l = attn_shift_l.clone()
            if q_len > 1:
                mask_w[:, :q_len - 1] = 0
                mask_l[:, :q_len - 1] = 0
            
            seq_logprob_w = (logprobs_w * mask_w).sum(dim=1)  # [B]
            seq_logprob_l = (logprobs_l * mask_l).sum(dim=1)  # [B]
            
            # Error: model prefers rejected (y_l) over chosen (y_w)
            errors = (seq_logprob_w < seq_logprob_l).cpu().numpy()
            
            for idx_in_batch, is_error in enumerate(errors):
                if is_error:
                    error_indices.add(i + idx_in_batch)
    
    print(f"Identified {len(error_indices)} error examples out of {len(dataset)} total")
    return error_indices


def main(args):
    output_dir = os.path.join(args.output_dir, args.wandb_project, args.run_name)
    if args.use_lora: output_dir += "_lora"
    model_name = (args.pretrained_dir).split("/")[-1]
    print('Output dir:', output_dir, '\nModel name:', model_name)

    batch_size_online_data = 0

    pref_dataset_dataloader, all_eval_dataloaders = load_data(args)
    
    # Get the dataset from dataloader for error identification
    # The dataset already has query, response_w, response_l fields from load_data processing
    train_dataset_for_id = pref_dataset_dataloader.dataset

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
        dataloader_batch_size=max(batch_size_online_data, 1),
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.inner_iteration_steps,
        tracker_project_name=args.wandb_project,
        use_score_scaling=args.use_score_scaling,
        use_score_norm=args.use_score_norm,
        temperature=args.temperature,
        use_tpu=args.use_tpu,
        ipo_loss=args.ipo_loss,
        project_kwargs={
            'project_dir': output_dir,
        },
        tracker_kwargs={
            "wandb": {
                "entity": "michaelzona", "name": args.run_name,
                "id": unique_str, "dir": wandb_output_dir,
            }
        },
        log_with='wandb',
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_dir)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    generation_kwargs["pad_token_id"]=tokenizer.eos_token_id
    
    # Stage 1: Identify error examples
    id_model = load_model(tokenizer, args)
    device = next(id_model.parameters()).device
    
    identification_steps = getattr(args, 'jtt_identification_steps', 100)
    error_indices = identify_error_examples(
        id_model, tokenizer, train_dataset_for_id, config, device, max_steps=identification_steps
    )
    
    # Clean up identification model
    del id_model
    gc.collect()
    if not args.use_tpu:
        torch.cuda.empty_cache()
    
    # Stage 2: Train final model with upweighted error examples
    print("\n=== Stage 2: Training final model with upweighted error examples ===")
    
    final_model = load_model(tokenizer, args)
    
    trainer = JTTDPOTrainer(
        model=final_model,
        config=config,
        tokenizer=tokenizer,
        additional_config_kwargs=vars(args),
    )
    
    upweight_factor = getattr(args, 'jtt_upweight_factor', 3.0)
    
    def empty_cache():
        gc.collect()
        if args.use_tpu:
            return
        torch.cuda.empty_cache()
        gc.collect()
    
    def empty_cache_decorator(func):
        empty_cache()
        return func

    # Create a set of error query hashes for matching (since indices may not match after shuffling)
    # Use query text as identifier
    error_query_set = set()
    for idx in error_indices:
        if idx < len(train_dataset_for_id):
            query = train_dataset_for_id[idx].get('query', train_dataset_for_id[idx].get('prompt', ''))
            error_query_set.add(str(query))
    
    @empty_cache_decorator
    @torch.no_grad()
    def process_pref_batch(pref_batch):
        # Process preference dataset
        pref_query = tokenizer(pref_batch["query"], padding='max_length' if args.use_tpu else True, truncation=True, max_length=128, return_tensors='pt').input_ids
        pref_query_tensors = accelerate.utils.send_to_device(pref_query, trainer.accelerator.device)

        # Tokenize together to be the same length
        all_pref = pref_batch["response_w"] + pref_batch["response_l"]
        tokenized = tokenizer(all_pref, padding='max_length' if args.use_tpu else True, truncation=True, max_length=64 + generation_kwargs['max_new_tokens'], return_tensors='pt').input_ids

        pref_response_w_tensors = tokenized[:len(pref_batch["response_w"])]
        pref_response_w_tensors = accelerate.utils.send_to_device(pref_response_w_tensors, trainer.accelerator.device)
        pref_response_l_tensors = tokenized[len(pref_batch["response_w"]):]
        pref_response_l_tensors = accelerate.utils.send_to_device(pref_response_l_tensors, trainer.accelerator.device)
        
        # Create preference_mask for upweighting error examples (match by query text)
        batch_size = len(pref_batch["query"])
        preference_mask = torch.ones(batch_size, device=trainer.accelerator.device, dtype=torch.float32)
        
        for i in range(batch_size):
            query = str(pref_batch["query"][i])
            if query in error_query_set:
                preference_mask[i] = upweight_factor

        return pref_batch, pref_query_tensors, pref_response_w_tensors, pref_response_l_tensors, preference_mask

    print("Starting final DPO training with upweighted error examples")
    total_iterations = 0
    columns_to_log = ["query", "response_w", "response_l"]

    for epoch in tqdm(range(args.num_train_epochs), desc="Epochs"):
        for sub_iteration, pref_batch in tqdm(enumerate(pref_dataset_dataloader), desc="Batches", total=len(pref_dataset_dataloader)):
            empty_cache()
            stats = {}
            pref_batch, pref_query_tensors, pref_response_w_tensors, pref_response_l_tensors, preference_mask = process_pref_batch(pref_batch)
            output_batch = {k: pref_batch[k] for k in columns_to_log}

            # Trainer step with preference_mask for upweighting (float tensor)
            train_stats = trainer.step(
                queries=pref_query_tensors,
                responses_w=pref_response_w_tensors,
                responses_l=pref_response_l_tensors,
                preference_mask=preference_mask,  # Float tensor for upweighting
            )
            for key in train_stats:
                stats[key] = train_stats[key]

            rewards = torch.zeros(pref_query_tensors.shape[0], dtype=torch.float32)
            rewards = accelerate.utils.send_to_device(rewards, trainer.accelerator.device)

            stats['epoch'] = epoch + sub_iteration / len(pref_dataset_dataloader)
            stats['total_iterations'] = total_iterations
            stats['gradient_steps'] = total_iterations * args.inner_iteration_steps
            stats['jtt/num_error_examples'] = len(error_indices)
            stats['jtt/upweight_factor'] = upweight_factor

            total_iterations += 1
            trainer.log_stats(
                stats=stats,
                batch=output_batch,
                rewards=rewards,
                columns_to_log=columns_to_log
            )
        trainer.end_of_epoch_step(epoch)
        
        # Evaluation step
        eval_metrics = trainer.evaluate(all_eval_dataloaders["eval_pref"])
        print(f"[Eval] evaluation : {eval_metrics}")
        trainer.log_stats(
            stats=eval_metrics,
            batch={"query": [], "response_w": [], "response_l": []},
            rewards=torch.zeros(1),
            columns_to_log=["query", "response_w", "response_l"],
        )
        if epoch % 10==0:
            save_model(trainer, output_dir, final_model, tokenizer, model_name + f"_epoch_{epoch}", epoch, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str, default='jtt_dpo', help='the wandb project name')
    parser.add_argument('--run_name', type=str, default='jtt_dpo', help='the wandb run name')
    parser.add_argument('--output_dir', type=str, default=None, help='the output directory')
    parser.add_argument('--tokenizer_type', type=str, default="EleutherAI/pythia-1.4b", help='the model name')
    parser.add_argument('--pretrained_dir', type=str, default="", help='the path to the pretrained model')
    parser.add_argument('--learning_rate', type=float, default=1.0e-6, help='the learning rate')
    parser.add_argument('--cosine_annealing_lr_eta_min', type=float, default=1.0e-7, help='the cosine annealing eta min')
    parser.add_argument('--num_train_epochs', type=int, default=5, help='the number of training epochs')
    parser.add_argument('--inner_iteration_steps', type=int, default=1, help='the number of training epochs')
    parser.add_argument('--eval_every_steps', type=int, default=10, help='how often to evaluate')
    parser.add_argument('--save_every_steps', type=int, default=1000, help='how often to save checkpoints')
    parser.add_argument('--num_eval_batches', type=int, default=8, help='the number of evaluation batches of size gold shard size')
    parser.add_argument('--downsample_ratio', type=float, default=1.0, help='the downsample ratio for the dataset, 1.0 means no downsampling')
    parser.add_argument('--clip_range', type=float, default=0.2, help='the clip range')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='the GAE lambda')
    parser.add_argument('--batch_size', type=int, default=64, help='the batch size')
    parser.add_argument('--max_gen_batch_size', type=int, default=8, help='the max generation batch size')
    parser.add_argument('--mini_batch_size', type=int, default=8, help='the chunk size')
    parser.add_argument('--seed', type=int, default=42, help='the random seed')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='the gradient accumulation steps')
    parser.add_argument('--use_score_scaling', type=bool, default=False, help='whether to use score scaling')
    parser.add_argument('--use_score_norm', type=bool, default=False, help='whether to use score normalization')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature for reweighting')

    parser.add_argument('--preference_dataset_path', type=str, default='tatsu-lab/alpaca_farm', help='the path to the preference dataset')
    parser.add_argument('--preference_dataset_subset', type=str, default='alpaca_human_preference', help='Dataset name')
    parser.add_argument('--preference_dataset_split', type=str, default='preference', help='Dataset name')
    parser.add_argument('--preference_num_samples', type=int, default=19000, help='the number of samples to use from the preference dataset')
    parser.add_argument('--batched', type=bool, default=True, help='Whether to use batched processing')
    parser.add_argument('--num_proc', type=int, default=32, help='Number of processes to use')
    parser.add_argument('--mixing_ratio', type=float, default=0.5, help='the mixing ratio for preference dataset')

    parser.add_argument('--num_actions_per_prompt', type=int, default=1, help='the number of actions per prompt for generation')
    parser.add_argument('--cache_dir', type=str, default='', help='the cache directory')

    parser.add_argument('--ipo_loss', type=bool, default=False, help='whether to use ipo loss')
    parser.add_argument('--use_tpu', type=bool, default=False, help='whether to use tpus')
    parser.add_argument('--approx_dpo', type=bool, default=False, help='whether to use approx dpo')

    parser.add_argument('--save_pstar_at_epoch', type=int, default=-1, help='the epoch after which to save p_star and grad_star. Set to -1 to disable.')
    parser.add_argument('--pstar_save_path', type=str, default='./pstar_grads.pt', help='the file path to save p_star and grad_star')
    parser.add_argument('--precomputed_gradients_path', type=str, default=None, help='Path to precomputed gradients and b values for ApproxDPO logistic regression training')

    parser.add_argument('--use_lora', type=bool, default=False, help='whether to wrap the policy with LoRA adapters')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=32.0, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, default='', help='comma-separated module names to apply LoRA to; if empty, auto-detect by model_type')
    parser.add_argument('--lora_merge_on_save', type=bool, default=False, help='merge LoRA weights into base weights when saving checkpoints')
    
    # JTT-DPO specific arguments
    parser.add_argument('--jtt_identification_steps', type=int, default=100, help='number of steps for identification model (Stage 1)')
    parser.add_argument('--jtt_upweight_factor', type=float, default=3.0, help='upweighting factor for error examples (λ_up)')
    parser.add_argument('--jtt_id_downsample_ratio', type=float, default=None, help='downsample ratio for identification stage (default: same as downsample_ratio)')

    args = parser.parse_args()
    main(args)

