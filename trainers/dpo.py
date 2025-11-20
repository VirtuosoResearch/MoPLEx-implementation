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
from trainers.dpo_trainer import DPOTrainer
from trainers.dpo_config import DPOConfig
from trainers.approx_dpo_trainer import ApproxDPOTrainer

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

    batch_size_online_data = 0

    pref_dataset_dataloader, all_eval_dataloaders = load_data(args)

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
    model = load_model(tokenizer, args)
    print(args.pretrained_dir)

    TrainerClass = ApproxDPOTrainer if args.approx_dpo else DPOTrainer
    trainer = TrainerClass(
        model=model,
        config=config,
        tokenizer=tokenizer,
        additional_config_kwargs=vars(args),
    )

    def empty_cache():
        gc.collect()
        if args.use_tpu:
            return
        torch.cuda.empty_cache()
        gc.collect()
    def empty_cache_decorator(func):
        empty_cache()
        return func

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

        return pref_batch, pref_query_tensors, pref_response_w_tensors, pref_response_l_tensors

    @empty_cache_decorator
    @torch.no_grad()
    def process_input_ids(input_ids):
        input_data = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        logits, _, _ = trainer.model(**input_data)
        old_logits, _, _ = trainer.ref_model(**input_data)
        old_logprobs = logprobs_from_logits(old_logits[:, :-1, :], input_ids[:, 1:])
        logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
        entropy = entropy_from_logits(logits)

        return logprobs, old_logprobs, entropy, logits

    @empty_cache_decorator
    @torch.no_grad()
    def process_batch_dpo(batch):
        query_tensors = tokenizer(batch["query"], padding='max_length' if args.use_tpu else True, truncation=True, max_length=128, return_tensors='pt')
        query_tensors = accelerate.utils.send_to_device(query_tensors, trainer.accelerator.device)

        # Get generations from SFTModel (including prompt)
        all_generation_tokens = []
        for _ in range(args.num_actions_per_prompt):  # generate multiple completions per prompt
            if query_tensors.input_ids.shape[0] > args.max_gen_batch_size:
                generation_tokens = []
                for i in tqdm(range(0, query_tensors.input_ids.shape[0], args.max_gen_batch_size), desc=f"Generating for epoch {epoch}"):
                    generation_tokens.append(trainer.accelerator.unwrap_model(trainer.model).generate(**query_tensors[i:i + args.max_gen_batch_size], **generation_kwargs))
                    torch.cuda.empty_cache()
                generation_tokens = torch.cat(generation_tokens, dim=0)
            else:
                generation_tokens = trainer.accelerator.unwrap_model(trainer.model).generate(**query_tensors, **generation_kwargs)
            all_generation_tokens.append(generation_tokens)
        all_generation_tokens = torch.cat(all_generation_tokens, dim=0)

        logprobs, old_logprobs, entropy, logits = process_input_ids(all_generation_tokens)
        texts = tokenizer.batch_decode(all_generation_tokens, skip_special_tokens=True)
        batch["response"] = [x.split(ASSISTANT_TOKEN)[-1] for x in texts]

        response_tensors = tokenizer(batch["response"], padding='max_length' if args.use_tpu else True, truncation=True, max_length=generation_kwargs['max_new_tokens'], return_tensors='pt').input_ids
        response_tensors = accelerate.utils.send_to_device(response_tensors, trainer.accelerator.device)

        query_tensors = query_tensors.input_ids
        # Ensure query and response tensors are same length
        query_tensors = query_tensors.repeat(args.num_actions_per_prompt, 0)
        assert query_tensors.shape[0] == response_tensors.shape[0], f"query_tensors.shape[0] = {query_tensors.shape[0]} != response_tensors.shape[0] = {response_tensors.shape[0]}"
        batch["query"] = batch["query"] * args.num_actions_per_prompt

        return batch, query_tensors, response_tensors, logprobs, old_logprobs, entropy, logits

    print("Starting training")
    total_iterations = 0
    columns_to_log = ["query", "response_w", "response_l"]
    columns_to_log_eval = ["query", "response"]

    for epoch in tqdm(range(args.num_train_epochs), desc="Epochs"):
        for sub_iteration, pref_batch in tqdm(enumerate(pref_dataset_dataloader), desc="Batches", total=len(pref_dataset_dataloader)):
            empty_cache()

            stats = {}
            if total_iterations % args.eval_every_steps == 0:
                # Eval
                for eval_name, eval_dataloader in all_eval_dataloaders.items():
                    print(f"Running evaluation on {eval_name}")
                    eval_batch = next(iter(eval_dataloader))
                    eval_batch, query_tensors, response_tensors, logprobs, old_logprobs, entropy, logits = process_batch_dpo(eval_batch)
                    all_to_log = {}
                    for k in columns_to_log_eval:
                        all_to_log[k] = (all_to_log.get(k, []) + [x.cpu().numpy().item() if isinstance(x, torch.Tensor) else x for x in eval_batch[k]])
                    del eval_batch, query_tensors, response_tensors
                    empty_cache()
                    stats[f"{eval_name}/entropy"] = entropy.mean().item()
                    stats[f"{eval_name}/logprobs"] = logprobs.mean().item() # policy model
                    stats[f"{eval_name}/old_logprobs"] = old_logprobs.mean().item() # reference model
                    stats[f"{eval_name}/approxkl"] = (0.5 * ((logprobs - old_logprobs) ** 2).mean()).item()
                    stats[f"{eval_name}/policykl"] = (logprobs - old_logprobs).mean().item()
                    stats[f"{eval_name}/sequence_approxkl"] = (0.5 * ((logprobs - old_logprobs) ** 2)).sum(-1).mean().item()
                    stats[f"{eval_name}/sequence_policykl"] = (logprobs - old_logprobs).sum(-1).mean().item()
                    # log table of completions
                    table_rows = list(r for r in zip(*[all_to_log[col] for col in columns_to_log_eval]))
                    stats[f"{eval_name}/table"] = wandb.Table(columns=[*columns_to_log_eval], rows=table_rows)

                empty_cache()

            pref_batch, pref_query_tensors, pref_response_w_tensors, pref_response_l_tensors = process_pref_batch(pref_batch)

            output_batch = {k: pref_batch[k] for k in columns_to_log}
            # Trainer step
            train_stats = trainer.step(queries=pref_query_tensors, responses_w=pref_response_w_tensors, responses_l=pref_response_l_tensors)
            for key in train_stats:
                stats[key] = train_stats[key]

            rewards = torch.zeros(pref_query_tensors.shape[0], dtype=torch.float32)
            rewards = accelerate.utils.send_to_device(rewards, trainer.accelerator.device)

            stats['epoch'] = epoch + sub_iteration / len(pref_dataset_dataloader)
            stats['total_iterations'] = total_iterations
            stats['gradient_steps'] = total_iterations * args.inner_iteration_steps

            if stats['total_iterations'] % args.save_every_steps == 0:
                num_batches = stats['total_iterations']
                save_model(trainer, output_dir, model, tokenizer, model_name + f"_num_batches_{num_batches}", epoch, args)

            total_iterations += 1
            trainer.log_stats(
                stats=stats,
                batch=output_batch,
                rewards=rewards,
                columns_to_log=columns_to_log
            )

        trainer.end_of_epoch_step(epoch)
        save_model(trainer, output_dir, model, tokenizer, model_name + f"_epoch_{num_batches}", epoch, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str, default='reweighted_bc', help='the wandb project name')
    parser.add_argument('--run_name', type=str, default='reweighted_bc', help='the wandb run name')
    parser.add_argument('--output_dir', type=str, default=None, help='the output directory')
    parser.add_argument('--tokenizer_type', type=str, default="EleutherAI/pythia-1.4b', help='the model name")
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

    parser.add_argument('--use_lora', type=bool, default=False, help='whether to wrap the policy with LoRA adapters')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=32.0, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, default='', help='comma-separated module names to apply LoRA to; if empty, auto-detect by model_type')
    parser.add_argument('--lora_merge_on_save', type=bool, default=False, help='merge LoRA weights into base weights when saving checkpoints')

    args = parser.parse_args()
    main(args)