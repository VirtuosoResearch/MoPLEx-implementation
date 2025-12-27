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
from trainers.maxmin_dpo_trainer import MaxMinDPOTrainer, MaxMinDPOConfig, EMRewardLearner
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
    entropy_from_logits,
    save_model
)
import argparse
from trainers.data_loader import get_dataset, load_data, generation_kwargs, load_model

PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'
EOS_TOKEN = '<|endoftext|>'


def prepare_preference_data_for_em(dataset):
    """
    Prepare preference data for EM algorithm from dataset.
    Returns a list of dicts with 'prompt', 'chosen', 'rejected', 'user_id' keys.
    
    For datasets without explicit user_id, we create synthetic user_ids based on
    a hash of the prompt (same prompt = same user) to group similar preferences.
    """
    preference_data = []
    
    for idx, item in enumerate(dataset):
        # Extract fields - handle different dataset formats
        prompt = item.get('prompt', item.get('query', ''))
        
        # Handle IMDB format: responses list with chosen index
        if 'responses' in item and 'chosen' in item:
            responses = item['responses']
            chosen_idx = item['chosen']
            chosen = responses[chosen_idx] if isinstance(responses, list) and len(responses) > chosen_idx else item.get('response_w', '')
            rejected = responses[1 - chosen_idx] if isinstance(responses, list) and len(responses) > (1 - chosen_idx) else item.get('response_l', '')
        else:
            # Standard format
            chosen = item.get('response_w', item.get('chosen', ''))
            rejected = item.get('response_l', item.get('rejected', ''))
        
        # Get user_id/annotator - create synthetic ID if not available
        user_id = item.get('user_id', item.get('annotator', item.get('source', None)))
        
        # If no explicit user_id, create one based on prompt hash
        # This groups samples with the same prompt together
        if user_id is None:
            # Use hash of prompt to create consistent user_id
            user_id = hash(str(prompt)) % (2**31)
        elif isinstance(user_id, str):
            # Convert string to int via hash
            user_id = hash(user_id) % (2**31)
        elif not isinstance(user_id, int):
            user_id = hash(str(prompt)) % (2**31)
        
        preference_data.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'user_id': user_id,
        })
    
    return preference_data


def main(args):
    output_dir = os.path.join(args.output_dir, args.wandb_project, args.run_name)
    if args.use_lora: output_dir += "_lora"
    model_name = (args.pretrained_dir).split("/")[-1]
    print('Output dir:', output_dir, '\nModel name:', model_name)

    batch_size_online_data = 0

    # Load preference dataset
    pref_dataset_dataloader, all_eval_dataloaders = load_data(args)

    unique_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f") + '-' + str(np.random.randint(100000))
    wandb_output_dir = tempfile.mkdtemp(dir=args.cache_dir)
    
    # Use MaxMinDPOConfig instead of DPOConfig
    config = MaxMinDPOConfig(
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
        # MaxMin-DPO specific parameters
        num_clusters=getattr(args, 'num_clusters', 2),
        maxmin_weight=getattr(args, 'maxmin_weight', 0.1),
        reward_model_name=getattr(args, 'reward_model_name', None),
        em_convergence_threshold=getattr(args, 'em_convergence_threshold', 1e-4),
        em_max_iterations=getattr(args, 'em_max_iterations', 20),
        reward_learning_rate=getattr(args, 'reward_learning_rate', 1e-5),
        reward_num_epochs=getattr(args, 'reward_num_epochs', 3),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_dir)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    generation_kwargs["pad_token_id"]=tokenizer.eos_token_id
    model = load_model(tokenizer, args)
    print(args.pretrained_dir)

    # Step 1: Learn reward models using EM algorithm
    print("\n=== Step 1: Learning Reward Models with EM Algorithm ===")
    
    # Get the full dataset for EM (before downsampling)
    # Use the same loading logic as load_data function - check local directory first
    if os.path.isdir(args.preference_dataset_path) and os.path.exists(
        os.path.join(args.preference_dataset_path, "dataset_dict.json")
    ):
        # Local directory with DatasetDict (e.g., imdb_synthetic/mixed_criteria, ultrafeedback)
        from datasets import load_from_disk
        pref_dataset_full = load_from_disk(args.preference_dataset_path)
        train_dataset_for_em = pref_dataset_full['train']
    elif "imdb" in args.preference_dataset_path.lower() and not os.path.isdir(args.preference_dataset_path):
        # IMDB from HuggingFace (not local directory, e.g., "ZHZisZZ/imdb_preference")
        pref_dataset_full = load_dataset(args.preference_dataset_path)
        # Process IMDB format
        def make_imdb_pref(batch):
            prompts = batch["prompt"]
            all_responses = batch["responses"]
            chosens = batch["chosen"]
            y_w_list = []
            y_l_list = []
            for resp_list, c in zip(all_responses, chosens):
                win = resp_list[c]
                lose = resp_list[1 - c]
                y_w_list.append(win)
                y_l_list.append(lose)
            return {
                "prompt": prompts,
                "response_w": y_w_list,
                "response_l": y_l_list,
            }
        
        pref_dataset_full = pref_dataset_full.map(
            make_imdb_pref,
            batched=True,
            num_proc=args.num_proc,
        )
        train_dataset_for_em = pref_dataset_full['train']
    else:
        # Other datasets (from HuggingFace or other sources)
        pref_dataset_full = load_dataset(args.preference_dataset_path)
        train_dataset_for_em = pref_dataset_full['train']
    
    # Downsample for EM if needed (can use more data for EM)
    em_downsample_ratio = getattr(args, 'em_downsample_ratio', None)
    if em_downsample_ratio is None:
        em_downsample_ratio = args.downsample_ratio
    if em_downsample_ratio < 1.0:
        train_dataset_for_em = train_dataset_for_em.shuffle(seed=args.seed).select(
            range(int(len(train_dataset_for_em) * em_downsample_ratio))
        )
    
    # Prepare preference data for EM
    preference_data = prepare_preference_data_for_em(train_dataset_for_em)
    print(f"Prepared {len(preference_data)} preference examples for EM algorithm")
    
    # Initialize EM learner
    reward_model_name = getattr(args, 'reward_model_name', None) or "distilbert-base-uncased"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    em_learner = EMRewardLearner(
        num_clusters=config.num_clusters,
        reward_model_name=reward_model_name,
        tokenizer=tokenizer,
        device=device,
        convergence_threshold=config.em_convergence_threshold,
        max_iterations=config.em_max_iterations,
        reward_learning_rate=config.reward_learning_rate,
        reward_num_epochs=config.reward_num_epochs,
    )
    
    # Run EM algorithm
    reward_models = em_learner.fit(preference_data)
    user_assignments = em_learner.user_assignments
    
    print(f"EM algorithm completed. User assignments: {len(user_assignments)} users assigned to {config.num_clusters} clusters")
    for cluster_id in range(config.num_clusters):
        count = sum(1 for uid, cid in user_assignments.items() if cid == cluster_id)
        print(f"  Cluster {cluster_id}: {count} users")

    # Step 2: Train with MaxMin-DPO
    print("\n=== Step 2: Training with MaxMin-DPO ===")
    
    trainer = MaxMinDPOTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        reward_models=reward_models,
        user_assignments=user_assignments,
        num_clusters=config.num_clusters,
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
        
        # Extract user_ids for MaxMin-DPO
        # Use the same method as in EM: hash of prompt to create consistent user_id
        user_ids = []
        batch_size = len(pref_batch["query"])
        
        # Generate user_ids based on prompt hash (same as EM algorithm)
        # This ensures consistency with the EM clustering
        for i in range(batch_size):
            prompt = pref_batch["query"][i]
            # Use same hash method as in prepare_preference_data_for_em
            user_id = hash(str(prompt)) % (2**31)
            user_ids.append(user_id)
        
        user_ids_tensor = torch.tensor(user_ids, device=trainer.accelerator.device, dtype=torch.long)

        return pref_batch, pref_query_tensors, pref_response_w_tensors, pref_response_l_tensors, user_ids_tensor

    print("Starting MaxMin-DPO training")
    total_iterations = 0
    columns_to_log = ["query", "response_w", "response_l"]

    for epoch in tqdm(range(args.num_train_epochs), desc="Epochs"):
        for sub_iteration, pref_batch in tqdm(enumerate(pref_dataset_dataloader), desc="Batches", total=len(pref_dataset_dataloader)):
            empty_cache()
            stats = {}
            pref_batch, pref_query_tensors, pref_response_w_tensors, pref_response_l_tensors, user_ids_tensor = process_pref_batch(pref_batch)
            output_batch = {k: pref_batch[k] for k in columns_to_log}

            # Trainer step with user_ids
            train_stats = trainer.step(
                queries=pref_query_tensors, 
                responses_w=pref_response_w_tensors, 
                responses_l=pref_response_l_tensors,
                user_ids=user_ids_tensor,
            )
            for key in train_stats:
                stats[key] = train_stats[key]

            rewards = torch.zeros(pref_query_tensors.shape[0], dtype=torch.float32)
            rewards = accelerate.utils.send_to_device(rewards, trainer.accelerator.device)

            stats['epoch'] = epoch + sub_iteration / len(pref_dataset_dataloader)
            stats['total_iterations'] = total_iterations
            stats['gradient_steps'] = total_iterations * args.inner_iteration_steps

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
            save_model(trainer, output_dir, model, tokenizer, model_name + f"_epoch_{epoch}", epoch, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str, default='maxmin_dpo', help='the wandb project name')
    parser.add_argument('--run_name', type=str, default='maxmin_dpo', help='the wandb run name')
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
    
    # MaxMin-DPO specific arguments
    parser.add_argument('--num_clusters', type=int, default=2, help='number of user clusters for MaxMin-DPO')
    parser.add_argument('--maxmin_weight', type=float, default=0.1, help='weight for MaxMin objective')
    parser.add_argument('--reward_model_name', type=str, default=None, help='base model name for reward models (default: distilbert-base-uncased)')
    parser.add_argument('--em_convergence_threshold', type=float, default=1e-4, help='convergence threshold for EM algorithm')
    parser.add_argument('--em_max_iterations', type=int, default=20, help='maximum number of EM iterations')
    parser.add_argument('--reward_learning_rate', type=float, default=1e-5, help='learning rate for reward model training')
    parser.add_argument('--reward_num_epochs', type=int, default=3, help='number of epochs for reward model training')
    parser.add_argument('--em_downsample_ratio', type=float, default=None, help='downsample ratio for EM algorithm (default: same as downsample_ratio)')

    args = parser.parse_args()
    main(args)

