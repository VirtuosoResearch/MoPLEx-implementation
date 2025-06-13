import argparse
import random
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import autocast

from dpo_cache_utils import generate_ref_cache, save_ref_cache, load_ref_cache, get_delta_theta


import wandb
from tqdm import tqdm

def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# def calculate_DPO_loss(model_preferred_logprob, model_dispreferred_logprob,
#                        ref_preferred_logprob, ref_dispreferred_logprob,
#                        beta=0.5):

#     preferred_relative_logprob = model_preferred_logprob - ref_preferred_logprob
#     dispreferred_relative_logprob = model_dispreferred_logprob - ref_dispreferred_logprob

#     reward_accuracies = (preferred_relative_logprob > dispreferred_relative_logprob).float().mean()
#     reward_margins = (preferred_relative_logprob - dispreferred_relative_logprob).mean()

#     loss = -F.logsigmoid(beta * (preferred_relative_logprob - dispreferred_relative_logprob)).mean()

#     return loss, preferred_relative_logprob.mean(), dispreferred_relative_logprob.mean(), reward_accuracies, reward_margins

# def get_log_prob(logits, labels, prompt_lengths):
#     prompt_lengths = prompt_lengths.to(labels.device)
    
#     log_probs = F.log_softmax(logits, dim=-1)
#     token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    
#     batch_size, seq_len = labels.shape
#     response_mask = torch.arange(seq_len, device=labels.device).unsqueeze(0) >= prompt_lengths.unsqueeze(1)
#     response_mask = response_mask.float()
    
#     response_log_probs = (token_log_probs * response_mask).sum(dim=-1)
#     response_lengths = response_mask.sum(dim=-1).clamp(min=1)
#     return response_log_probs / response_lengths

def collate_fn(batch, tokenizer, max_length, device):
    prompt_encodings = tokenizer(
        ['Instruct: ' + item['prompt'] + '\n' for item in batch],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    chosen_encodings = tokenizer(
        ['Output: ' + item['chosen'] for item in batch],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    rejected_encodings = tokenizer(
        ['Output: ' + item['rejected'] for item in batch],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    prompt_preferred_ids = torch.cat([
        prompt_encodings.input_ids,
        chosen_encodings.input_ids
    ], dim=-1).to(device)
    
    prompt_dispreferred_ids = torch.cat([
        prompt_encodings.input_ids,
        rejected_encodings.input_ids
    ], dim=-1).to(device)

    prompt_preferred_mask = torch.cat([
        prompt_encodings.attention_mask,
        chosen_encodings.attention_mask
    ], dim=-1).to(device)
    
    prompt_dispreferred_mask = torch.cat([
        prompt_encodings.attention_mask,
        rejected_encodings.attention_mask
    ], dim=-1).to(device)

    prompt_lengths = prompt_encodings.attention_mask.sum(dim=-1)

    # assign an id for each sample
    sample_ids = list(range(len(batch)))

    return {
        'prompt_preferred_ids': prompt_preferred_ids,
        'prompt_dispreferred_ids': prompt_dispreferred_ids,
        'prompt_preferred_mask': prompt_preferred_mask,
        'prompt_dispreferred_mask': prompt_dispreferred_mask,
        'prompt_lengths': prompt_lengths, 
        'sample_ids': torch.tensor(sample_ids, device=device)
    }

def train(model, ref_model, tokenizer, optimizer, train_dataloader, epochs=1, beta=0.1, 
          loss_type='exact', delta_theta=None, ref_cache=None):
    model.train()
    ref_model.eval()

    for epoch in range(epochs):
        for batch in tqdm(train_dataloader):
            print(batch.keys())
            optimizer.zero_grad()

            model_preferred_logits = model(
                input_ids=batch['prompt_preferred_ids'],
                attention_mask=batch['prompt_preferred_mask']
            ).logits
            
            model_preferred_logprob = get_log_prob(
                model_preferred_logits,
                batch['prompt_preferred_ids'],
                batch['prompt_lengths']
            )

            model_dispreferred_logits = model(
                input_ids=batch['prompt_dispreferred_ids'],
                attention_mask=batch['prompt_dispreferred_mask']
            ).logits
            
            model_dispreferred_logprob = get_log_prob(
                model_dispreferred_logits,
                batch['prompt_dispreferred_ids'],
                batch['prompt_lengths']
            )

            if loss_type == 'exact':
                with torch.no_grad():
                    ref_preferred_logits = ref_model(
                        input_ids=batch['prompt_preferred_ids'],
                        attention_mask=batch['prompt_preferred_mask']
                    ).logits
                    
                    ref_preferred_logprob = get_log_prob(
                        ref_preferred_logits,
                        batch['prompt_preferred_ids'],
                        batch['prompt_lengths']
                    )

                    ref_dispreferred_logits = ref_model(
                        input_ids=batch['prompt_dispreferred_ids'],
                        attention_mask=batch['prompt_dispreferred_mask']
                    ).logits
                    
                    ref_dispreferred_logprob = get_log_prob(
                        ref_dispreferred_logits,
                        batch['prompt_dispreferred_ids'],
                        batch['prompt_lengths']
                    )

                loss, preferred_relative_logprob, dispreferred_relative_logprob, reward_accuracies, reward_margins = calculate_DPO_loss(
                    model_preferred_logprob,
                    model_dispreferred_logprob,
                    ref_preferred_logprob,
                    ref_dispreferred_logprob,
                    beta=beta
                )

            elif loss_type == 'approx':
                assert delta_theta is not None and ref_cache is not None, \
                    "delta_theta and ref_cache are required for approximate DPO."

                reward_diff_star = []
                grad_diff = []

                for sample_id in batch['sample_ids'].tolist():
                    r_pref = ref_cache[f"{sample_id}_preferred"]
                    r_dispref = ref_cache[f"{sample_id}_dispreferred"]

                    reward_diff_star.append(r_pref['logp'] - r_dispref['logp'])
                    grad_diff.append(r_pref['grad'] - r_dispref['grad'])

                reward_diff_star = torch.tensor(reward_diff_star, device=delta_theta.device)
                grad_diff = torch.stack(grad_diff).to(delta_theta.device)

                loss, reward_accuracies, reward_margins = approximate_dpo_loss_cached(
                    reward_diff_star, grad_diff, delta_theta, beta
                )
                preferred_relative_logprob = reward_diff_star + beta * (grad_diff @ delta_theta)
                dispreferred_relative_logprob = torch.zeros_like(preferred_relative_logprob)  # dummy

            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            loss.backward()
            optimizer.step()

            wandb.log({
                'loss': loss.item(),
                'preferred_relative_logprob': preferred_relative_logprob.item(),
                'dispreferred_relative_logprob': dispreferred_relative_logprob.item(),
                'reward_accuracy': reward_accuracies.item(),
                'reward_margin': reward_margins.item()
            })

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--dataset_name", type=str, default="jondurbin/truthy-dpo-v0.1")
    parser.add_argument("--wandb_project", type=str, default="truthy-dpo")
    parser.add_argument("--dpo_type", type=str, choices=["exact", "approx"], default="exact", help="Choose DPO loss type: 'exact' or 'approx'")
    parser.add_argument("--ref_cache_path", type=str, default=None, help="Path to cached reference gradients (required if dpo_type == 'approx')")

    args = parser.parse_args()

    seed_everything(args.seed)

    wandb.login()
    wandb.init(project=args.wandb_project, config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    ref_model.requires_grad_(False)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    dataset = load_dataset(args.dataset_name, split="train")
    collate = partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length, device=device)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    delta_theta, ref_cache = None, None
    if args.dpo_type == "approx":
        assert args.ref_cache_path is not None, "Must provide --ref_cache_path for approx mode."

        try:
            print(f"Loading reference cache from {args.ref_cache_path}...")
            ref_cache = load_ref_cache(args.ref_cache_path)
        except FileNotFoundError:
            print("Cache not found. Generating...")
            ref_cache = generate_ref_cache(ref_model, tokenizer, dataset, device,
                                           max_length=args.max_length)
            save_ref_cache(ref_cache, args.ref_cache_path)
            print(f"Saved reference cache to {args.ref_cache_path}")

        delta_theta = get_delta_theta(model, ref_model)

    train(model, ref_model, tokenizer, optimizer, train_dataloader,
          epochs=args.epochs, beta=args.beta,
          loss_type=args.dpo_type, delta_theta=delta_theta, ref_cache=ref_cache)

    model.save_pretrained("model-DPO")

if __name__ == "__main__":
    main()
