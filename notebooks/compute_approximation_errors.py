# %%
from datasets import concatenate_datasets, load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trainers.network_utils import AutoModelForCausalLMWithValueHead

#from alpaca_farm.models.reward_model import RewardModel, RewardConfig
import torch
from absl import flags, app
import os
import accelerate
import gc
import datetime
import numpy as np
import tempfile
from tqdm import tqdm
import wandb
import re
from collections import defaultdict
from functools import reduce
from trainers.utils import (
    logprobs_from_logits,
    entropy_from_logits,
)

# %%
PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'
EOS_TOKEN = '<|endoftext|>'

class FLAGS:
    # data
    preference_dataset_path = 'tatsu-lab/alpaca_farm'
    preference_dataset_subset = "alpaca_human_preference"
    preference_dataset_split = "preference"
    batch_size = 8
    batched = True
    num_proc = 32

    downsample_ratio = 0.1
    seed=42

    # model 
    pretrained_dir = "meta-llama/Llama-3.2-1B"
    cache_dir = "../cache/"


# %%
# Load dataset

if FLAGS.preference_dataset_path == 'tatsu-lab/alpaca_farm':
    pref_dataset = load_dataset(FLAGS.preference_dataset_path, FLAGS.preference_dataset_subset, split="preference")
else:
    split='train' if 'length' in FLAGS.preference_dataset_path else FLAGS.preference_dataset_split
    pref_dataset = load_dataset(FLAGS.preference_dataset_path, split=split)
pref_dataset = pref_dataset.train_test_split(test_size=0.1, seed=FLAGS.seed)

def process_dataset(batch):
    new_batch = defaultdict(list)
    for inst, inp, out1, out2, pref in zip(batch['instruction'], batch['input'], batch['output_1'], batch['output_2'], batch['preference']):
        if pref == 1:
            selected = out1
            rejected = out2
        else:
            selected = out2
            rejected = out1
        if inp:
            text = f"{PROMPT_TOKEN}{inst}\n{inp}{EOS_TOKEN}{ASSISTANT_TOKEN}"
        else:
            text = f"{PROMPT_TOKEN}{inst}{EOS_TOKEN}{ASSISTANT_TOKEN}"
        
        new_batch['prompt'].append(text)
        new_batch['y_w'].append(f"{text}{selected}{EOS_TOKEN}")
        new_batch['y_l'].append(f"{text}{rejected}{EOS_TOKEN}")
    return new_batch

pref_dataset = pref_dataset.map(
    process_dataset,
    batched=FLAGS.batched,
    num_proc=FLAGS.num_proc,
)

pref_dataset, eval_pref_dataset = pref_dataset['train'], pref_dataset['test']
remove_columns = ['instruction', 'input', 'output_1', 'output_2', 'preference', 'raw_preference', 'prompt', 'y_w', 'y_l']

if FLAGS.downsample_ratio < 1.0:
    downsample_ratio = float(FLAGS.downsample_ratio)
    if downsample_ratio <= 0 or downsample_ratio > 1:
        raise ValueError(f"downsample_ratio must be between 0 and 1, but got {downsample_ratio}")
    pref_dataset = pref_dataset.shuffle(seed=FLAGS.seed).select(range(int(len(pref_dataset) * downsample_ratio)))
    eval_pref_dataset = eval_pref_dataset.shuffle(seed=FLAGS.seed).select(range(int(len(eval_pref_dataset) * downsample_ratio)))
    

def process_dataset(batch):
    new_batch = {}
    new_batch['query'] = batch['prompt']
    new_batch['text_w'] =  batch['y_w'] 
    new_batch['text_l'] = batch['y_l']
    new_batch['response_w'] = [x.split(ASSISTANT_TOKEN)[-1] for x in batch['y_w']]
    new_batch['response_l'] = [x.split(ASSISTANT_TOKEN)[-1] for x in batch['y_l']]
    
    shapes = {}
    for k, v in new_batch.items():
        shapes[k] = len(v)
    if reduce(lambda x,y: x if x==y else -1, list(shapes.values())) == -1:
        assert False, f"Shapes of all columns must be equal, but got {shapes}, {list(shapes.values())}"
    return new_batch


pref_dataset = pref_dataset.map(
    process_dataset,
    batched=FLAGS.batched,
    num_proc=FLAGS.num_proc,
    remove_columns=remove_columns,
)

eval_pref_dataset = eval_pref_dataset.map(
    process_dataset,
    batched=FLAGS.batched,
    num_proc=FLAGS.num_proc,
    remove_columns=remove_columns,
)

pref_dataset_dataloader = torch.utils.data.DataLoader(
    pref_dataset,
    batch_size=FLAGS.batch_size,
    collate_fn=None,
    shuffle=False,
    drop_last=True,
)

train_as_eval_pref_dataset_dataloader = torch.utils.data.DataLoader(
    pref_dataset,
    batch_size=FLAGS.batch_size,
    collate_fn=None,
    shuffle=False,
    drop_last=True,
)

eval_pref_dataset_dataloader = torch.utils.data.DataLoader(
    eval_pref_dataset,
    batch_size=FLAGS.batch_size,
    collate_fn=None,
    shuffle=False,
    drop_last=True,
)

# %%

tokenizer = AutoTokenizer.from_pretrained(FLAGS.pretrained_dir)
tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
eos = tokenizer.eos_token


policy = AutoModelForCausalLM.from_pretrained(
    FLAGS.pretrained_dir,
    cache_dir=FLAGS.cache_dir, 
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
policy.resize_token_embeddings(len(tokenizer))
model = AutoModelForCausalLMWithValueHead(policy)

# %%
# get reference model outputs
from trainers.utils import masked_mean

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

generation_kwargs = {
    "top_k": 0.0, # no top-k sampling
    "top_p": 1.0, # no nucleus sampling
    "do_sample": True, # yes, we want to sample
    "pad_token_id": tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
    "max_new_tokens": 256, # specify how many tokens you want to generate at most
    "temperature": 1.0, # control the temperature of the softmax, 1.0 means no change, lower means more greedy, higher means more diverse
    "use_cache": True, # whether or not the model should use the past key/values attentions (if the model supports it)
}

def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

@torch.no_grad()
def process_pref_batch(pref_batch):
    ### Process preference dataset
    pref_query = tokenizer(pref_batch["query"], padding= True, truncation=True, max_length=128, return_tensors='pt').input_ids
    pref_query_tensors = accelerate.utils.send_to_device(pref_query, device)
    
    # Tokenize together to be the same length
    all_pref = pref_batch["response_w"] + pref_batch["response_l"]
    tokenized = tokenizer(all_pref, padding=True, truncation=True, max_length=64+generation_kwargs['max_new_tokens'], return_tensors='pt').input_ids
    
    pref_response_w_tensors = tokenized[:len(pref_batch["response_w"])]
    pref_response_w_tensors = accelerate.utils.send_to_device(pref_response_w_tensors, device)
    
    pref_response_l_tensors = tokenized[len(pref_batch["response_w"]):]
    assert pref_response_l_tensors.shape[0] == len(pref_batch["response_l"])
    pref_response_l_tensors = accelerate.utils.send_to_device(pref_response_l_tensors, device)

    return pref_batch, pref_query_tensors, pref_response_w_tensors, pref_response_l_tensors

bs = sub_bs = FLAGS.batch_size
total_iterations = 0

params = [weight for weight in model.parameters() if weight.requires_grad]
all_outputs = []; all_gradients = []
for sub_iteration, pref_batch in tqdm(enumerate(pref_dataset_dataloader), desc="Batches"):
    if sub_iteration == 0:
        print(pref_batch["query"][0])

    empty_cache()

    stats = {}

    pref_batch, pref_query_tensors, pref_response_w_tensors, pref_response_l_tensors = process_pref_batch(pref_batch) # use pref data completions
    queries=pref_query_tensors; responses_w=pref_response_w_tensors; responses_l=pref_response_l_tensors

    #### Run Trainer step
    assert queries.ndim == 2 and responses_w.ndim == 2 and responses_l.ndim == 2
    
    first = True
    for i in tqdm(range(0, bs, sub_bs), desc="Training with Minibatches", leave=False):
        queries_ = queries[i : i + sub_bs]
        responses_w_ = responses_w[i : i + sub_bs]
        responses_l_ = responses_l[i : i + sub_bs]

        input_ids_w = torch.cat((queries, responses_w), dim=1)
        input_ids_l = torch.cat((queries, responses_l), dim=1)

        # mask out query tokens, keep response tokens. Remove last token from response tokens.
        mask_w = torch.cat((torch.zeros_like(queries), torch.ones_like(responses_w)), dim=1)[:,:-1]
        mask_l = torch.cat((torch.zeros_like(queries), torch.ones_like(responses_l)), dim=1)[:,:-1]
        assert mask_w.shape == mask_l.shape, f"mask_w and mask_l should have the same shape, {mask_w.shape}, {mask_l.shape}."
        mask = mask_w
        
        def forward_pass(input_ids):
            
            input_data = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
            logits, _, _ = model(**input_data)
            
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            entropy = entropy_from_logits(logits)
            return logprobs, entropy, logits
    
        logprobs_w, entropy_w, logits_w = forward_pass(input_ids_w)
        logprobs_l, entropy_l, logits_l = forward_pass(input_ids_l)

        outputs = masked_mean(logprobs_w - logprobs_l, mask, axis=1)
        gradients = torch.autograd.grad(outputs.mean(), params, retain_graph=False, create_graph=False, allow_unused=True)
        gradients = [gradient.view(-1) if gradient is not None else torch.zeros_like(param).view(-1) for param, gradient in zip(params, gradients)]

        all_outputs.append(outputs.detach().cpu().numpy())
        all_gradients.append(torch.cat(gradients).cpu().numpy())
    total_iterations += 1
    if total_iterations >= 5:
        break

# %%
state_dict = {key: val.cpu().clone() for key, val in model.state_dict().items()}
state_dict.pop("lm_head.weight")
print("")


# %%
ref_outputs = np.concatenate(all_outputs, axis=0)
ref_gradients = np.stack(all_gradients, axis=0)

# %%
# Specify the path to your fine-tuned model
model_path = f"../outputs/scalable-preference-optimization/alpaca_farm_beta0.05_lr1e-7_bs4_ga4_sd3/scalable-preference-optimization_alpaca_farm_beta0.05_lr1e-7_bs4_ga4_sd3_num_batches_0"
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, torch_dtype=torch.bfloat16)
# %%
finetuned_state_dict = {key: val.cpu().clone() for key, val in model.state_dict().items()}
finetuned_state_dict.pop("lm_head.weight")
print("")

finetuned_vector = []
for key in state_dict.keys():
    diff = finetuned_state_dict[key] - state_dict[key]
    finetuned_vector.append(diff.view(-1))
finetuned_vector = torch.cat(finetuned_vector).numpy()


# %%
model = model.to(device)
total_iterations = 0
all_outputs = []
for sub_iteration, pref_batch in tqdm(enumerate(pref_dataset_dataloader), desc="Batches"):
    if sub_iteration == 0:
        print(pref_batch["query"][0])

    empty_cache()

    stats = {}

    pref_batch, pref_query_tensors, pref_response_w_tensors, pref_response_l_tensors = process_pref_batch(pref_batch) # use pref data completions
    queries=pref_query_tensors; responses_w=pref_response_w_tensors; responses_l=pref_response_l_tensors

    #### Run Trainer step
    assert queries.ndim == 2 and responses_w.ndim == 2 and responses_l.ndim == 2
    
    first = True
    for i in tqdm(range(0, bs, sub_bs), desc="Training with Minibatches", leave=False):
        queries_ = queries[i : i + sub_bs]
        responses_w_ = responses_w[i : i + sub_bs]
        responses_l_ = responses_l[i : i + sub_bs]

        input_ids_w = torch.cat((queries, responses_w), dim=1)
        input_ids_l = torch.cat((queries, responses_l), dim=1)

        # mask out query tokens, keep response tokens. Remove last token from response tokens.
        mask_w = torch.cat((torch.zeros_like(queries), torch.ones_like(responses_w)), dim=1)[:,:-1]
        mask_l = torch.cat((torch.zeros_like(queries), torch.ones_like(responses_l)), dim=1)[:,:-1]
        assert mask_w.shape == mask_l.shape, f"mask_w and mask_l should have the same shape, {mask_w.shape}, {mask_l.shape}."
        mask = mask_w
        
        def forward_pass(input_ids):
            
            input_data = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
            logits, _, _ = model(**input_data)
            
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            entropy = entropy_from_logits(logits)
            return logprobs, entropy, logits
    
        logprobs_w, entropy_w, logits_w = forward_pass(input_ids_w)
        logprobs_l, entropy_l, logits_l = forward_pass(input_ids_l)

        outputs = masked_mean(logprobs_w - logprobs_l, mask, axis=1)
        all_outputs.append(outputs.detach().cpu().numpy())
    total_iterations += 1
    if total_iterations >= 5:
        break

finetuned_outputs = np.concatenate(all_outputs, axis=0)
# %%
print("ref_outputs", ref_outputs.mean())
print("finetuned_outputs", finetuned_outputs.mean())
print("gradient_term", (finetuned_vector.reshape(1, -1) * ref_gradients).mean())
print("Estimation", ref_outputs.mean() + (finetuned_vector.reshape(1, -1) * ref_gradients).mean())