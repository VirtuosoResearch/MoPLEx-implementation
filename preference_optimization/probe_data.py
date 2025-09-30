# %%
from alignment import get_datasets

# constants
data_args = {
    "HuggingFaceH4/ultrafeedback_binarized": 1.0
}
dataset_splits = ["train_prefs", "test_prefs"]
dataset_configs = None

raw_datasets = get_datasets(
    data_args,
    splits=dataset_splits,
    configs=dataset_configs,
    columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
    # seed=training_args.seed,
)
column_names = list(raw_datasets["train"].features)

'''
Process the dataset into prompts, chosen, and rejected
'''

# %%
from typing import Optional, Literal
from alignment import get_tokenizer
from alignment.data import maybe_insert_system_message, is_openai_format

class model_args:

    model_name_or_path = "Qwen/Qwen2-0.5B-Instruct"
    model_revision = "main"
    tokenizer_name_or_path = None
    trust_remote_code = True

class data_args:
    truncation_side = "left"
    chat_template = None
    preprocessing_num_workers = 12
    auto_insert_empty_system_msg = False

tokenizer = get_tokenizer(model_args, data_args)

if "Qwen" in model_args.model_name_or_path:
    # If BOS is missing, set a safe value many Qwen3 builds use:
    if tokenizer.bos_token_id is None:
        # Commonly Qwen3 uses the PAD/EOD token id as BOS (151643 in many builds)
        # Prefer whatever your tokenizer reports:
        fallback = getattr(tokenizer, "pad_token_id", None)
        if fallback is None:
            # last resort: use eos if present
            fallback = tokenizer.eos_token_id
        tokenizer.bos_token_id = fallback
        tokenizer.bos_token = tokenizer.convert_ids_to_tokens(fallback)

def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "simpo"],
    auto_insert_empty_system_msg: bool = True,
    change_template = None,
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "simpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(
                    f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
                )

            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)

            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            if example["text_chosen"].startswith(tokenizer.bos_token):
                example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token):]
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            if example["text_rejected"].startswith(tokenizer.bos_token):
                example["text_rejected"] = example["text_rejected"][len(tokenizer.bos_token):]
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require either the "
                f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example

if "criterion" in column_names: # keep the criterion column if it exists
    column_names.remove("criterion")
raw_datasets = raw_datasets.map(
    apply_chat_template,
    fn_kwargs={
        "tokenizer": tokenizer,
        "task": "simpo",
        "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        "change_template": None,
    },
    num_proc=data_args.preprocessing_num_workers,
    remove_columns=column_names,
    desc="Formatting comparisons with prompt template",
)

# Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
for split in ["train", "test"]:
    raw_datasets[split] = raw_datasets[split].rename_columns(
        {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
    )

# {'prompt': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nDo you know something about crystallography and structure factor?<|im_end|>\n',
#  'chosen': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>assistant\nCrystallography is the science of the arrangement of atoms in solids. It is a vast and interdisciplinary field that has applications in physics, chemistry, materials science, biology, and engineering.\n\nThe structure factor is a mathematical function that is used to describe the diffraction of waves by a crystal. It is a complex number that is related to the atomic positions in the crystal.\n\nThe structure factor can be used to calculate the intensity of the diffracted waves. This information can be used to determine the atomic positions in the crystal and to study the structure of materials.\n\nCrystallography is a powerful tool for understanding the structure of materials. It has been used to determine the structures of many important materials, including metals, semiconductors, and pharmaceuticals. It is also used to study the structure of biological materials, such as proteins and DNA.\n\nThe structure factor is a key concept in crystallography. It is used to describe the diffraction of waves by a crystal and to calculate the intensity of the diffracted waves. This information can be used to determine the atomic positions in the crystal and to study the structure of materials.<|im_end|>\n',
#  'rejected': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>assistant\nCertainly! Crystallography is the study of the structure, arrangement of atoms, and properties of crystals. Structure factor, on the other hand, is a mathematical parameter that describes the arrangement of atoms or molecules in a crystal. It is used to determine the crystallographic properties of a crystal, such as its unit cell dimensions and symmetry.\n\nIf you have any specific questions about crystallography or structure factors, I'd be happy to help!<|im_end|>\n"}

# %%
from datasets import load_dataset, DatasetDict, concatenate_datasets
import hashlib
import random
import time

load_specific_pairs = True
load_specific_pairs_idxes = [2, 3]

# Load revision with the fixes to overall_score
ds = load_dataset("openbmb/UltraFeedback", split="train", cache_dir="./cache/")

# Load TrutfulQA prompts to ensure we remove samples from evol_instruct
tqa_a = load_dataset("truthful_qa", "generation", split="validation")
tqa_b = load_dataset("truthful_qa", "multiple_choice", split="validation")

total_rows = ds.num_rows

ds = ds.filter(lambda x: x["source"] != "truthful_qa", num_proc=4)
print(f"Remaining samples after removing the TruthfulQA source [{ds.num_rows} / {total_rows}]")

contaminated_prompts = list(set(tqa_a["question"] + tqa_b["question"]))
ds = ds.filter(lambda x: x["instruction"] not in contaminated_prompts, num_proc=4)
print(f"Remaining samples after removing the contaminated prompts [{ds.num_rows} / {total_rows}]")

def get_pairwise_completions(completions, criterion="overall_score", seed=42):
    random.seed(seed)
    start = time.time()
    if criterion == "overall_score" or criterion == 'fine-grained_score':
        scores_and_completions = [(c[criterion], c["response"], c["model"]) for c in completions]
    elif criterion == "helpfulness" or criterion == 'honesty' or criterion == 'instruction_following' or criterion == 'truthfulness':
        scores_and_completions = [(float(c['annotations'][criterion]['Rating']), c["response"], c["model"]) \
                                  if c['annotations'][criterion]['Rating'] != 'N/A' else (0.0, c["response"], c["model"])   
                                  for c in completions]
    else:
        raise ValueError(f"Criterion {criterion} not supported!")

    if len(scores_and_completions) < 2:
        return None, None
    
    if load_specific_pairs:
        pairs = [scores_and_completions[idx] for idx in load_specific_pairs_idxes]
        chosen = max(pairs, key=lambda x: x[0])
        if pairs[0][0] == pairs[1][0]:
            rejected = pairs[1]
        else:
            rejected = min(pairs, key=lambda x: x[0])
    else:
        chosen = max(scores_and_completions, key=lambda x: x[0])
        rejected = random.choice(scores_and_completions)
        while rejected == chosen:
            end = time.time()
            if end - start > 3:
                print("Timeout")
                print(chosen, rejected)
                break
            rejected = random.choice(scores_and_completions)
    return chosen, rejected

def format_prompt(x, criterion="overall_score"):
    prompt = x["instruction"]
    chosen, rejected = get_pairwise_completions(x["completions"], criterion=criterion)
    chosen_messages = []
    rejected_messages = []
    chosen_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": chosen[1] if chosen is not None else "N/A"},
    ]
    rejected_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": rejected[1] if rejected is not None else "N/A"},
    ]
    return {
        "prompt": prompt,
        "prompt_id": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "chosen": chosen_messages,
        "rejected": rejected_messages,
        "messages": chosen_messages, # Use best-ranked example for SFT
        "score_chosen": chosen[0] if chosen is not None else -100.0,
        "score_rejected": rejected[0] if rejected is not None else -100.0,
        "criterion": criterion,
    }

ds_list = []
for criterion in ["overall_score", "helpfulness", "honesty", "instruction_following", "truthfulness"]:
    tmp_ds = ds.map(format_prompt, num_proc=8, remove_columns=ds.column_names, fn_kwargs={"criterion": criterion}, desc=f"Formatting prompts for {criterion}")
    tmp_ds = tmp_ds.filter(lambda x: x["score_chosen"] != -100 or x["score_rejected"] != -100, num_proc=8)
    ds_list.append(tmp_ds)

# %%
# load indexes 
import numpy as np

indexes = np.load("./data_processing/indexes/load_indexes.npy")
if indexes is not None:
    for i, ds in enumerate(ds_list):
        ds_list[i] = ds.select(indexes)

# %%
indexes = set()
for ds in ds_list:
    count = 0
    for i in range(len(ds)):
        if ds[i]["chosen"] == ds[i]["rejected"]:
            count += 1
            indexes.add(i)
    print(f"Number of ties: {count} / {len(ds)}")


# %%
# compute the conflict rate between different criteria
from itertools import combinations

conflict_indexes = []
for i, j in combinations(range(len(ds_list)), 2):
    ds1 = ds_list[i]
    ds2 = ds_list[j]
    
    conflict_count = 0
    total_count = 0
    idxes = []
    for k, item in enumerate(ds1):
        item_2 = ds2[k]
        chosen_1 = item["chosen"]
        chosen_2 = item_2["chosen"]
        # print(chosen_1 == chosen_2)
        if chosen_1 != chosen_2:
            conflict_count += 1
            idxes.append(k)
        total_count += 1
        # if chosen_1 != chosen_2:
        #     print(chosen_1)
        #     print(chosen_2)
        #     break
    conflict_indexes.append(idxes)
    print(i, j, conflict_count, total_count, conflict_count / len(ds1))

# %%
for i, j in combinations(range(len(conflict_indexes)), 2):
    common = set(conflict_indexes[i]).intersection(set(conflict_indexes[j]))
    print(i, j, len(common), len(conflict_indexes[i]), len(conflict_indexes[j]))

indexes = set()
for k in [0, 1, 2, 3]:
    idxes = conflict_indexes[k]
    indexes = indexes.union(set(idxes))
print(f"Total conflict examples: {len(indexes)}")

for k in [5, 8]:
    idxes = conflict_indexes[k][:int(len(conflict_indexes[k]) * 0.9)]
    indexes = indexes.difference(set(idxes))
print(f"Total conflict examples: {len(indexes)}")

# %%
import numpy as np
indexes = list(indexes)
np.save("./data_processing/indexes/load_indexes.npy", indexes)

# %%
for i, ds in enumerate(ds_list):
    ds_list[i] = ds.select(list(indexes))

# %%

def remove_last_step_for_rl(example):
    example["messages"] = example["messages"][:-1]  # remove the assistant response
    return example

def filter_empty_messages(example):
    if example["messages"][-1]["role"] == "user":
        example["messages"] = example["messages"][:-1]
    if example["chosen"][-1]["role"] == "user":
        example["chosen"] = example["chosen"][:-1]
    if example["rejected"][-1]["role"] == "user":
        example["rejected"] = example["rejected"][:-1]
    return example


from collections import defaultdict
all_ds = defaultdict(list)
for ds in ds_list:
    split_dataset = ds.train_test_split(test_size=2000, seed=42, shuffle=True)
    test_datasets = split_dataset["test"].train_test_split(0.5, seed=42, shuffle=True)

    all_ds["train_prefs"].append(split_dataset["train"].map(filter_empty_messages))
    all_ds["train_sft"].append(split_dataset["train"].map(filter_empty_messages))
    # Keep more examples for test accuracy
    all_ds["test_prefs"].append(concatenate_datasets([test_datasets["train"], test_datasets["test"]]).map(filter_empty_messages))
    all_ds["test_sft"].append(test_datasets["train"].map(filter_empty_messages))
    all_ds["train_gen"].append(all_ds["train_sft"][-1].map(remove_last_step_for_rl))
    all_ds["test_gen"].append(all_ds["test_sft"][-1].map(remove_last_step_for_rl))

for k in all_ds.keys():
    all_ds[k] = concatenate_datasets(all_ds[k])

assistant_rows = []
# check that gen split does not end with `assistant`, should print 0
for idx, row in enumerate(all_ds["train_gen"]):
    if row["messages"][-1]["role"] == "assistant":
        assistant_rows.append(row)
for row in all_ds["test_gen"]:
    if row["messages"][-1]["role"] == "assistant":
        assistant_rows.append(row)
assert len(assistant_rows) == 0

splits = ["train_prefs", "test_prefs"]
columns_to_keep = ['prompt', 'chosen', 'rejected', 'messages', 'criterion']

raw_datasets = DatasetDict()

for split in splits:
    dataset = all_ds[split]
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
    if 'train' in split:
        raw_datasets['train'] = dataset
    elif 'test' in split:
        raw_datasets['test'] = dataset

# %%
