from datasets import load_dataset, DatasetDict, concatenate_datasets
import hashlib
import random
import time
import numpy as np

def load_ultrafeedback_multi_preferences(criterions = ["overall_score", "helpfulness", "honesty", "instruction_following", "truthfulness"],
                                         load_specific_pairs = False, 
                                         load_specific_pairs_idxes = [2, 3],
                                         load_indexes_path = None,
                                         test_size = 2000):
    splits = ["train_prefs", "test_prefs"]
    columns_to_keep = ['prompt', 'chosen', 'rejected', 'messages', 'criterion']
    
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
    for criterion in criterions:
        tmp_ds = ds.map(format_prompt, num_proc=8, remove_columns=ds.column_names, fn_kwargs={"criterion": criterion}, desc=f"Formatting prompts for {criterion}")
        tmp_ds = tmp_ds.filter(lambda x: x["score_chosen"] != -100 or x["score_rejected"] != -100, num_proc=8)
        ds_list.append(tmp_ds)

    indexes = np.load(load_indexes_path) if load_indexes_path is not None else None
    if indexes is not None:
        for i, ds in enumerate(ds_list):
            ds_list[i] = ds.select(indexes)

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
        split_dataset = ds.train_test_split(test_size=test_size, seed=42, shuffle=True)
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

    all_ds = DatasetDict(all_ds)
    raw_datasets = DatasetDict()

    for split in splits:
        dataset = all_ds[split]
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
        if 'train' in split:
            raw_datasets['train'] = dataset
        elif 'test' in split:
            raw_datasets['test'] = dataset

    return raw_datasets