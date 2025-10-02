from datasets import load_dataset, DatasetDict, concatenate_datasets
import numpy as np
import random
import time

annotator_ids = np.load('./data_processing/collective-alignment/annotator_ids.npy', allow_pickle=True).tolist()


def load_collective_alignment(annotators = None,
                            load_specific_pairs = False, 
                            load_specific_pairs_idxes = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]],
                            load_indexes_path = None, ):
    ds = load_dataset("openai/collective-alignment-1", "comparisons", split="train", cache_dir="./cache/")
    total_rows = ds.num_rows

    def explode_assessments(batch):
        out = {
            "prompt_id": [],
            "prompt": [],
            "responses": [],          # keep the full candidates for context
            "assessment": [],         # the single assessment for this new row
        }
        for i in range(len(batch["prompt_id"])):
            meta = batch["metadata"][i]
            assessments = meta.get("assessments", [])
            for a in assessments:
                out["prompt_id"].append(batch["prompt_id"][i])
                out["prompt"].append(batch["prompt"][i])
                out["responses"].append(batch["responses"][i])
                out["assessment"].append(a)
        return out

    ds = ds.map(
        explode_assessments,
        batched=True,
        remove_columns=ds.column_names,
    )
    
    if annotators is not None:
        annotator_set = set([annotator_ids[idx] for idx in annotators])
        ds = ds.filter(lambda x: x['assessment']['annotator_id'] in annotator_set, num_proc=8)
        
    # Parse rankings and get pairwise comparisons
    def parse_ranking(ranking_str):
        """
        Parse a ranking string like 'B>A=C=D' into groups.
        """
        groups = ranking_str.split(">")        # split by '>'
        parsed = [group.split("=") for group in groups]
        # flatten the list
        return [item.strip() for sublist in parsed for item in sublist]

    def get_pairwise_completions(responses, ranking, pair_indexes, seed=42):
        random.seed(seed)
        start = time.time()

        ranking = parse_ranking(ranking)
        scores_and_completions = [(len(ranking) - ranking.index(resp['response_index']), resp['messages'][0], resp['response_index']) for resp in responses]
        scores_and_completions.sort(key=lambda x: x[2]) # sort by score descending
        
        pairs = [scores_and_completions[idx] for idx in pair_indexes]
        chosen = max(pairs, key=lambda x: x[0])
        if pairs[0][0] == pairs[1][0]:
            rejected = pairs[1]
        else:
            rejected = min(pairs, key=lambda x: x[0])
        return chosen, rejected

    def format_prompt(x, pair_indexes=[0, 1]):
        prompt = x["prompt"]['messages'][-1]
        ranking = x["assessment"]['ranking_blocks']['personal']
        if not ranking:
            ranking = x["assessment"]['ranking_blocks']['world']
        ranking = ranking[0]['ranking']
        chosen, rejected = get_pairwise_completions(x["responses"], ranking, pair_indexes=pair_indexes)
        chosen_messages = []
        rejected_messages = []
        chosen_messages = [
            prompt,
            chosen[1],
        ]
        rejected_messages = [
            prompt,
            rejected[1],
        ]
        return {
            "prompt": prompt,
            "prompt_id": x["prompt_id"],
            "chosen": chosen_messages,
            "rejected": rejected_messages,
            "messages": chosen_messages, # Use best-ranked example for SFT
            "score_chosen": chosen[0] if chosen is not None else -100.0,
            "score_rejected": rejected[0] if rejected is not None else -100.0,
            "criterion": x['assessment']['annotator_id'],
        }

    ds_list = []
    for pairs_idxes in load_specific_pairs_idxes:
        tmp_ds = ds.map(format_prompt, num_proc=8, remove_columns=ds.column_names, fn_kwargs={"pair_indexes": pairs_idxes})
        tmp_ds = tmp_ds.filter(lambda x: x["score_chosen"] != -100 or x["score_rejected"] != -100, num_proc=8)
        ds_list.append(tmp_ds)


    # Post processing
    splits = ["train_prefs", "test_prefs"]
    columns_to_keep = ['prompt', 'chosen', 'rejected', 'messages', 'criterion']
    test_size = int(len(ds_list[0]) * 0.2) # take 20% as test set
    
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
        # all_ds["train_gen"].append(all_ds["train_sft"][-1].map(remove_last_step_for_rl))
        # all_ds["test_gen"].append(all_ds["test_sft"][-1].map(remove_last_step_for_rl))

    for k in all_ds.keys():
        all_ds[k] = concatenate_datasets(all_ds[k])

    # assistant_rows = []
    # # check that gen split does not end with `assistant`, should print 0
    # for idx, row in enumerate(all_ds["train_gen"]):
    #     if row["messages"][-1]["role"] == "assistant":
    #         assistant_rows.append(row)
    # for row in all_ds["test_gen"]:
    #     if row["messages"][-1]["role"] == "assistant":
    #         assistant_rows.append(row)
    # assert len(assistant_rows) == 0

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