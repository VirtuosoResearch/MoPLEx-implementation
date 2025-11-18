import os
from datasets import concatenate_datasets, load_from_disk, DatasetDict, load_dataset

PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'
EOS_TOKEN = '<|endoftext|>'

generation_kwargs = {
    "top_k": 0.0,  # no top-k sampling
    "top_p": 1.0,  # no nucleus sampling
    "do_sample": True,  # yes, we want to sample
    "max_new_tokens": 256,  # specify how many tokens you want to generate at most
    "temperature": 1.0,  # control the temperature of the softmax
    "use_cache": True,  # whether the model should use past key/values attentions
}

def get_dataset(path, num_samples=-1, return_test_data=True, num_samples_test=1000):
    assert os.path.exists(path)
    folders = os.listdir(path)
    regex = r"^\d+-\d+$"
    folders = [x for x in folders if re.search(regex, x)]
    folders.sort(key=lambda x: int(x.split("-")[0]))
    total_samples = int(folders[-1].split("-")[-1])

    assert 0 < num_samples <= total_samples - num_samples_test, f"num_samples {num_samples} must be between 0 and {total_samples} - {num_samples_test}"
    assert 0 < num_samples_test <= total_samples, f"num_samples_test {num_samples_test} must be between 0 and {total_samples}"

    num_samples_train = num_samples if num_samples > 0 else total_samples - num_samples_test
    test_folders = [x for x in folders if int(x.split("-")[0]) >= num_samples_train]
    folders = [x for x in folders if int(x.split("-")[0]) < num_samples_train]

    datasets = [load_from_disk(os.path.join(path, x)) for x in folders]
    full_data = concatenate_datasets(datasets)

    if num_samples > 0:
        full_data = full_data.select(range(num_samples))

    if return_test_data:
        test_datasets = [load_from_disk(os.path.join(path, x)) for x in test_folders]
        test_data = concatenate_datasets(test_datasets)
        if num_samples_test > 0:
            test_data = test_data.select(range(num_samples_test))
        return full_data, test_data

    return full_data


def construct_dataset(
    args,
    num_samples=-1,
    concatenate_prompt=False,
    num_samples_test=1000,
):
    data, test_data = get_dataset(args.path, num_samples=num_samples, return_test_data=True, num_samples_test=num_samples_test)

    # print("##"*20)
    # print("data[0]: ", data[0])
    # print("test_data[0]: ", test_data[0])
    # print("$$"*20)
    # exit()
    if concatenate_prompt:
        def map_fn(d):
            for k in ["y_ref", "y_w", "y_l"]:
                d[k] = d["prompt"] + d[k]
            return d

        data = data.map(
            map_fn,
            num_proc=args.num_proc,
        )

    dataset_name = os.path.basename(args.path).split(".")[0]

    ds = DatasetDict({
        "train": data,
        "test": test_data,
    })
    return dataset_name, ds

def load_imdb_dataset(args):
    pref_dataset = load_dataset(args.preference_dataset_path)

    def make_imdb_pref(batch):
        prompts = batch["prompt"]
        all_responses = batch["responses"]
        chosens = batch["chosen"]
        y_w_list = []
        y_l_list = []
        for resp_list, c in zip(all_responses, chosens):
            win = resp_list[c]
            lose = resp_list[1 - c]
            y_w_list.append(f"{ASSISTANT_TOKEN} {win}")
            y_l_list.append(f"{ASSISTANT_TOKEN} {lose}")
        return {
            "prompt": prompts,
            "y_w": y_w_list,
            "y_l": y_l_list,
        }

    for split in pref_dataset.keys():
        pref_dataset[split] = pref_dataset[split].map(
            make_imdb_pref,
            batched=True,
            num_proc=args.num_proc,
        )

    return pref_dataset

def data_process(pref_dataset, args):
    pref_dataset, eval_pref_dataset = pref_dataset['train'], pref_dataset['test']
    remove_columns = ['output', 'text', 'alpaca_text', 'y_ref', 'y_1', 'y_2', 'y_w', 'y_w_alpaca', 'y_l', 'y_l_alpaca', 'y_w_score', 'y_l_score', 'score_diff', 'prompt', 'alpaca_prompt']

    pref_dataset = pref_dataset.shuffle(seed=args.seed).select(range(int(len(pref_dataset) * args.downsample_ratio)))
    eval_pref_dataset = eval_pref_dataset.shuffle(seed=args.seed).select(range(int(len(eval_pref_dataset) * args.downsample_ratio)))

    def process_dataset(batch):
        new_batch = {}
        new_batch['query'] = batch['prompt']
        new_batch['text_w'] = batch['y_w']
        new_batch['text_l'] = batch['y_l']
        new_batch['response_w'] = [x.split(ASSISTANT_TOKEN)[-1] for x in batch['y_w']]
        new_batch['response_l'] = [x.split(ASSISTANT_TOKEN)[-1] for x in batch['y_l']]

        shapes = {}
        for k, v in new_batch.items():
            shapes[k] = len(v)
        return new_batch

    pref_dataset = pref_dataset.map(
        process_dataset,
        batched=args.batched,
        num_proc=args.num_proc,
        remove_columns=remove_columns if "alpacafarm" in args.preference_dataset_path else None,
    )

    eval_pref_dataset = eval_pref_dataset.map(
        process_dataset,
        batched=args.batched,
        num_proc=args.num_proc,
        remove_columns=remove_columns if "alpacafarm" in args.preference_dataset_path else None,
    )

    return pref_dataset, eval_pref_dataset