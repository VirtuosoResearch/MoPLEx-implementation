import os
from datasets import concatenate_datasets, load_from_disk, DatasetDict, load_dataset
import torch
from transformers import AutoModelForCausalLM
from trainers.network_utils import AutoModelForCausalLMWithValueHead
from peft import get_peft_model, LoraConfig
import re
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
    path = getattr(args, 'path', None) or args.preference_dataset_path
    data, test_data = get_dataset(path, num_samples=num_samples, return_test_data=True, num_samples_test=num_samples_test)

    if concatenate_prompt:
        def map_fn(d):
            for k in ["y_ref", "y_w", "y_l"]:
                d[k] = d["prompt"] + d[k]
            return d

        data = data.map(
            map_fn,
            num_proc=args.num_proc,
        )

    dataset_name = os.path.basename(path).split(".")[0]

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

def load_data(args):
    # Check if path is a local directory with DatasetDict (e.g., imdb_synthetic)
    if os.path.isdir(args.preference_dataset_path) and os.path.exists(
        os.path.join(args.preference_dataset_path, "dataset_dict.json")
    ):
        pref_dataset_name = os.path.basename(args.preference_dataset_path)
        pref_dataset = load_from_disk(args.preference_dataset_path)
    elif args.preference_dataset_path.startswith('Asap7772'):
        pref_dataset_name = os.path.basename(args.preference_dataset_path)
        pref_dataset = load_dataset(args.preference_dataset_path)
    elif "imdb" in args.preference_dataset_path.lower():
        pref_dataset_name = os.path.basename(args.preference_dataset_path)
        pref_dataset = load_imdb_dataset(args)
    elif "ultrafeedback" in args.preference_dataset_path.lower():
        # UltraFeedback synthetic datasets are already in the correct format
        pref_dataset_name = os.path.basename(args.preference_dataset_path)
        pref_dataset = load_from_disk(args.preference_dataset_path)
    elif "mixinstruct" in args.preference_dataset_path.lower():
        # Mix-instruct synthetic datasets are already in the correct format
        pref_dataset_name = os.path.basename(args.preference_dataset_path)
        pref_dataset = load_from_disk(args.preference_dataset_path)
    else:
        pref_dataset_name, pref_dataset = construct_dataset(
            args=args,
            num_samples=args.preference_num_samples,
            concatenate_prompt=False,
        )
    print('Loaded dataset', pref_dataset_name)

    pref_dataset, eval_pref_dataset = pref_dataset['train'], pref_dataset['test']
    remove_columns = ['output', 'text', 'alpaca_text', 'y_ref', 'y_1', 'y_2', 'y_w', 'y_w_alpaca', 'y_l', 'y_l_alpaca', 'y_w_score', 'y_l_score', 'score_diff', 'prompt', 'alpaca_prompt']

    # Optionally restrict the size of the evaluation dataset using args.num_samples_test.
    # If args.num_samples_test is None or missing, use the full evaluation dataset.
    num_samples_test = getattr(args, "num_samples_test", None)
    if num_samples_test is not None:
        if num_samples_test <= 0:
            raise ValueError(f"num_samples_test must be positive when provided, got {num_samples_test}.")
        n_eval = len(eval_pref_dataset)
        num_eval = min(int(num_samples_test), n_eval)
        eval_pref_dataset = eval_pref_dataset.select(range(num_eval))

    pref_dataset = pref_dataset.shuffle(seed=args.seed).select(range(int(len(pref_dataset) * args.downsample_ratio)))
    eval_pref_dataset = eval_pref_dataset.shuffle(seed=args.seed).select(range(int(len(eval_pref_dataset) * args.downsample_ratio)))

    print(f"len(training set): {len(pref_dataset)}\nlen(test set): {len(eval_pref_dataset)}")

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
    print("#"*20)
    print(pref_dataset[0].keys())

    eval_pref_dataset = eval_pref_dataset.map(
        process_dataset,
        batched=args.batched,
        num_proc=args.num_proc,
        remove_columns=remove_columns if "alpacafarm" in args.preference_dataset_path else None,
    )

    pref_dataset_dataloader = torch.utils.data.DataLoader(
        pref_dataset,
        batch_size=max(args.batch_size, 1),
        collate_fn=None,
        shuffle=True,
        drop_last=True,
    )
    train_as_eval_pref_dataset_dataloader = torch.utils.data.DataLoader(
        pref_dataset,
        batch_size=args.mini_batch_size,
        collate_fn=None,
        shuffle=True,
        drop_last=True,
    )
    eval_pref_dataset_dataloader = torch.utils.data.DataLoader(
        eval_pref_dataset,
        batch_size=args.mini_batch_size,
        collate_fn=None,
        shuffle=True,
        drop_last=True,
    )
    all_eval_dataloaders = {
        "train_as_eval_pref": train_as_eval_pref_dataset_dataloader,
        "eval_pref": eval_pref_dataset_dataloader,
    }

    return pref_dataset_dataloader, all_eval_dataloaders

def load_model(tokenizer, args):
    policy = AutoModelForCausalLM.from_pretrained(
        args.pretrained_dir,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map='auto',
    )
    policy.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        model_type = getattr(policy.config, "model_type", "").lower()
        auto_targets = []
        if model_type in ["gpt2", "gpt_neo", "gptj", "gpt_neox", "mpt", "falcon", "pythia"]:
            if model_type == "gpt2":
                auto_targets = ["c_attn", "c_proj", "c_fc"]
            else:
                auto_targets = [
                    "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",
                    "c_attn", "c_proj", "c_fc"
                ]
        elif model_type in ["llama", "mistral", "qwen2", "qwen3", "opt"]:
            auto_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            auto_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "c_fc", "Wqkv", "out_proj", "fc1", "fc2"]
        cli_targets = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
        target_modules = cli_targets if len(cli_targets) > 0 else auto_targets
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy = get_peft_model(policy, lora_cfg)
        policy.print_trainable_parameters()

    model = AutoModelForCausalLMWithValueHead(policy)
    return model