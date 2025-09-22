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
# %%
