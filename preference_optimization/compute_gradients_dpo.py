# %%
from alignment import get_datasets, get_tokenizer
from data_processing.load_ultrafeedback import load_ultrafeedback_multi_preferences
from data_processing.load_collective_alignment import load_collective_alignment
from data_processing.load_imdb_preference_with_source import load_imdb_preference_with_source
from run_simpo import apply_chat_template


class data_args:
    load_multi_preference = True
    load_specific_pairs = True
    load_multi_preference_dataset = "imdb_preference_with_source"

    preprocessing_num_workers = 4
    auto_insert_empty_system_msg = True
    chat_template = None
    truncation_side = None

    preference_sources = None
    annotator_ids = None
    test_size = 2000 # No use, predefined

class model_args:
    model_name_or_path = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer_name_or_path = None
    model_revision = "main"
    trust_remote_code = True

###############
# Load datasets
###############
if data_args.load_multi_preference:
    if data_args.load_multi_preference_dataset == "openbmb/UltraFeedback":
        if data_args.load_specific_pairs:
            criterions = data_args.load_multi_preference_criterions.split(",")
            raw_datasets = load_ultrafeedback_multi_preferences(criterions=criterions, load_specific_pairs=data_args.load_specific_pairs,
                                                            load_specific_pairs_idxes=[2, 3],
                                                            load_indexes_path="./data_processing/indexes/load_indexes.npy", 
                                                            test_size=data_args.test_size)
        else:
            criterions = data_args.load_multi_preference_criterions.split(",")
            raw_datasets = load_ultrafeedback_multi_preferences(criterions=criterions)
    elif data_args.load_multi_preference_dataset == "openai/collective-alignment-1":
        raw_datasets = load_collective_alignment(
            annotators = None,
            load_specific_pairs_idxes=[[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
        )
    elif data_args.load_multi_preference_dataset == "imdb_preference_with_source":
        # Parse sources if provided
        sources = None
        if data_args.preference_sources:
            sources = [
                src.strip() for src in data_args.preference_sources.split(",") if src.strip()
            ]
        
        # Parse annotator IDs if provided
        annotator_ids = None
        if hasattr(data_args, 'annotator_ids') and data_args.annotator_ids:
            try:
                annotator_ids = [
                    int(idx.strip()) for idx in data_args.annotator_ids.split(",") if idx.strip()
                ]
            except ValueError as exc:
                raise ValueError(
                    "`annotator_ids` must be a comma separated list of integers."
                ) from exc
        
        # Parse subset_id if provided (overrides annotator_ids)
        subset_id = None
        if hasattr(data_args, 'subset_id') and data_args.subset_id is not None:
            subset_id = data_args.subset_id
        
        raw_datasets = load_imdb_preference_with_source(
            seed=42,
            test_size=data_args.test_size,
            sources=sources,
            annotator_ids=annotator_ids,
            # subset_id=subset_id,
        )
else:
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
        # seed=training_args.seed,
    )
column_names = list(raw_datasets["train"].features)

#####################################
# Load tokenizer and process datasets
#####################################
data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
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


if "mistral" in model_args.model_name_or_path.lower():
    change_template = "mistral"
else:
    change_template = None

#####################
# Apply chat template
#####################
if "criterion" in column_names: # keep the criterion column if it exists
    column_names.remove("criterion")
if "annotator" in column_names: # keep the annotator column if it exists
    column_names.remove("annotator")
raw_datasets = raw_datasets.map(
    apply_chat_template,
    fn_kwargs={
        "tokenizer": tokenizer,
        "task": "simpo",
        "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        "change_template": change_template,
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
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]

# %%
# load the model
import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

class training_args:
    max_length = 2048
    max_prompt_length = 1024
    truncation_mode = "keep_end" 
    label_pad_token_id = -100
    dpo_beta = 0.1
    label_smoothing = 0.0
    
    # LoRA configurations
    lora_r = 8
    lora_alpha = 32
    lora_dropout = 0.1
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16,
        use_cache=True,
        device_map=None,
        quantization_config=None,
        attn_implementation=None,
    )

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    **model_kwargs,
)

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=training_args.lora_r,
    lora_alpha=training_args.lora_alpha,
    lora_dropout=training_args.lora_dropout,
    target_modules=training_args.lora_target_modules,
    bias="none"
)

# Apply LoRA adapter
model = get_peft_model(model, peft_config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# %%
# get data loader on the preference data
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
from trl.trainer.utils import DPODataCollatorWithPadding
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union


def build_tokenized_answer(prompt, answer):
    """
    Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
    It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
    Reference:
        https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
    """

    full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
    prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
    answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

    # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
    full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

    # Prepare input tokens for token by token comparison
    full_input_ids = np.array(full_tokenized["input_ids"])

    if len(full_input_ids) != len(full_concat_input_ids):
        raise ValueError("Prompt input ids and answer input ids should have the same length.")

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = len(prompt_input_ids)

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

    if len(prompt_input_ids) != len(prompt_attention_mask):
        raise ValueError("Prompt input ids and attention mask should have the same length.")

    answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
    answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

    return dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        input_ids=answer_input_ids,
        attention_mask=answer_attention_mask,
    )

def tokenize_row(feature, model = None) -> Dict:
    """Tokenize a single row from a SimPO specific dataset.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
    in case the prompt + chosen or prompt + rejected responses is/are too long. First
        we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

    We also create the labels for the chosen/rejected responses, which are of length equal to
        the sum of the length of the prompt and the chosen/rejected response, with
        label_pad_token_id for the prompt tokens.
    """
    batch = {}
    prompt = feature["prompt"]
    chosen = feature["chosen"]
    rejected = feature["rejected"]

    # Check issues below for more details
    #  1. https://github.com/huggingface/trl/issues/907
    #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
    #  3. https://github.com/LianjiaTech/BELLE/issues/337
    if not isinstance(prompt, str):
        raise ValueError(f"prompt should be an str but got {type(prompt)}")
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

    if not isinstance(chosen, str):
        raise ValueError(f"chosen should be an str but got {type(chosen)}")
    chosen_tokens = build_tokenized_answer(prompt, chosen)

    if not isinstance(rejected, str):
        raise ValueError(f"rejected should be an str but got {type(rejected)}")
    rejected_tokens = build_tokenized_answer(prompt, rejected)

    # Last prompt token might get merged by tokenizer and
    # it should not be included for generation if that happens
    prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

    chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
    rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
    prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

    for k, v in prompt_tokens.items():
        prompt_tokens[k] = v[:prompt_len_input_ids]

    # Make sure prompts only have one different token at most an
    # and length only differs by 1 at most
    num_diff_tokens = sum(
        [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
    )
    num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
    if num_diff_tokens > 1 or num_diff_len > 1:
        raise ValueError(
            "Chosen and rejected prompt_input_ids might only differ on the "
            "last token due to tokenizer merge ops."
        )

    # add BOS token to head of prompt. Avoid adding if it's already there
    bos_token_id = tokenizer.bos_token_id
    if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
        prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
        prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
    if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
        chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
        chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
    if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
        rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
        rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

    # add EOS token to end of answer. Avoid adding if it's already there
    eos_token_id = tokenizer.eos_token_id
    if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
        chosen_tokens["input_ids"].append(eos_token_id)
        chosen_tokens["attention_mask"].append(1)
    if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
        rejected_tokens["input_ids"].append(eos_token_id)
        rejected_tokens["attention_mask"].append(1)

    longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

    # if combined sequence is too long, truncate the prompt
    for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
        if len(answer_tokens["prompt_input_ids"]) + longer_response_length > training_args.max_length:
            if training_args.truncation_mode == "keep_start":
                for k in ["prompt_input_ids", "prompt_attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: training_args.max_prompt_length]
            elif training_args.truncation_mode == "keep_end":
                for k in ["prompt_input_ids", "prompt_attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][-training_args.max_prompt_length :]
            else:
                raise ValueError(f"Unknown truncation mode: {training_args.truncation_mode}")

    # if that's still too long, truncate the response
    for answer_tokens in [chosen_tokens, rejected_tokens]:
        if len(answer_tokens["prompt_input_ids"]) + longer_response_length > training_args.max_length:
            for k in ["input_ids", "attention_mask"]:
                answer_tokens[k] = answer_tokens[k][: training_args.max_length - training_args.max_prompt_length]

    # Create labels
    chosen_sequence_tokens = {
        k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
    }
    rejected_sequence_tokens = {
        k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
    }
    chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
    chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
        training_args.label_pad_token_id
    ] * len(chosen_tokens["prompt_input_ids"])
    rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
    rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
        training_args.label_pad_token_id
    ] * len(rejected_tokens["prompt_input_ids"])

    for k, toks in {
        "chosen_": chosen_sequence_tokens,
        "rejected_": rejected_sequence_tokens,
        "": prompt_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}{type_key}"] = tokens
 
    if "criterion" in feature:
        batch["criterion"] = feature["criterion"]
    else:
        batch["criterion"] = "uniform"

    return batch

train_dataset = train_dataset.map(tokenize_row, num_proc=12)
eval_dataset = eval_dataset.map(tokenize_row, num_proc=12)

data_collator = DPODataCollatorWithPadding(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
        is_encoder_decoder=False,
    )
train_dataloader = DataLoader(
    train_dataset,
    shuffle=False,
    collate_fn=data_collator,
    batch_size=1,
)
test_dataloader = DataLoader(
    eval_dataset,
    shuffle=False,
    collate_fn=data_collator,
    batch_size=1,
)

# %%
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from trl.trainer.utils import pad_to_length

def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]],
    is_encoder_decoder: bool = False,
    label_pad_token_id: int = -100,
    padding_value: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        is_encoder_decoder: Whether the model is an encoder-decoder model.
        label_pad_token_id: The label pad token id.
        padding_value: The padding value to use for the concatenated inputs_ids.
        device: The device for the concatenated inputs.

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    concatenated_batch = {}

    max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            if "labels" in k or is_encoder_decoder:
                pad_value = label_pad_token_id
            elif k.endswith("_input_ids"):
                pad_value = padding_value
            elif k.endswith("_attention_mask"):
                pad_value = 0
            concatenated_key = k.replace("chosen", "concatenated")
            tensor = pad_to_length(batch[k], max_length, pad_value=pad_value)
            if device is not None:
                tensor = tensor.to(device=device)
            concatenated_batch[concatenated_key] = tensor
    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            if "labels" in k or is_encoder_decoder:
                pad_value = label_pad_token_id
            elif k.endswith("_input_ids"):
                pad_value = padding_value
            elif k.endswith("_attention_mask"):
                pad_value = 0
            concatenated_key = k.replace("rejected", "concatenated")
            tensor = pad_to_length(batch[k], max_length, pad_value=pad_value)
            if device is not None:
                tensor = tensor.to(device=device)
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    tensor,
                ),
                dim=0,
            ).to(device=device)

    if is_encoder_decoder:
        concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
        concatenated_batch["concatenated_attention_mask"] = (
            batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
        )

    return concatenated_batch


def _compute_sequence_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id: int,
    *,
    is_encoder_decoder: bool = False,
) -> Tuple[torch.FloatTensor, torch.Tensor, torch.FloatTensor, torch.FloatTensor]:
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    if not is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

    loss_mask = labels != label_pad_token_id
    safe_labels = labels.clone()
    safe_labels[~loss_mask] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1),
        dim=2,
        index=safe_labels.unsqueeze(2),
    ).squeeze(2)

    masked_logps = per_token_logps * loss_mask
    sequence_logps = masked_logps.sum(-1)
    token_counts = loss_mask.sum(-1)

    return sequence_logps, loss_mask, masked_logps, token_counts


def concatenated_forward(model, batch):
    device = next(model.parameters()).device

    concatenated_batch = concatenated_inputs(
        batch,
        label_pad_token_id=training_args.label_pad_token_id,
        device=device,
    )

    input_ids = concatenated_batch["concatenated_input_ids"].to(device)
    attention_mask = concatenated_batch["concatenated_attention_mask"].to(device)
    labels = concatenated_batch.get("concatenated_labels")
    if labels is None:
        raise KeyError("concatenated_labels missing from batch; check data collator output.")
    labels = labels.to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    logits = outputs.logits
    sequence_logps, loss_mask, masked_logps, token_counts = _compute_sequence_logps(
        logits,
        labels,
        training_args.label_pad_token_id,
    )

    len_chosen = batch["chosen_labels"].shape[0]

    return {
        "chosen_logps": sequence_logps[:len_chosen],
        "rejected_logps": sequence_logps[len_chosen:],
        "chosen_logits": logits[:len_chosen],
        "rejected_logits": logits[len_chosen:],
        "concatenated_logits": logits,
        "concatenated_labels": labels,
        "loss_mask": loss_mask,
        "per_token_logps": masked_logps,
        "token_counts": token_counts,
        "model_outputs": outputs,
        "num_chosen": len_chosen,
    }


def compute_loss_and_outputs(
    model,
    batch,
    *,
    beta: Optional[float] = None,
    label_smoothing: Optional[float] = None,
    reference_chosen_logps: Optional[torch.FloatTensor] = None,
    reference_rejected_logps: Optional[torch.FloatTensor] = None,
):
    beta = training_args.dpo_beta if beta is None else beta
    label_smoothing = training_args.label_smoothing if label_smoothing is None else label_smoothing

    forward_dict = concatenated_forward(model, batch)

    policy_chosen_logps = forward_dict["chosen_logps"]
    policy_rejected_logps = forward_dict["rejected_logps"]
    device = policy_chosen_logps.device

    if reference_chosen_logps is None:
        reference_chosen_logps = batch.get("reference_chosen_logps")
    if reference_rejected_logps is None:
        reference_rejected_logps = batch.get("reference_rejected_logps")

    if reference_chosen_logps is None:
        reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
    else:
        reference_chosen_logps = reference_chosen_logps.to(device)
    if reference_rejected_logps is None:
        reference_rejected_logps = torch.zeros_like(policy_rejected_logps)
    else:
        reference_rejected_logps = reference_rejected_logps.to(device)

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios

    losses = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )

    loss = losses.mean()

    forward_dict.update(
        {
            "loss": loss,
            "losses": losses,
            "dpo_logits": logits,
            "policy_chosen_logps": policy_chosen_logps,
            "policy_rejected_logps": policy_rejected_logps,
            "reference_chosen_logps": reference_chosen_logps,
            "reference_rejected_logps": reference_rejected_logps,
            "chosen_rewards": policy_chosen_logps - reference_chosen_logps,
            "rejected_rewards": policy_rejected_logps - reference_rejected_logps,
            "beta": beta,
            "label_smoothing": label_smoothing,
        }
    )

    return loss, forward_dict


# def compute_label_token_gradients(
#     model,
#     batch,
#     *,
#     beta: Optional[float] = None,
#     label_smoothing: Optional[float] = None,
#     reference_chosen_logps: Optional[torch.FloatTensor] = None,
#     reference_rejected_logps: Optional[torch.FloatTensor] = None,
# ):
#     model.train()
#     model.zero_grad(set_to_none=True)

#     loss, forward_dict = compute_loss_and_outputs(
#         model,
#         batch,
#         beta=beta,
#         label_smoothing=label_smoothing,
#         reference_chosen_logps=reference_chosen_logps,
#         reference_rejected_logps=reference_rejected_logps,
#     )

#     concatenated_logits = forward_dict["concatenated_logits"]
#     concatenated_logits.retain_grad()

#     loss.backward()

#     gradients = concatenated_logits.grad
#     if gradients is None:
#         raise RuntimeError("Gradients were not computed for concatenated logits.")

#     mask = forward_dict["loss_mask"].to(gradients.dtype)
#     masked_gradients = gradients[:, :-1, :] * mask.unsqueeze(-1)

#     num_chosen = forward_dict["num_chosen"]
#     chosen_grads = masked_gradients[:num_chosen].detach()
#     rejected_grads = masked_gradients[num_chosen:].detach()

#     model.zero_grad(set_to_none=True)

#     return {
#         "loss": loss.detach(),
#         "chosen_gradients": chosen_grads,
#         "rejected_gradients": rejected_grads,
#         "mask": mask.detach(),
#         "metadata": {
#             "beta": forward_dict["beta"],
#             "label_smoothing": forward_dict["label_smoothing"],
#         },
#     }

# %%
# Iterate over the training set and compute the gradients to dpo logits
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

# Create directory to store gradients
base_gradient_dir = "gradients"
os.makedirs(base_gradient_dir, exist_ok=True)

project_gradients_dim = 400  # Dimension of the projected space
removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings" ]
gradients_dim = 0
for name, param in model.named_parameters():
    if any([key in name for key in removing_keys]):
        continue
    if param.requires_grad:
        gradients_dim += param.numel()

# Concatenate gradients for each annotator and create projection matrix
project_matrix = (2 * np.random.randint(2, size=(gradients_dim, project_gradients_dim)) - 1).astype(float)
project_matrix *= 1 / np.sqrt(project_gradients_dim)

num_annotators = 40
counts_for_annotators = [0] * num_annotators

# %%
# Iterate over training data
def get_model_params(model):
    return [param for param in model.parameters() if param.requires_grad]

params = get_model_params(model)
model.eval()  # Set model to evaluation mode
for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Computing gradients")):
    # Get gradients for this batch
    loss, forward_dict = compute_loss_and_outputs(
        model,
        batch,
    )
    logits = forward_dict["dpo_logits"].mean()*training_args.dpo_beta
    
    # Get the chosen and rejected gradients
    grads = torch.autograd.grad(logits, params, retain_graph=False, create_graph=False)

    # flatten the gradients 
    grads = [g.flatten() for g in grads if g is not None]
    flat_grads = torch.cat(grads).cpu().numpy()  # Convert to numpy

    # Project the gradients
    projected_grads = np.dot(flat_grads, project_matrix)

    # Get annotator IDs for this batch (assuming they're in the batch)
    annotator_ids = batch.get('annotator', ['default'] * batch['chosen_input_ids'].size(0))

    # Create directory for this annotator
    annotator_dir = os.path.join(base_gradient_dir, "annotator_" + str(annotator_ids[0]))
    os.makedirs(annotator_dir, exist_ok=True)
    
    # Save the projected gradients and projection matrix
    idx = counts_for_annotators[annotator_ids[0]]
    counts_for_annotators[annotator_ids[0]] += 1
    torch.save(projected_grads, os.path.join(annotator_dir, f'projected_gradients_{idx}.pt'))



# # Keep track of example index across batches
# global_idx = 0

# # Iterate over training data
# model.train()
# for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Computing gradients")):
#     # Get the flattened gradients for this batch
#     batch_grads = compute_dpo_logits_gradient(model, batch, beta=training_args.dpo_beta)
    
#     # Project the gradients using the Rademacher matrix
#     projected_grads = torch.matmul(rademacher_matrix, batch_grads.cpu())
    
#     # Get annotator IDs for this batch
#     annotator_ids = batch.get('annotator', ['default'] * batch['chosen_input_ids'].size(0))
    
#     # Save projected gradients for each example in the batch
#     for idx, annotator in enumerate(annotator_ids):
#         # Create directory for this annotator if it doesn't exist
#         annotator_dir = os.path.join(base_gradient_dir, str(annotator))
#         os.makedirs(annotator_dir, exist_ok=True)
        
#         # Save the projected gradients
#         torch.save(projected_grads, os.path.join(annotator_dir, f'projected_gradients_{global_idx + idx}.pt'))
    
#     global_idx += len(annotator_ids)

# # Save the projection matrix for future reference
# torch.save(rademacher_matrix, os.path.join(base_gradient_dir, 'projection_matrix.pt'))
