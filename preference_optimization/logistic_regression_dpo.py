# %%
from alignment import get_datasets, get_tokenizer
from data_processing.load_ultrafeedback import load_ultrafeedback_multi_preferences
from data_processing.load_collective_alignment import load_collective_alignment
from data_processing.load_imdb_preference_with_source import load_imdb_preference_with_source
from run_simpo import apply_chat_template
import os

class data_args:
    load_multi_preference = True
    load_specific_pairs = True
    load_multi_preference_dataset = "imdb_preference_with_source"

    preprocessing_num_workers = 4
    auto_insert_empty_system_msg = True
    chat_template = None
    truncation_side = None

    preference_sources = None
    annotator_ids = "0,1,2"
    test_size = 2000 # No use, predefined

class model_args:
    model_name_or_path = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer_name_or_path = None
    model_revision = "main"
    trust_remote_code = True

class training_args:
    max_length = 2048
    max_prompt_length = 1024
    truncation_mode = "keep_end" 
    label_pad_token_id = -100
    dpo_beta = 2.0
    label_smoothing = 0.0
    
    # LoRA configurations
    lora_r = 8
    lora_alpha = 32
    lora_dropout = 0.1
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

base_gradient_dir = "gradients/imdb_qwen/"
os.makedirs(base_gradient_dir, exist_ok=True)

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

# load initial lora parameters
state_dict = torch.load(f"{base_gradient_dir}/initial_lora_weights.pt")
model.load_state_dict(state_dict, strict=False)

# print norm of the lora parameters
total_norm = 0.0
for name, param in model.named_parameters():
    if param.requires_grad and "lora_" in name:
        param_norm = param.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** (1. / 2)
print("Lora parameters norm: ", total_norm)

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
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

class logistic_regression_args:
    lr_regularization_lambda = 1
    lr_iters = 50
    use_customized_process = True

# Create directory to store gradients
project_gradients_dim = 400  # Dimension of the projected space
removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings" ]
gradients_dim = 0
for name, param in model.named_parameters():
    if any([key in name for key in removing_keys]):
        continue
    if param.requires_grad:
        gradients_dim += param.numel()

# Concatenate gradients for each annotator and create projection matrix
np.random.seed(42)  # For reproducibility   
project_matrix = (2 * np.random.randint(2, size=(gradients_dim, project_gradients_dim)) - 1).astype(float)
project_matrix *= 1 / np.sqrt(project_gradients_dim)

def customize_logistic_regression(gradients, outputs=None, labels=None, l2_strength=1e3):
    from scipy.optimize import minimize
    from sklearn.metrics import log_loss

    if outputs is not None:
        X = np.concatenate([gradients, outputs.reshape(-1, 1)], axis=1) # f_theta^star + gX
    else:
        X = gradients

    def logistic_loss(variable_coefs):
        # Reinsert the fixed coefficient
        if outputs is not None:
            fixed_index = gradients.shape[1]; fixed_value = 1
            full_coefs = np.insert(variable_coefs, fixed_index, fixed_value)
        else:
            full_coefs = variable_coefs
        logits = X @ full_coefs.reshape(-1, 1)
        if labels is not None:
            probs = 1 / (1 + np.exp(-logits))
            loss = log_loss(labels, probs).mean()
        else:
            loss = np.log(1 + np.exp(-logits)).mean()

        # L2 penalty only on the variable coefficients
        l2_penalty = l2_strength * np.sum(variable_coefs ** 2)
        return loss + l2_penalty

    def logistic_jac(variable_coefs):
        if outputs is not None:
            fixed_index = gradients.shape[1]; fixed_value = 1.0
            coefs = np.insert(variable_coefs, fixed_index, fixed_value)
        else:
            fixed_index = None
            coefs = variable_coefs
        logits = X @ coefs.reshape(-1, 1)
        probs = 1 / (1 + np.exp(-logits)); y = np.ones_like(probs) 
        grad = X.T @ (probs - y) / X.shape[0]  # shape (d+1, 1)
        grad = grad.flatten()
        if fixed_index is not None:
            grad = np.delete(grad, fixed_index)  # remove derivative for fixed coefficient
        grad += 2 * l2_strength * variable_coefs  # L2 grad
        return grad
    
    initial_guess = np.zeros(X.shape[1] - 1) if outputs is not None else np.zeros(X.shape[1])
    result = minimize(logistic_loss, initial_guess, method='BFGS', options={'maxiter': logistic_regression_args.lr_iters})
    print(result)

    # evaluate the trained model
    if labels is not None:
        accuracy = np.mean(np.round(1 / (1 + np.exp(-X @ result.x.reshape(-1, 1)))) == labels)
        print("Accuracy: ", accuracy)

    return result.x

def fit_linear_model(gradients, outputs=None, labels=None, seed=0, use_customized_process=True):
    if use_customized_process:
        proj_coef = customize_logistic_regression(gradients, outputs=outputs, l2_strength=logistic_regression_args.lr_regularization_lambda)
        print("L2 norm before projection", np.linalg.norm(proj_coef))
    else:
        if labels is None:
            # randomly assign labels as 0 or 1
            labels = np.random.binomial(n=1, p=0.7, size=gradients.shape[0])
            # reverse the gradients for the 0 labels
            mask = np.copy(labels)
            mask[labels == 0] = -1
            mask = mask.reshape(-1, 1)
            gradients = gradients*mask
        else:
            ref_label = labels[0]
            origin_labels = np.copy(labels)
            labels[origin_labels == ref_label] = 1
            labels[origin_labels != ref_label] = -1

            mask = np.copy(labels)
            mask[labels == 0] = -1
            mask = mask.reshape(-1, 1)
            gradients = gradients*mask

        if outputs is None:
            # estimate parameters: train a logistic regression model
            clf = LogisticRegression(penalty='l2',  solver='lbfgs', C=1/logistic_regression_args.lr_regularization_lambda) 
            clf.fit(gradients, labels)
            print("Linear regression score: ", clf.score(gradients, labels))
            proj_coef = clf.coef_.copy().flatten().reshape(-1, 1)
            print("L2 norm before projection", np.linalg.norm(proj_coef))
        else:
            # concatenate outputs
            print("Also using outputs for linear regression")
            outputs = outputs*mask.flatten()
            gradients = np.concatenate([gradients, -outputs.reshape(-1, 1)], axis=1)
            # estimate parameters: train a logistic regression model
            clf = LogisticRegression(penalty='l2',  solver='lbfgs', C=1/logistic_regression_args.lr_regularization_lambda, fit_intercept=False) 
            clf.fit(gradients, labels)
            print("Linear regression score: ", clf.score(gradients, labels))
            proj_coef = clf.coef_.copy().flatten().reshape(-1, 1)[:-1] # remove the last column corresponding to the output
            print("L2 norm before projection", np.linalg.norm(proj_coef))
        
    # convert the coefficients to the original space
    if project_matrix is not None:
        coef = project_matrix @ proj_coef.flatten()
    else:
        coef = proj_coef.flatten()
    print("L2 norm after projection", np.linalg.norm(coef))

    return coef

# load gradients of corresponding annotators
def get_annotator_gradients(annotator_id):
    tmp_gradient_dir = base_gradient_dir + f"annotator_{annotator_id}/"
    gradient_list = []
    for file in os.listdir(tmp_gradient_dir):
        if file.startswith("projected_gradients_"):
            gradient_feature = torch.load(os.path.join(tmp_gradient_dir, file))
            gradient_list.append(gradient_feature)
    gradient_list = np.array(gradient_list)
    return gradient_list

def load_gradients(annotator_ids):
    all_annotator_gradients = []
    for annotator_id in annotator_ids:
        annotator_gradients = get_annotator_gradients(annotator_id)
        all_annotator_gradients.append(annotator_gradients)
    all_annotator_gradients = np.concatenate(all_annotator_gradients, axis=0)
    return all_annotator_gradients


def generate_state_dict(model, state_dict, coef, device="cpu", removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings", "quant", "absmax"]):
    new_state_dict = {}; cur_len = 0
    for key, param in model.named_parameters():
        if not param.requires_grad: continue
        param_len = param.numel()
        if any([rkey in key for rkey in removing_keys]):
            continue
            # new_state_dict[key] = state_dict[key].clone()
        else:
            assert "lora" in key
            new_state_dict[key] = state_dict[key].clone().to(device) + \
                torch.Tensor(coef[cur_len:cur_len+param_len].reshape(param.shape)).to(device)
            cur_len += param_len
    return new_state_dict

annotator_ids = [int(idx.strip()) for idx in data_args.annotator_ids.split(",") if idx.strip()]

all_annotator_gradients = load_gradients(annotator_ids)

# fit a linear model
coef = fit_linear_model(all_annotator_gradients, use_customized_process=logistic_regression_args.use_customized_process)
new_state_dict = generate_state_dict(model, state_dict, coef, device=model.device)
pretrain_state_dict = state_dict
finetuned_state_dict = new_state_dict
model.load_state_dict(pretrain_state_dict, strict=False)
model.load_state_dict(finetuned_state_dict, strict=False)

# %%
# evaluate the model after DPO training
# write a plain evaluation loop here
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
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

    # average logps over non-pad tokens
    sequence_logps = sequence_logps / token_counts.clamp_min(1)

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

model.eval()
all_preds = []
all_labels = []

# split metric by annotators
annotator_accuracy = {annotator_id: {"preds": [], "labels": []} for annotator_id in annotator_ids}

model.eval()
for step, batch in enumerate(tqdm(test_dataloader)):
    with torch.no_grad():
        _, forward_dict = compute_loss_and_outputs(model, batch)
    dpo_logits = forward_dict["dpo_logits"].cpu().numpy()
    preds = (dpo_logits > 0).astype(int).flatten()
    all_preds.extend(preds.tolist())
    # all labels are 1 since we have already reversed the gradients for the negative labels
    all_labels.extend([1]*len(preds))

    # update annotator accuracy
    for annotator_id in batch.get("annotator", []):
        annotator_accuracy[annotator_id]["preds"].extend(preds.tolist())
        annotator_accuracy[annotator_id]["labels"].extend([1]*len(preds))

accuracy = accuracy_score(all_labels, all_preds)
print(f"Overall accuracy after DPO training: {accuracy}")
for annotator_id in annotator_ids:
    annotator_acc = accuracy_score(annotator_accuracy[annotator_id]["labels"], annotator_accuracy[annotator_id]["preds"])
    print(f"Annotator {annotator_id} accuracy after DPO training: {annotator_acc}")

# %%
# write the results to a csv file
import pandas as pd
os.makedirs("./results/dpo_qwen/", exist_ok=True)

if not os.path.exists(os.path.join("./results/dpo_qwen/", "dpo_evaluation_results.csv")):
    results = []
    results.append({"annotator_id": "overall", "annotator_ids": data_args.annotator_ids, "accuracy": accuracy})
    for annotator_id in annotator_ids:
        annotator_acc = accuracy_score(annotator_accuracy[annotator_id]["labels"], annotator_accuracy[annotator_id]["preds"])
        results.append({"annotator_id": annotator_id, "annotator_ids": data_args.annotator_ids, "accuracy": annotator_acc})
    results = pd.DataFrame(results)
    results.to_csv(os.path.join("./results/dpo_qwen/", "dpo_evaluation_results.csv"), index=False)
    print(f"Results saved to {os.path.join('./results/dpo_qwen/', 'dpo_evaluation_results.csv')}")
else:   
    # load existing results and append new results
    results = pd.read_csv(os.path.join("./results/dpo_qwen/", "dpo_evaluation_results.csv"))
    new_results = []
    for annotator_id in annotator_ids:
        annotator_acc = accuracy_score(annotator_accuracy[annotator_id]["labels"], annotator_accuracy[annotator_id]["preds"])
        new_results.append({"annotator_id": annotator_id, "annotator_ids": data_args.annotator_ids, "accuracy": annotator_acc})
    new_results = pd.DataFrame(new_results)
    results = pd.concat([results, new_results], axis=0)
    results.to_csv(os.path.join("./results/dpo_qwen/", "dpo_evaluation_results.csv"), index=False)
