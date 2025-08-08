import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from absl import flags
from collections import OrderedDict
from functools import reduce

#------------------------------------------------------------------
# Utility functions for Jacobian computation
#------------------------------------------------------------------

FLAGS = flags.FLAGS
PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'
EOS_TOKEN = '<|endoftext|>'

def _select_params(model, param_filter=None):
    names, params = [], []
    for n, p in model.named_parameters():
        print(f"Checking param {n} with requires_grad={p.requires_grad}")
        #if p.requires_grad and (param_filter is None or param_filter(n)):
        if p.requires_grad:
            names.append(n); params.append(p)
    if not params:
        raise ValueError("No params selected; adjust param_filter.")
    return names, params

def _logprobs_from_logits(logits, labels):
    # logits: (B, L, V), labels: (B, L)
    logp = torch.log_softmax(logits, dim=-1)
    # guard -100 so gather won’t crash
    labels_safe = labels.clone()
    labels_safe[labels_safe == -100] = 0
    return logp.gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1)  # (B, L)

def validate_tokenwise_simple(
    model_theta1,
    model_theta2,
    batch,                       # dict with input_ids (B,L), attention_mask (B,L), labels (B,L)
    *,
    param_filter=None,           # e.g. lambda n: "lm_head" in n (keep P small)
    max_tokens_per_sample=None,  # optional cap per sample
):
    """
    Computes per-token Jacobians for tokens where labels != -100, using only what's in `batch`.
    Returns lists per sample because T_b varies.
    """
    device = next(model_theta1.parameters()).device
    model_theta1.eval(); model_theta2.eval()

    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels         = batch["labels"].to(device)       # -100 on prompt/pad
    B, L = labels.shape

    # 1) Forward passes
    out1 = model_theta1(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logp1_full = _logprobs_from_logits(out1.logits, labels)        # (B, L)

    with torch.no_grad():
        out2 = model_theta2(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logp2_full = _logprobs_from_logits(out2.logits, labels)    # (B, L)

    # 2) Param subset + Δθ in the SAME order
    param_names, params1 = _select_params(model_theta1, param_filter)
    name_to_p2 = dict(model_theta2.named_parameters())
    params2 = [name_to_p2[n] for n in param_names]
    P = sum(p.numel() for p in params1)

    with torch.no_grad():
        theta1_sub = torch.cat([p.detach().reshape(-1) for p in params1])
        theta2_sub = torch.cat([p.detach().reshape(-1) for p in params2])
        delta = theta2_sub - theta1_sub                                       # (P,)

    # 3) Tokenwise Jacobian per sample (only where labels != -100)
    J_list, f1_list, f2_list, f_lin_list = [], [], [], []
    mae_list, rmse_list, pos_list = [], [], []

    for b in range(1):
        valid_idx = torch.nonzero(labels[b] != -100, as_tuple=False).squeeze(-1)  # response tokens
        if valid_idx.numel() == 0:
            continue
        if max_tokens_per_sample is not None and valid_idx.numel() > max_tokens_per_sample:
            valid_idx = valid_idx[:max_tokens_per_sample]
        T_b = valid_idx.numel()

        J_b = torch.zeros(T_b, P, device=device, dtype=logp1_full.dtype)
        f1_b = logp1_full[b, valid_idx].clone()      # (T_b,)
        f2_b = logp2_full[b, valid_idx].clone()      # (T_b,)

        # SAC-style: autograd per scalar token
        for t, pos in enumerate(valid_idx.tolist()):
            model_theta1.zero_grad(set_to_none=True)
            s_bt = logp1_full[b, pos]               # scalar log-prob at θ1
            grads = torch.autograd.grad(
                outputs=s_bt, inputs=params1,
                retain_graph=(t < T_b - 1),
                create_graph=False,
                allow_unused=True
            )
            row = []
            for p, g in zip(params1, grads):
                row.append(torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1))
            J_b[t] = torch.cat(row, dim=0)

        f_lin_b = f1_b + J_b @ delta                 # (T_b,)
        mae  = (f2_b - f_lin_b).abs().mean().item()
        rmse = torch.sqrt(((f2_b - f_lin_b)**2).mean()).item()

        J_list.append(J_b.detach())
        f1_list.append(f1_b.detach())
        f2_list.append(f2_b.detach())
        f_lin_list.append(f_lin_b.detach())
        mae_list.append(mae); rmse_list.append(rmse)
        pos_list.append(valid_idx.detach().cpu())

    return {
        "J_list": J_list,
        "f1_list": f1_list,
        "f2_list": f2_list,
        "f_lin_list": f_lin_list,
        "mae_list": mae_list,
        "rmse_list": rmse_list,
        "param_names": param_names,
        "positions": pos_list,   # which token indices we used per sample
    }


# def compute_sample_jacobian_wrt_params(
#     model_theta1,                    # model at epoch 1 (expansion point)
#     model_theta2,                    # model at epoch 2 (ground truth)
#     batch,                           # dict: input_ids (B,L), attention_mask (B,L), labels (B,L)
#     pos_idx,                         # LongTensor (B,), position per sample to evaluate
#     param_filter=None,               # optional: subset layers (e.g., last block / lm_head)
# ):
#     """
#     Build J of shape (B, P), where each row is grad of a single scalar output
#     (log-prob of label at pos_idx[b]) w.r.t. selected parameters.
#     SAC-style: autograd.grad per sample; no file I/O.
#     """
#     device = next(model_theta1.parameters()).device
#     model_theta1.eval(); model_theta2.eval()

#     input_ids      = batch["input_ids"].to(device)
#     attention_mask = batch["attention_mask"].to(device)
#     labels         = batch["labels"].to(device)
#     B, L = labels.shape
#     assert pos_idx.shape[0] == B, "pos_idx must have one position per sample"

#     # Param selection (order matters and must match across theta1/theta2)
#     param_names, params1 = select_params(model_theta1, param_filter)
#     name_to_p2 = dict(model_theta2.named_parameters())
#     params2 = [name_to_p2[n] for n in param_names]

#     P = sum(p.numel() for p in params1)
#     J = torch.zeros(B, P, device=device)

#     # Forward once at theta1 (build graph for autograd)
#     out1 = model_theta1(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
#     logp1_full = F.log_softmax(out1.logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B,L)

#     # f1: pick one scalar per sample (its chosen position)
#     f1 = []
#     for b in range(B):
#         pos = int(pos_idx[b].item())
#         if labels[b, pos].item() == -100:
#             raise ValueError(f"labels[{b},{pos}] == -100; choose a valid position.")
#         f1.append(logp1_full[b, pos])
#     f1 = torch.stack(f1)  # (B,)

#     # Build J row-by-row (one grad per sample)
#     # (retain_graph True for all but the last to reuse the graph)
#     for b in range(B):
#         model_theta1.zero_grad(set_to_none=True)
#         s_b = f1[b]  # scalar: log-prob at selected position for sample b
#         grads = torch.autograd.grad(
#             outputs=s_b, inputs=params1,
#             retain_graph=(b < B - 1), create_graph=False, allow_unused=True
#         )
#         row = []
#         for p, g in zip(params1, grads):
#             row.append(torch.zeros_like(p, device=device).reshape(-1) if g is None else g.reshape(-1))
#         J[b] = torch.cat(row, dim=0)

#     # Δθ for the SAME param subset and order
#     with torch.no_grad():
#         theta1_subset = torch.cat([p.detach().reshape(-1) for p in params1])
#         theta2_subset = torch.cat([p.detach().to(device).reshape(-1) for p in params2])
#     delta = theta2_subset - theta1_subset  # (P,)

#     # First-order prediction at theta2
#     f_lin = f1 + J @ delta  # (B,)

#     # Ground truth f2 at theta2 (same positions)
#     with torch.no_grad():
#         out2 = model_theta2(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
#         logp2_full = F.log_softmax(out2.logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B,L)
#         f2 = torch.stack([logp2_full[b, int(pos_idx[b].item())] for b in range(B)])  # (B,)

#     # Metrics
#     mae  = (f2 - f_lin).abs().mean().item()
#     rmse = torch.sqrt(((f2 - f_lin)**2).mean()).item()
#     return {"J": J, "f1": f1, "f2": f2, "f_lin": f_lin, "mae": mae, "rmse": rmse, "param_names": param_names}



#------------------------------------------------------------------
# Utility functions for model parameter loading
#------------------------------------------------------------------

def load_params_from_vector(model, vector, param_filter=None):
    """Load parameters from a flat vector into model (in-place)."""
    device = next(model.parameters()).device
    idx = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if param_filter is None or param_filter(name):
            numel = p.numel()
            new_val = vector[idx: idx + numel].view_as(p).to(device)
            with torch.no_grad():
                p.copy_(new_val)
            idx += numel
        else:
            # skip params not in filter
            continue
    if idx != vector.numel():
        raise ValueError(f"Vector has {vector.numel()} elems, loaded {idx}")
    
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


    


model_name = "meta-llama/Llama-3.2-1B"  

# # Two independent model copies
# model_theta_star = AutoModelForCausalLM.from_pretrained(model_name)
# model_theta = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)

if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

model_theta_star = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)
model_2 = AutoModelForCausalLM.from_pretrained("models/model2", trust_remote_code=True)
model_theta      = AutoModelForCausalLM.from_pretrained("models/model1", trust_remote_code=True)



model_theta_star.resize_token_embeddings(len(tokenizer))
model_theta.resize_token_embeddings(len(tokenizer))


# # Count params the loader will try to fill:
# load_uses_named = True  # your load_params_from_vector uses named_parameters()
# total_model_params = sum(p.numel() for n,p in model_theta_star.named_parameters() if p.requires_grad)
# print("model total requires_grad params:", total_model_params)


# # Load p_epoch1 into model_theta1
# ckpt1 = torch.load("./params/pstar_epoch15.pt")
# vector1 = ckpt1["params"] if "params" in ckpt1 else ckpt1  # adjust key if needed
# load_params_from_vector(model_theta_star, vector1)

# # Load p_epoch2 into model_theta2 (or use your 2nd epoch path)
# ckpt2 = torch.load("./params/p_epoch16.pt")
# vector2 = ckpt2["params"] if "params" in ckpt2 else ckpt2
# load_params_from_vector(model_theta, vector2)

device = torch.device("cuda")
model_theta_star.to(device)
model_theta.to(device)

# Load a sample batch
eval_pref_dataset = load_dataset(
    "Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_minlength",
    split="train" 
)
remove_columns = ['output', 'text', 'alpaca_text', 'y_ref', 'y_1', 'y_2', 'y_w', 'y_w_alpaca', 'y_l', 'y_l_alpaca', 'y_w_score', 'y_l_score', 'score_diff', 'prompt', 'alpaca_prompt']
pref_dataset = eval_pref_dataset.map(
    process_dataset,
    batched=True,
    num_proc=32,
    remove_columns=remove_columns,
)


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token  # exactly what you did in training
# IMPORTANT: resize before loading vectors
model_theta_star.resize_token_embeddings(len(tokenizer))
model_theta.resize_token_embeddings(len(tokenizer))

def collate_pref_batch(examples):
    """
    Keep only the fields we need and keep them as python lists
    so we can tokenize them together (like your training code).
    """
    batch = {
        "query":        [ex["query"]        for ex in examples],
        "response_w":   [ex["response_w"]   for ex in examples],
        "response_l":   [ex["response_l"]   for ex in examples],
    }
    return batch

@torch.no_grad()
def get_small_pref_batch_tensors(
    tokenizer,
    dataset,                 # e.g., your eval_pref_dataset
    device="cuda",
    batch_size=2,            # small B (1–4)
    max_query_len=128,
    max_resp_len=64,
    max_new_tokens=256,
    pad_to_max_length=False, # True mimics padding='max_length' in your code
):
    """
    Returns:
      pref_batch (dict of lists)
      pref_query_tensors: LongTensor [B, Lq]
      pref_response_w_tensors: LongTensor [B, Lr]
      pref_response_l_tensors: LongTensor [B, Lr]
    """
    print(dataset.column_names)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                        collate_fn=collate_pref_batch)
    pref_batch = next(iter(loader))  # one small batch of raw strings

    # 1) Tokenize queries
    query_enc = tokenizer(
        pref_batch["query"],
        padding=("max_length" if pad_to_max_length else True),
        truncation=True,
        max_length=max_query_len,
        return_tensors="pt",
    )
    pref_query_tensors = query_enc.input_ids.to(device)

    # 2) Tokenize responses together (W then L) so they get the same sequence length
    all_pref = pref_batch["response_w"] + pref_batch["response_l"]
    resp_enc = tokenizer(
        all_pref,
        padding=("max_length" if pad_to_max_length else True),
        truncation=True,
        max_length=max_resp_len + max_new_tokens,  # match your training code
        return_tensors="pt",
    )
    tokenized = resp_enc.input_ids  # LongTensor [B_w + B_l, Lr]
    B_w = len(pref_batch["response_w"])

    # Split back into winner/loser chunks
    pref_response_w_tensors = tokenized[:B_w].to(device)        # [B, Lr]
    pref_response_l_tensors = tokenized[B_w:].to(device)        # [B, Lr]


    # print(pref_response_w_tensors.shape, pref_response_l_tensors.shape)
    # print(pref_response_w_tensors[0])

    return pref_batch, pref_query_tensors, pref_response_w_tensors, pref_response_l_tensors


pref_batch, pref_query_ids, pref_resp_w_ids, pref_resp_l_ids = get_small_pref_batch_tensors(
    tokenizer=tokenizer,
    dataset=pref_dataset,   
    device=device,
    batch_size=2,                # tiny batch
    max_query_len=128,
    max_resp_len=64,
    max_new_tokens=256,
    pad_to_max_length=False      # set True to mimic your TPU branch
)
print(pref_query_ids.shape, pref_resp_w_ids.shape, pref_resp_l_ids.shape)


def build_teacher_forcing_inputs(tokenizer, pref_query_ids, pref_resp_w_ids):
    """
    input_ids = [query, response_w[:-1]]
    labels    = [-100*len(query), response_w]
    Also returns query_len per sample (count of non-pad tokens in query).
    """
    pad_id = tokenizer.pad_token_id
    B = pref_query_ids.size(0)

    # lengths (handle left/right padding)
    query_len = (pref_query_ids != pad_id).sum(dim=1)  # (B,)
    resp_len  = (pref_resp_w_ids != pad_id).sum(dim=1) # (B,)

    input_ids = torch.cat([pref_query_ids, pref_resp_w_ids[:, :-1]], dim=1)  # (B, Lq + Lr-1)
    labels = torch.cat([
        torch.full_like(pref_query_ids, -100),   # mask prompt in loss
        pref_resp_w_ids
    ], dim=1)  # (B, Lq + Lr)

    # Align labels with logits[:, :-1] vs labels[:, 1:]
    labels = labels[:, 1:1 + input_ids.size(1)]  # (B, Lq + Lr - 1)
    attention_mask = torch.ones_like(input_ids)

    return input_ids, labels, attention_mask, query_len, resp_len

# def validate_one_small_batch(
#     tokenizer,
#     model_theta1, model_theta2,
#     pref_batch, pref_query_ids, pref_resp_w_ids,
#     pos_rel_in_resp,                 # LongTensor (B,) position inside response_w you want
#     param_filter=None,               # e.g. lambda n: "lm_head" in n or "layers.31." in n
# ):
#     device = next(model_theta1.parameters()).device
#     model_theta1.eval(); model_theta2.eval()

#     # Build teacher-forcing inputs
#     input_ids, labels_shifted, attn_mask, query_len, resp_len = build_teacher_forcing_inputs(
#         tokenizer, pref_query_ids, pref_resp_w_ids
#     )
#     input_ids      = input_ids.to(device)
#     labels_shifted = labels_shifted.to(device)
#     attn_mask      = attn_mask.to(device)

#     # Also build the "full" labels ([-100*Lq, response_w]) to index by absolute label position
#     labels_full = torch.cat([
#         torch.full_like(pref_query_ids, -100),
#         pref_resp_w_ids
#     ], dim=1).to(device)  # (B, L_full)

#     # Forward both models to get f2 ground-truth at the chosen positions
#     with torch.no_grad():
#         out2 = model_theta2(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
#         # full log-probs w.r.t. labels_full index space:
#         logp2_full = logprobs_from_logits(out2.logits, labels_full)  # (B, L_full)

#     # Compute absolute label positions for each sample:
#     # For response token r (0-based in response_w), the absolute label index is Lq + r
#     pad_id = tokenizer.pad_token_id
#     Lq = (pref_query_ids != pad_id).sum(dim=1)              # (B,)
#     pos_label_idx = Lq + pos_rel_in_resp                    # (B,) absolute indices in labels_full

#     # Build batch_inputs with "labels_full" for Jacobian function
#     batch_inputs = {
#         "input_ids": input_ids,
#         "attention_mask": attn_mask,
#         "labels_full": labels_full,
#     }

#     # J and f1 at θ1
#     with torch.inference_mode(False):
#         J, f1, param_names = compute_sample_jacobian_wrt_params(
#             model_theta1, batch_inputs, pos_label_idx, param_filter=param_filter
#         )

#     # Δθ for the SAME param subset/order
#     name_to_p2 = dict(model_theta2.named_parameters())
#     params1 = [p for n, p in model_theta1.named_parameters()
#                if p.requires_grad and (param_filter is None or param_filter(n))]
#     params2 = [name_to_p2[n] for n, p in model_theta1.named_parameters()
#                if p.requires_grad and (param_filter is None or param_filter(n))]

#     with torch.no_grad():
#         theta1_subset = torch.cat([p.detach().reshape(-1) for p in params1])
#         theta2_subset = torch.cat([p.detach().reshape(-1) for p in params2])
#         delta = theta2_subset - theta1_subset  # (P,)

#     # First-order prediction
#     f_lin = f1 + J @ delta  # (B,)

#     # Ground-truth f2 at the same absolute label positions
#     f2 = torch.stack([logp2_full[b, int(pos_label_idx[b].item())] for b in range(input_ids.size(0))])

#     # Metrics
#     mae  = (f2 - f_lin).abs().mean().item()
#     rmse = torch.sqrt(((f2 - f_lin)**2).mean()).item()

#     return {
#         "f1": f1.detach(), "f2": f2.detach(), "f_lin": f_lin.detach(),
#         "J": J.detach(), "delta": delta.detach(),
#         "mae": mae, "rmse": rmse, "param_names": param_names,
#         "chosen_positions": pos_label_idx.detach().cpu()
#     }


# 1) build teacher-forcing batch
# input_ids = [query, resp_w[:-1]]
input_ids = torch.cat([pref_query_ids, pref_resp_w_ids[:, :-1]], dim=1)  # (B, L)
attention_mask = torch.ones_like(input_ids)

# labels_full = [-100 * len(query), resp_w]
labels_full = torch.cat([
    torch.full_like(pref_query_ids, -100),
    pref_resp_w_ids
], dim=1)


labels = labels_full[:, 1:1 + input_ids.size(1)]  # (B, L)

batch = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels": labels,
}

param_filter = lambda n: ("lm_head" in n)  
out = validate_tokenwise_simple(
    model_theta1=model_theta_star,   
    model_theta2=model_theta,        
    batch=batch,
    param_filter=param_filter,
    max_tokens_per_sample=64,        
)


mean_mae  = sum(out["mae_list"])  / max(1, len(out["mae_list"]))
mean_rmse = sum(out["rmse_list"]) / max(1, len(out["rmse_list"]))
print(f"[tokenwise] samples={len(out['J_list'])}  mean_MAE={mean_mae:.3e}  mean_RMSE={mean_rmse:.3e}")


if out["J_list"]:
    J0, f10, f20, f_lin0 = out["J_list"][0], out["f1_list"][0], out["f2_list"][0], out["f_lin_list"][0]
    print("sample0 shapes:", J0.shape, f10.shape, f20.shape, f_lin0.shape)
