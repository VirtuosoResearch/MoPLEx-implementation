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

FLAGS = flags.FLAGS
PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'
EOS_TOKEN = '<|endoftext|>'

def _select_params(model, param_filter=None):
    names, params = [], []
    for n, p in model.named_parameters():
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
    print(batch.keys())
    B, L = labels.shape

    # 1) Forward passes
    out1 = model_theta1(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    logp1_full = _logprobs_from_logits(out1, labels)        # (B, L)

    with torch.no_grad():
        out2 = model_theta2(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        logp2_full = _logprobs_from_logits(out2, labels)    # (B, L)

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

def _sum_logprobs_over_response(model, input_ids, resp_mask):
    """
    Compute the total log-probability over the response segment only.

    Args:
        model: Hugging Face Causal LM model.
        input_ids: (B, L) input token IDs.
        resp_mask: (B, L-1) mask indicating which positions (in labels) belong to the response.

    Returns:
        sum_logprobs: (B,) sum of log-probs over response tokens for each sample.
        token_logprobs_masked: (B, L-1) per-token log-probs (0 for non-response positions).
    """
    # Causal LM predicts the next token, so logits are aligned with input_ids[:-1]
    logits = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False
    ).logits  # (B, L, V)

    # Shift labels and logits to align (labels = next token)
    labels = input_ids[:, 1:]       # (B, L-1)
    logits = logits[:, :-1, :]      # (B, L-1, V)

    # Convert logits to log-probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # (B, L-1, V)

    # Gather log-prob for the correct label at each position
    token_logprobs = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, L-1)

    # Mask out non-response tokens (set them to 0)
    resp_mask_f = resp_mask.to(token_logprobs.dtype)
    token_logprobs_masked = token_logprobs * resp_mask_f

    # Sum over response tokens
    sum_logprobs = token_logprobs_masked.sum(dim=1)  # (B,)

    return sum_logprobs, token_logprobs_masked

def approximation(model_theta1, model_theta2, pref_batch, pref_query_ids, pref_resp_w_ids, pref_resp_l_ids, device="cuda"):
    model_theta1.to(device)
    model_theta2.to(device)

    input_ids_w = torch.cat((pref_query_ids, pref_resp_w_ids), dim=1)  # (B, L)
    input_ids_l = torch.cat((pref_query_ids, pref_resp_l_ids), dim=1)  # (B, L)
    
    mask_w = torch.cat((torch.zeros_like(pref_query_ids), torch.ones_like(pref_resp_w_ids)), dim=1)[:, :-1]  # (B, L)
    mask_l = torch.cat((torch.zeros_like(pref_query_ids), torch.ones_like(pref_resp_l_ids)), dim=1)[:, :-1]  # (B, L)
    assert mask_w.shape == mask_l.shape, f"mask_w and mask_l should have the same shape, {mask_w.shape}, {mask_l.shape}."
    mask = mask_w

    t1_w_sum, t1_w_tok = _sum_logprobs_over_response(model_theta1, input_ids_w, mask_w)
    t1_l_sum, t1_l_tok = _sum_logprobs_over_response(model_theta1, input_ids_l, mask_l)
    
    grad_theta1_w = torch.autograd.grad(t1_w_sum.sum(), model_theta1.parameters(), allow_unused=True, retain_graph=False, create_graph=False)
    grad_theta1_l = torch.autograd.grad(t1_l_sum.sum(), model_theta1.parameters(), allow_unused=True, retain_graph=False, create_graph=False)

    def flat_vec_from_tensors(tensors, like_params):
        flat_parts = []
        for g, p in zip(tensors, like_params):
            if g is None:
                flat_parts.append(torch.zeros_like(p).reshape(-1))
            else:
                flat_parts.append(g.reshape(-1))
        return torch.cat(flat_parts, dim=0)

    gradvec_w = flat_vec_from_tensors(grad_theta1_w, model_theta1.parameters())  # (P,)
    gradvec_l = flat_vec_from_tensors(grad_theta1_l, model_theta1.parameters())  # (P,)

    # theta2 outputs
    with torch.no_grad():
        t2_w_sum, t2_w_tok = _sum_logprobs_over_response(model_theta2, input_ids_w, mask_w)
        t2_l_sum, t2_l_tok = _sum_logprobs_over_response(model_theta2, input_ids_l, mask_l)
    
    params1 = [p for p in model_theta1.parameters()]
    params2 = [p for p in model_theta2.parameters()]
    with torch.no_grad():
        delta_theta = torch.cat([(p2.detach() - p1.detach()).reshape(-1) for p1, p2 in zip(params1, params2)], dim=0)  # (P,)

    print(delta_theta.shape)
    # --- linear approximation: <grad, delta> is a scalar ---
    corr_w = gradvec_w @ delta_theta   # scalar
    corr_l = gradvec_l @ delta_theta   # scalar

    t1_w_lin = t1_w_sum + corr_w       # (B,) + scalar -> (B,)
    t1_l_lin = t1_l_sum + corr_l       # (B,) + scalar -> (B,)
    print(t1_w_sum, t1_w_lin, t2_w_sum)
    print(t1_l_sum, t1_l_lin, t2_l_sum)


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_theta_star = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model_theta      = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model_theta_star.resize_token_embeddings(len(tokenizer))
    model_theta.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    eval_pref_dataset = load_dataset(
        "Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_minlength",
        split="train" 
    )

    remove_columns = ['output', 'text', 'alpaca_text', 'y_ref', 'y_1', 'y_2', 'y_w', 'y_w_alpaca', 'y_l', 'y_l_alpaca', 'y_w_score', 'y_l_score', 'score_diff', 'prompt', 'alpaca_prompt']

    print(eval_pref_dataset)
    print(eval_pref_dataset)
    exit(0)

    pref_dataset = eval_pref_dataset.map(
        process_dataset,
        batched=True,
        num_proc=32,
        remove_columns=remove_columns,
    )

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

    results = approximation(
        model_theta1=model_theta_star,
        model_theta2=model_theta,
        pref_batch=pref_batch,
        pref_query_ids=pref_query_ids,
        pref_resp_w_ids=pref_resp_w_ids,
        pref_resp_l_ids=pref_resp_l_ids,
        device=device,
    )