import argparse
import random
from typing import List, Dict

import torch
from tqdm import tqdm
from datasets import load_dataset
from sklearn.cluster import KMeans
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.py.demo import model_editing

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gpt2-xl", type=str)
parser.add_argument("--dataset", default="azhx/counterfact", type=str)
parser.add_argument("--sample", default=30, type=int)
parser.add_argument("--groups", default=5, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--alg", default="MEND", type=str)
args = parser.parse_args()

device = args.device if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
random.seed(args.seed)

def make_input_and_target(request: Dict):
    prompt = request["prompt"].format(request["subject"])
    target = request["target_new"]["str"]
    target_eval = target if target.startswith(" ") else " " + target
    return prompt, target_eval

def check_efficacy_greedy(model, tokenizer, prompt, target_eval):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    tgt_ids = tokenizer(target_eval, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    out = model.generate(
        **enc,
        max_new_tokens=len(tgt_ids),
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )[0]
    gen_ids = out[enc["input_ids"].shape[1]:][:len(tgt_ids)]
    return torch.equal(gen_ids, tgt_ids)

def kmeans_cosine(X: torch.Tensor, k: int, seed: int = 42, iters: int = 300):
    """X: [N, D]"""
    assert X.dim() == 2
    X = X.detach().cpu()
    Xn = X / (X.norm(dim=1, keepdim=True) + 1e-12)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=iters, algorithm="auto", verbose=0)
    km.fit(Xn.numpy())
    return torch.from_numpy(km.labels_).long()

if args.sample == 0:
    ds = load_dataset(args.dataset, split="test")
else:
    ds = load_dataset(args.dataset, split=f"test[:{args.sample}]")

tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading a temporary model to fetch lm_head for grouping...")
model_tmp = AutoModelForCausalLM.from_pretrained(args.model).to(device)
model_tmp.eval()

recs: List[Dict] = []
target_embs = []

print("Collecting requests/prompts/targets/neighbors and target embeddings...")
for dp in tqdm(ds, total=len(ds)):
    request = dp["requested_rewrite"] 
    prompt, target_eval = make_input_and_target(request)

    neighbors = dp.get("paraphrase_prompts", [])
    neighbor = neighbors[0] if len(neighbors) > 0 else prompt


    tid_last = tokenizer(target_eval, return_tensors="pt", add_special_tokens=False).input_ids[0][-1].item()
    tgt_emb = model_tmp.lm_head.weight[tid_last].detach().float().cpu()

    recs.append({
        "request": request, 
        "prompt": prompt,
        "target_eval": target_eval,
        "neighbor": neighbor,
        "tid_last": tid_last,
    })
    target_embs.append(tgt_emb)

del model_tmp
if device == "cuda":
    torch.cuda.empty_cache()

target_embs = torch.stack(target_embs, dim=0)  # [N, d]

labels = kmeans_cosine(target_embs, k=args.groups, seed=args.seed)
groups = [[] for _ in range(args.groups)]
for i, lab in enumerate(labels.tolist()):
    groups[lab].append(i)

def apply_group_edits_and_eval(idx_list: List[int]):
    if len(idx_list) == 0:
        return 0, 0

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    for i in idx_list:
        req_list = [recs[i]["request"]] 
        model, _orig = model_editing(model, tokenizer, req_list, alg_name=args.alg)

    suc = 0
    for i in idx_list:
        suc += int(check_efficacy_greedy(model, tokenizer, recs[i]["prompt"], recs[i]["target_eval"]))

    par = 0
    for i in idx_list:
        par += int(check_efficacy_greedy(model, tokenizer, recs[i]["neighbor"], recs[i]["target_eval"]))

    return suc, par

print("\n=== Per-group results (independent edits per group) ===")
total_suc, total_par, total_cnt = 0, 0, 0
for gi, idxs in enumerate(groups):
    if len(idxs) == 0:
        print(f"Group {gi:>3}: size=0 (skipped)")
        continue
    suc, par = apply_group_edits_and_eval(idxs)
    eff = 100.0 * suc / len(idxs)
    par_rate = 100.0 * par / len(idxs)
    print(f"Group {gi:>3}: size={len(idxs)}, success={suc}, efficacy={eff:.2f}%, paraphrase={par_rate:.2f}%")
    total_suc += suc
    total_par += par
    total_cnt += len(idxs)

if total_cnt > 0:
    overall_eff = 100.0 * total_suc / total_cnt
    overall_par = 100.0 * total_par / total_cnt
    print(f"\nOverall efficacy (weighted): {overall_eff:.2f}%  (total success {total_suc}/{total_cnt})")
    print(f"Overall paraphrase (weighted): {overall_par:.2f}%  (total success {total_par}/{total_cnt})")
else:
    print("\nNo data to evaluate.")
