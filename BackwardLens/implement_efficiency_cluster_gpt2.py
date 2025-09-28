import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import argparse
import math
import random

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gpt2-xl", type=str)
parser.add_argument("--sample", default=500, type=int)
parser.add_argument("--layer", default=40, type=int)
parser.add_argument("--groups", default=10, type=int, help="number of clusters")
parser.add_argument("--eta", default=0.24, type=float)
parser.add_argument("--kmeans_iters", default=25, type=int)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

DATASET = "azhx/counterfact"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
random.seed(args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.model)
# Some GPT-2 variants need this to avoid padding issues in generate
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
model.eval()

ds = load_dataset(DATASET, split=f"test[:{args.sample}]")

def get_prompt_and_target(ex):
    rr = ex["requested_rewrite"]
    prompt_tmp = rr["prompt"]
    subject = rr["subject"]
    prompt = prompt_tmp.format(subject)
    target_new = rr["target_new"]["str"]
    if not target_new.startswith(" "):
        target_new = " " + target_new
    return prompt, target_new

def check_efficacy_greedy(model, tokenizer, prompt, target):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    tgt_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    out = model.generate(
        **enc,
        max_new_tokens=len(tgt_ids),
        do_sample=False, num_beams=1,
        pad_token_id=tokenizer.eos_token_id
    )[0]
    gen_ids = out[enc["input_ids"].shape[1]:][:len(tgt_ids)]
    return torch.equal(gen_ids, tgt_ids)

layer = None
down_proj = None
W0 = None

def collect_records():
    global layer, down_proj, W0
    recs = []
    layer = model.transformer.h[args.layer]
    down_proj = layer.mlp.c_proj
    if W0 is None:
        W0 = down_proj.weight.clone()

    cache = {}
    def hook_fn(mod, inp, out):
        cache["h_last"] = inp[0][:, -1, :].detach()

    handle = down_proj.register_forward_hook(hook_fn)

    for ex in tqdm(ds, desc="Collect h_last/targets"):
        prompt, target = get_prompt_and_target(ex)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        cache.clear()
        _ = model(**inputs, use_cache=False)
        h_last = cache["h_last"].cpu().squeeze(0)

        target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        target_id_last = target_ids[0, -1].item()
        target_emb = model.lm_head.weight[target_id_last].detach().cpu()

        recs.append({
            "prompt": prompt,
            "target": target,
            "h": h_last,
            "tgt_emb": target_emb
        })

    handle.remove()
    return recs

# --- Step 2: cosine k-means (PyTorch, CPU) ---
def kmeans_cosine(X, k, iters=25, seed=42):
    torch.manual_seed(seed)
    N, D = X.shape
    Xn = X / (X.norm(dim=1, keepdim=True) + 1e-12)
    perm = torch.randperm(N)[:k]
    C = Xn[perm].clone()

    empty_retries = 0
    for _ in range(iters):
        sim = Xn @ C.T
        labels = sim.argmax(dim=1)

        new_C = torch.zeros_like(C)
        for j in range(k):
            idx = (labels == j).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                ridx = torch.randint(0, N, (1,))
                new_C[j] = Xn[ridx]
                empty_retries += 1
            else:
                new_C[j] = Xn[idx].mean(dim=0)
        C = new_C / (new_C.norm(dim=1, keepdim=True) + 1e-12)

    return labels, C

def rank1_delta(h_cpu, tgt_emb_cpu, eta):
    h = h_cpu.unsqueeze(1)
    t = tgt_emb_cpu.unsqueeze(0)
    return eta * (h @ t)

def run_group_pipeline(recs, n_groups, eta, kmeans_iters):
    global down_proj, W0
    H = torch.stack([r["tgt_emb"] for r in recs], dim=0)
    labels, _ = kmeans_cosine(H, k=n_groups, iters=kmeans_iters, seed=args.seed)

    groups = [[] for _ in range(n_groups)]
    for i, lab in enumerate(labels.tolist()):
        groups[lab].append(i)

    per_group = []
    total_success = 0
    total_count = 0

    layer = model.transformer.h[args.layer]
    down_proj = layer.mlp.c_proj

    for gi, idxs in enumerate(groups):
        if len(idxs) == 0:
            per_group.append({"group": gi, "size": 0, "success": 0, "eff": float("nan")})
            continue

        with torch.no_grad():
            down_proj.weight.copy_(W0)

        delta_sum_cpu = None
        for i in idxs:
            d = rank1_delta(recs[i]["h"], recs[i]["tgt_emb"], eta)
            if delta_sum_cpu is None:
                delta_sum_cpu = d
            else:
                delta_sum_cpu += d
        with torch.no_grad():
            delta_dev = delta_sum_cpu.to(down_proj.weight.dtype).to(down_proj.weight.device)
            down_proj.weight.add_(delta_dev)

        success = 0
        for i in idxs:
            ok = check_efficacy_greedy(model, tokenizer, recs[i]["prompt"], recs[i]["target"])
            success += int(ok)

        eff = 100.0 * success / len(idxs)
        per_group.append({"group": gi, "size": len(idxs), "success": success, "eff": eff})
        total_success += success
        total_count += len(idxs)

    with torch.no_grad():
        down_proj.weight.copy_(W0)

    overall_eff = 100.0 * total_success / max(1, total_count)
    return groups, per_group, overall_eff

def main():
    recs = collect_records()
    groups, per_group, overall_eff = run_group_pipeline(
        recs, n_groups=args.groups, eta=args.eta, kmeans_iters=args.kmeans_iters
    )

    print("\n=== Per-group results (independent edits per group) ===")
    for r in per_group:
        if r["size"] == 0:
            print(f"Group {r['group']:>3}: size=0 (skipped)")
        else:
            print(f"Group {r['group']:>3}: size={r['size']}, success={r['success']}, efficacy={r['eff']:.2f}%")
    print(f"\nOverall (weighted): {overall_eff:.2f}%  (total success {sum(r['success'] for r in per_group)}/{sum(r['size'] for r in per_group)})")

if __name__ == "__main__":
    main()
