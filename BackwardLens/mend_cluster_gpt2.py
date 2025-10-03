import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import argparse
from sklearn.cluster import KMeans
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gpt2-xl", type=str)
parser.add_argument("--sample", default=500, type=int)
parser.add_argument("--layer", default=47, type=int)
parser.add_argument("--groups", default=10, type=int, help="number of clusters")
parser.add_argument("--eta", default=0.24, type=float)
parser.add_argument("--kmeans_iters", default=25, type=int)
parser.add_argument("--seed", default=42, type=int)

# --- MEND-specific args ---
parser.add_argument("--mend_hidden", default=512, type=int, help="hidden size of MendEditor")
parser.add_argument("--mend_ckpt", default="", type=str, help="path to a saved MendEditor state_dict (.pt). Optional.")
args = parser.parse_args()

DATASET = "azhx/counterfact"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
random.seed(args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
model.eval()

if args.sample == 0:
    ds = load_dataset(DATASET, split="test")
else:
    ds = load_dataset(DATASET, split=f"test[:{args.sample}]")

def get_prompt_and_target(ex):
    rr = ex["requested_rewrite"]
    prompt_tmp = rr["prompt"]
    subject = rr["subject"]
    prompt = prompt_tmp.format(subject)
    target_new = rr["target_new"]["str"]
    if not target_new.startswith(" "):
        target_new = " " + target_new
    neighbor = ex["paraphrase_prompts"][0]
    return prompt, target_new, neighbor

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

# -------------------------
# MEND: small editor network (handles u_dim != delta_dim)
# -------------------------
class MendEditor(torch.nn.Module):
    """
    Minimal MEND editor: 输入 concat([u, delta])，输出残差并加回，得到 (ũ, δ̃)。
    显式接受 u 的维度(d_u=4d)和 delta 的维度(d_delta=d)。
    """
    def __init__(self, d_u: int, d_delta: int, hidden: int = 512):
        super().__init__()
        self.d_u = d_u
        self.d_delta = d_delta
        in_dim = d_u + d_delta
        # 取一个合理的隐层，不要过大
        h = min(hidden, max(128, min(d_u, d_delta)))
        self.fc1 = torch.nn.Linear(in_dim, h)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(h, in_dim)

        # 近似恒等初始化：小权重、零偏置
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=1e-4)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-4)

    @torch.no_grad()
    def forward(self, u, delta):
        # u:[d_u], delta:[d_delta]
        x = torch.cat([u, delta], dim=0)    # [d_u + d_delta]
        r = self.fc2(self.act(self.fc1(x))) # [d_u + d_delta]
        y = x + r
        u_t = y[: self.d_u]
        d_t = y[self.d_u :]
        return u_t, d_t

layer = None
down_proj = None
W0 = None
mend = None   # MendEditor instance

def setup_layer_and_editor():
    global layer, down_proj, W0, mend
    layer = model.transformer.h[args.layer]
    down_proj = layer.mlp.c_proj
    if W0 is None:
        W0 = down_proj.weight.detach().clone()
    # infer dims from c_proj.weight: [in, out] = [4d, d]
    in_dim, out_dim = down_proj.weight.shape  # 预期: (6400, 1600) 对 GPT-2 XL
    if mend is None:
        mend_local = MendEditor(d_u=in_dim, d_delta=out_dim, hidden=args.mend_hidden).to(device)
        if args.mend_ckpt and os.path.isfile(args.mend_ckpt):
            sd = torch.load(args.mend_ckpt, map_location=device)
            mend_local.load_state_dict(sd, strict=False)
            print(f"[MEND] Loaded editor weights from: {args.mend_ckpt}")
        mend = mend_local
    return

def collect_records():
    recs = []
    setup_layer_and_editor()

    cache = {}
    def hook_fn(mod, inp, out):
        # inp[0]: [B, T, d_in]; 这里是进入 c_proj 的激活（4d）
        cache["h_last"] = inp[0][:, -1, :].detach()

    handle = down_proj.register_forward_hook(hook_fn)

    for ex in tqdm(ds, desc="Collect h_last/targets"):
        prompt, target, neighbor = get_prompt_and_target(ex)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        cache.clear()
        _ = model(**inputs, use_cache=False)
        h_last = cache["h_last"].cpu().squeeze(0).clone()

        inputs_nei = tokenizer(neighbor, return_tensors="pt").to(device)
        cache.clear()
        _ = model(**inputs_nei, use_cache=False)
        h_last_neighbor = cache["h_last"].cpu().squeeze(0).clone()

        target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        target_id_last = target_ids[0, -1].item()
        target_emb = model.lm_head.weight[target_id_last].detach().cpu()

        recs.append({
            "prompt": prompt,
            "neighbor": neighbor,
            "target": target,
            "h": h_last,                     # 保留用于调试/对齐
            "h_neighbor": h_last_neighbor,
            "tgt_emb": target_emb
        })

    handle.remove()
    return recs

# --- cosine k-means (same as before) ---
def kmeans_cosine(X, k, iters, seed: int = 42):
    print(X.shape)
    assert X.dim() == 2, "X must be [N, D]"
    X_cpu = X.detach().cpu()
    Xn = X_cpu / (X_cpu.norm(dim=1, keepdim=True) + 1e-12)
    X_np = Xn.numpy()
    km = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=10,
        max_iter=iters,
        algorithm="auto",
        verbose=0,
    )
    km.fit(X_np)
    labels_np = km.labels_
    labels = torch.from_numpy(labels_np).long()
    return labels

# -------------------------
# MEND: compute (u, delta) from one (prompt, target)
# -------------------------
def compute_u_and_delta(prompt: str, target: str):
    setup_layer_and_editor()

    f_cache = {}
    def f_hook(mod, inp, out):
        f_cache["u"] = inp[0][:, -1, :].detach()  # [1, in_dim]

    h_f = down_proj.register_forward_hook(f_hook)

    with torch.enable_grad():
        if not down_proj.weight.requires_grad:
            down_proj.weight.requires_grad_(True)

        model.zero_grad(set_to_none=True)

        enc = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**enc, use_cache=False)
        logits_last = outputs.logits[:, -1, :]  # [1, V]

        tgt_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        tgt_last = tgt_ids[0, -1].view(1).to(device)

        loss = torch.nn.functional.cross_entropy(logits_last, tgt_last)
        loss.backward()

    u = f_cache["u"].squeeze(0).to(device)  # [in_dim]

    with torch.no_grad():
        dW = down_proj.weight.grad            # [in_dim, out_dim]
        u_norm2 = (u.pow(2).sum() + 1e-12)
        delta = (u @ dW) / u_norm2            # [out_dim]

    h_f.remove()
    return u, delta

def mend_rank1_update(prompt: str, target: str, eta: float):
    setup_layer_and_editor()
    tgt_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()

    delta_W_sum = None
    prefix = ""
    for j, tok_id in enumerate(tgt_ids):
        enc = tokenizer(prompt + prefix, return_tensors="pt").to(device)

        f_cache = {}
        def f_hook(mod, inp, out):
            f_cache["u"] = inp[0][:, -1, :].detach()  # [1, in_dim]
        h_f = down_proj.register_forward_hook(f_hook)

        with torch.enable_grad():
            if not down_proj.weight.requires_grad:
                down_proj.weight.requires_grad_(True)
            model.zero_grad(set_to_none=True)
            outputs = model(**enc, use_cache=False)
            logits_last = outputs.logits[:, -1, :]           # [1, V]
            tgt_for_step = torch.tensor([tok_id], device=device)
            loss = torch.nn.functional.cross_entropy(logits_last, tgt_for_step)
            loss.backward()

        u = f_cache["u"].squeeze(0).to(device)               # [in_dim]
        with torch.no_grad():
            dW = down_proj.weight.grad                       # [in_dim, out_dim]
            u_norm2 = (u.pow(2).sum() + 1e-12)
            delta = (u @ dW) / u_norm2                       # [out_dim]

        h_f.remove()

        u_n = u / (u.norm(p=2) + 1e-12)
        d_n = delta / (delta.norm(p=2) + 1e-12)
        u_t, d_t = mend(u_n, d_n)
        u_t = u_t / (u_t.norm(p=2) + 1e-12)
        d_t = d_t / (d_t.norm(p=2) + 1e-12)

        dW_j = -eta * torch.ger(u_t, d_t)                    # [in, out]

        if delta_W_sum is None:
            delta_W_sum = dW_j.detach().cpu()
        else:
            delta_W_sum.add_(dW_j.detach().cpu())

        piece = tokenizer.decode([tok_id])
        prefix += piece

    return delta_W_sum if delta_W_sum is not None else torch.zeros_like(down_proj.weight.detach().cpu())


def run_group_pipeline(recs, n_groups, eta, kmeans_iters):
    global down_proj, W0
    setup_layer_and_editor()

    # group by target decoder embedding cosine K-Means (same as before)
    H = torch.stack([r["tgt_emb"] for r in recs], dim=0)
    labels = kmeans_cosine(H, k=n_groups, iters=kmeans_iters, seed=args.seed)

    groups = [[] for _ in range(n_groups)]
    for i, lab in enumerate(labels.tolist()):
        groups[lab].append(i)

    per_group = []
    total_success = 0
    total_count = 0
    total_par = 0

    for gi, idxs in enumerate(groups):
        if len(idxs) == 0:
            per_group.append({"group": gi, "size": 0, "success": 0, "eff": float("nan")})
            continue

        # reset weight before applying group delta
        with torch.no_grad():
            down_proj.weight.copy_(W0)

        delta_sum_cpu = None
        for i in idxs:
            dW_i = mend_rank1_update(recs[i]["prompt"], recs[i]["target"], eta)  # [in, out]
            if delta_sum_cpu is None:
                delta_sum_cpu = dW_i
            else:
                delta_sum_cpu.add_(dW_i)

        # apply accumulated group update: down_proj.weight += ΔW
        with torch.no_grad():
            delta_dev = delta_sum_cpu.to(down_proj.weight.dtype).to(down_proj.weight.device)
            down_proj.weight.add_(delta_dev)

        # evaluate on original prompts
        success = 0
        for i in idxs:
            ok = check_efficacy_greedy(model, tokenizer, recs[i]["prompt"], recs[i]["target"])
            success += int(ok)

        # evaluate on paraphrases
        pa = 0
        for i in idxs:
            ok = check_efficacy_greedy(model, tokenizer, recs[i]["neighbor"], recs[i]["target"])
            pa += int(ok)

        eff = 100.0 * success / len(idxs)
        par = 100.0 * pa / len(idxs)
        per_group.append({
            "group": gi, "size": len(idxs), "success": success, "eff": eff,
            "par_suc": pa, "paraphrase": par
        })
        total_success += success
        total_par += pa
        total_count += len(idxs)

    # restore original weights
    with torch.no_grad():
        down_proj.weight.copy_(W0)

    overall_eff = 100.0 * total_success / max(1, total_count)
    overall_par = 100.0 * total_par / max(1, total_count)
    return groups, per_group, overall_eff, overall_par

def main():
    recs = collect_records()
    groups, per_group, overall_eff, overall_par = run_group_pipeline(
        recs, n_groups=args.groups, eta=args.eta, kmeans_iters=args.kmeans_iters
    )

    print("\n=== Per-group results (independent edits per group) ===")
    for r in per_group:
        if r["size"] == 0:
            print(f"Group {r['group']:>3}: size=0 (skipped)")
        else:
            print(f"Group {r['group']:>3}: size={r['size']}, success={r['success']}, efficacy={r['eff']:.2f}%, paraphrase={r['paraphrase']:.2f}%")
    print(f"\nOverall efficacy (weighted): {overall_eff:.2f}%  (total success {sum(r['success'] for r in per_group)}/{sum(r['size'] for r in per_group)})")
    print(f"\nOverall paraphrase (weighted): {overall_par:.2f}%  (total success {sum(r['par_suc'] for r in per_group)}/{sum(r['size'] for r in per_group)})")

if __name__ == "__main__":
    main()


# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from datasets import load_dataset
# from tqdm import tqdm
# import argparse
# import math
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import random
# import os

# parser = argparse.ArgumentParser()
# parser.add_argument("--model", default="gpt2-xl", type=str)
# parser.add_argument("--sample", default=500, type=int)
# parser.add_argument("--layer", default=47, type=int)
# parser.add_argument("--groups", default=10, type=int, help="number of clusters")
# parser.add_argument("--eta", default=0.24, type=float)
# parser.add_argument("--kmeans_iters", default=25, type=int)
# parser.add_argument("--seed", default=42, type=int)

# # --- MEND-specific args ---
# parser.add_argument("--mend_hidden", default=512, type=int, help="hidden size of MendEditor")
# parser.add_argument("--mend_ckpt", default="", type=str, help="path to a saved MendEditor state_dict (.pt). Optional.")
# args = parser.parse_args()

# DATASET = "azhx/counterfact"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.manual_seed(args.seed)
# random.seed(args.seed)

# tokenizer = AutoTokenizer.from_pretrained(args.model)
# # Some GPT-2 variants need this to avoid padding issues in generate
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
# model.eval()

# if args.sample == 0:
#     ds = load_dataset(DATASET, split="test")
# else:
#     ds = load_dataset(DATASET, split=f"test[:{args.sample}]")

# def get_prompt_and_target(ex):
#     rr = ex["requested_rewrite"]
#     prompt_tmp = rr["prompt"]
#     subject = rr["subject"]
#     prompt = prompt_tmp.format(subject)
#     target_new = rr["target_new"]["str"]
#     if not target_new.startswith(" "):
#         target_new = " " + target_new
#     neighbor = ex["paraphrase_prompts"][0]
#     return prompt, target_new, neighbor

# def check_efficacy_greedy(model, tokenizer, prompt, target):
#     enc = tokenizer(prompt, return_tensors="pt").to(device)
#     tgt_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
#     out = model.generate(
#         **enc,
#         max_new_tokens=len(tgt_ids),
#         do_sample=False, num_beams=1,
#         pad_token_id=tokenizer.eos_token_id
#     )[0]
#     gen_ids = out[enc["input_ids"].shape[1]:][:len(tgt_ids)]
#     return torch.equal(gen_ids, tgt_ids)

# # -------------------------
# # MEND: small editor network
# # -------------------------
# class MendEditor(torch.nn.Module):
#     """
#     A minimal MEND-style editor: takes concat([u, delta]) and outputs a residual Δ on [u, delta].
#     Default init is near-identity (residual ~ 0), so it safely falls back to raw grad rank-1 if untrained.
#     """
#     def __init__(self, d_model: int, hidden: int = 512):
#         super().__init__()
#         h = min(hidden, max(128, d_model))  # keep it sane
#         self.in_dim = 2 * d_model
#         self.out_dim = 2 * d_model
#         self.fc1 = torch.nn.Linear(self.in_dim, h)
#         self.act = torch.nn.ReLU()
#         self.fc2 = torch.nn.Linear(h, self.out_dim)

#         # Identity-like init: small weights, zero biases -> residual ≈ 0 at start
#         torch.nn.init.zeros_(self.fc1.bias)
#         torch.nn.init.zeros_(self.fc2.bias)
#         torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=1e-4)
#         torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-4)

#     @torch.no_grad()
#     def forward(self, u, delta):
#         # u, delta: [D]
#         x = torch.cat([u, delta], dim=0)              # [2D]
#         r = self.fc2(self.act(self.fc1(x)))           # [2D]
#         y = x + r                                     # residual -> near-identity
#         u_t, d_t = torch.split(y, [u.numel(), delta.numel()], dim=0)
#         return u_t, d_t

# layer = None
# down_proj = None
# W0 = None
# mend = None   # MendEditor instance
# d_model = None

# def setup_layer_and_editor():
#     global layer, down_proj, W0, mend, d_model
#     layer = model.transformer.h[args.layer]
#     down_proj = layer.mlp.c_proj
#     if W0 is None:
#         W0 = down_proj.weight.detach().clone()
#     # infer d_model from c_proj weight: [d, d]
#     d_model = down_proj.weight.shape[0]
#     if mend is None:
#         mend_local = MendEditor(d_model=d_model, hidden=args.mend_hidden).to(device)
#         if args.mend_ckpt and os.path.isfile(args.mend_ckpt):
#             sd = torch.load(args.mend_ckpt, map_location=device)
#             mend_local.load_state_dict(sd, strict=False)
#             print(f"[MEND] Loaded editor weights from: {args.mend_ckpt}")
#         mend = mend_local
#     return

# def collect_records():
#     recs = []
#     setup_layer_and_editor()

#     cache = {}
#     def hook_fn(mod, inp, out):
#         # inp[0]: [B, T, d]; cache last-token activation entering c_proj
#         cache["h_last"] = inp[0][:, -1, :].detach()

#     handle = down_proj.register_forward_hook(hook_fn)

#     for ex in tqdm(ds, desc="Collect h_last/targets"):
#         prompt, target, neighbor = get_prompt_and_target(ex)

#         inputs = tokenizer(prompt, return_tensors="pt").to(device)
#         cache.clear()
#         _ = model(**inputs, use_cache=False)
#         h_last = cache["h_last"].cpu().squeeze(0).clone()

#         inputs_nei = tokenizer(neighbor, return_tensors="pt").to(device)
#         cache.clear()
#         _ = model(**inputs_nei, use_cache=False)
#         h_last_neighbor = cache["h_last"].cpu().squeeze(0).clone()

#         target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
#         target_id_last = target_ids[0, -1].item()
#         target_emb = model.lm_head.weight[target_id_last].detach().cpu()

#         recs.append({
#             "prompt": prompt,
#             "neighbor": neighbor,
#             "target": target,
#             "h": h_last,                     # not used by MEND directly, but kept for parity/debug
#             "h_neighbor": h_last_neighbor,   # idem
#             "tgt_emb": target_emb
#         })

#     handle.remove()
#     return recs

# # --- cosine k-means (same as before) ---
# def kmeans_cosine(X, k, iters, seed: int = 42):
#     print(X.shape)
#     assert X.dim() == 2, "X must be [N, D]"
#     X_cpu = X.detach().cpu()
#     Xn = X_cpu / (X_cpu.norm(dim=1, keepdim=True) + 1e-12)
#     X_np = Xn.numpy()
#     km = KMeans(
#         n_clusters=k,
#         random_state=seed,
#         n_init=10,
#         max_iter=iters,
#         algorithm="auto",
#         verbose=0,
#     )
#     km.fit(X_np)
#     labels_np = km.labels_
#     labels = torch.from_numpy(labels_np).long()
#     return labels

# # -------------------------
# # MEND: compute (u, delta) from one (prompt, target)
# # -------------------------
# @torch.no_grad()
# def _get_next_token_logits(inputs):
#     # forward-only helper
#     outputs = model(**inputs, use_cache=False)
#     logits_last = outputs.logits[:, -1, :]  # [B, V]
#     return logits_last

# def compute_u_and_delta(prompt: str, target: str):
#     """
#     返回:
#       u     = 进入 c_proj 的最后一个 token 的输入激活 [d]
#       delta = loss 对 c_proj 输出在最后一个 token 的梯度 [d]
#     实现细节:
#       - 用 forward hook 抓 u
#       - 直接从 c_proj.weight.grad 恢复 delta:  delta = (dW) @ (u / ||u||^2)
#     """
#     setup_layer_and_editor()

#     f_cache = {}

#     def f_hook(mod, inp, out):
#         # inp[0]: [B, T, d]，拿最后一个 token 的输入激活
#         f_cache["u"] = inp[0][:, -1, :].detach()  # [1, d]

#     h_f = down_proj.register_forward_hook(f_hook)

#     # 强制开启梯度，并确保 c_proj.weight 能产生梯度
#     with torch.enable_grad():
#         # 有些环境里参数可能被关了，保险起见打开
#         if not down_proj.weight.requires_grad:
#             down_proj.weight.requires_grad_(True)

#         model.zero_grad(set_to_none=True)

#         enc = tokenizer(prompt, return_tensors="pt").to(device)
#         outputs = model(**enc, use_cache=False)
#         logits_last = outputs.logits[:, -1, :]  # [1, V]

#         tgt_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
#         tgt_last = tgt_ids[0, -1].view(1).to(device)

#         loss = torch.nn.functional.cross_entropy(logits_last, tgt_last)
#         loss.backward()  # 这一步会写入 down_proj.weight.grad

#     # 取 u
#     u = f_cache["u"].squeeze(0).to(device)  # [d]

#     # 用权重梯度恢复 delta
#     with torch.no_grad():
#         dW = down_proj.weight.grad  # [d, d]
#         # 数值防护
#         u_norm2 = (u.pow(2).sum() + 1e-12)
#         delta = dW @ (u / u_norm2)  # [d]

#     # 清理
#     h_f.remove()
#     return u, delta


# # @torch.no_grad()
# def mend_rank1_update(prompt: str, target: str, eta: float):
#     """
#     One MEND edit step for c_proj using (prompt, target):
#       1) compute (u, delta) via single-sample fwd/bwd
#       2) (u~, δ~) = MendEditor([u, δ])
#       3) ΔW = eta * δ~ u~^T
#     Returns the CPU tensor ΔW to accumulate across a group.
#     """
#     setup_layer_and_editor()
#     u, delta = compute_u_and_delta(prompt, target)          # [d], [d] on device
#     # normalize inputs (common in MEND)
#     u_n = u / (u.norm(p=2) + 1e-12)
#     d_n = delta / (delta.norm(p=2) + 1e-12)
#     u_t, d_t = mend(u_n, d_n)                               # near-identity if untrained
#     # optional renorm to keep scale reasonable
#     u_t = u_t / (u_t.norm(p=2) + 1e-12)
#     d_t = d_t / (d_t.norm(p=2) + 1e-12)

#     # outer product -> [d, d]
#     delta_W = eta * torch.ger(d_t, u_t)                     # same shape as c_proj.weight
#     return delta_W.detach().cpu()

# def run_group_pipeline(recs, n_groups, eta, kmeans_iters):
#     global down_proj, W0
#     setup_layer_and_editor()

#     # group by target decoder embedding cosine K-Means (same as before)
#     H = torch.stack([r["tgt_emb"] for r in recs], dim=0)
#     labels = kmeans_cosine(H, k=n_groups, iters=kmeans_iters, seed=args.seed)

#     groups = [[] for _ in range(n_groups)]
#     for i, lab in enumerate(labels.tolist()):
#         groups[lab].append(i)

#     per_group = []
#     total_success = 0
#     total_count = 0
#     total_par = 0

#     for gi, idxs in enumerate(groups):
#         if len(idxs) == 0:
#             per_group.append({"group": gi, "size": 0, "success": 0, "eff": float("nan")})
#             continue

#         # reset weight before applying group delta
#         with torch.no_grad():
#             down_proj.weight.copy_(W0)

#         delta_sum_cpu = None
#         # ----- MEND edits accumulated within a group -----
#         for i in idxs:
#             # Use the (prompt, target) of each item to compute a MEND-style ΔW_i
#             dW_i = mend_rank1_update(recs[i]["prompt"], recs[i]["target"], eta)
#             if delta_sum_cpu is None:
#                 delta_sum_cpu = dW_i
#             else:
#                 delta_sum_cpu.add_(dW_i)

#         # apply accumulated group update
#         with torch.no_grad():
#             delta_dev = delta_sum_cpu.to(down_proj.weight.dtype).to(down_proj.weight.device)
#             down_proj.weight.add_(delta_dev)

#         # evaluate on original prompts
#         success = 0
#         for i in idxs:
#             ok = check_efficacy_greedy(model, tokenizer, recs[i]["prompt"], recs[i]["target"])
#             success += int(ok)

#         # evaluate on paraphrases
#         pa = 0
#         for i in idxs:
#             ok = check_efficacy_greedy(model, tokenizer, recs[i]["neighbor"], recs[i]["target"])
#             pa += int(ok)

#         eff = 100.0 * success / len(idxs)
#         par = 100.0 * pa / len(idxs)
#         per_group.append({
#             "group": gi, "size": len(idxs), "success": success, "eff": eff,
#             "par_suc": pa, "paraphrase": par
#         })
#         total_success += success
#         total_par += pa
#         total_count += len(idxs)

#     # restore original weights
#     with torch.no_grad():
#         down_proj.weight.copy_(W0)

#     overall_eff = 100.0 * total_success / max(1, total_count)
#     overall_par = 100.0 * total_par / max(1, total_count)
#     return groups, per_group, overall_eff, overall_par

# def main():
#     recs = collect_records()
#     groups, per_group, overall_eff, overall_par = run_group_pipeline(
#         recs, n_groups=args.groups, eta=args.eta, kmeans_iters=args.kmeans_iters
#     )

#     print("\n=== Per-group results (independent edits per group) ===")
#     for r in per_group:
#         if r["size"] == 0:
#             print(f"Group {r['group']:>3}: size=0 (skipped)")
#         else:
#             print(f"Group {r['group']:>3}: size={r['size']}, success={r['success']}, efficacy={r['eff']:.2f}%, paraphrase={r['paraphrase']:.2f}%")
#     print(f"\nOverall efficacy (weighted): {overall_eff:.2f}%  (total success {sum(r['success'] for r in per_group)}/{sum(r['size'] for r in per_group)})")
#     print(f"\nOverall paraphrase (weighted): {overall_par:.2f}%  (total success {sum(r['par_suc'] for r in per_group)}/{sum(r['size'] for r in per_group)})")

# if __name__ == "__main__":
#     main()
