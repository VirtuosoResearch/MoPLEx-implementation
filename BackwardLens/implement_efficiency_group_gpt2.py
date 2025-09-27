import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gpt2-xl", type=str)
parser.add_argument("--sample", default=200, type=int)
parser.add_argument("--layer", default=40, type=int)
args = parser.parse_args()

DATASET = "azhx/counterfact"
ETA = 0.24 
TOPK = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(args.model)
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
    gen_ids = out[enc["input_ids"].shape[1]:]
    gen_ids = gen_ids[:len(tgt_ids)]
    return torch.equal(gen_ids, tgt_ids)

def forward_pass_shift_edit(model, tokenizer, prompt, target, layer_idx=args.layer, eta=ETA):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    layer = model.transformer.h[layer_idx]
    down_proj = layer.mlp.c_proj
    cache = {}
    def hook_fn(mod, inp, out):
        cache["h_last"] = inp[0][:, -1, :].detach()
    handle = down_proj.register_forward_hook(hook_fn)
    _ = model(**inputs, use_cache=False)
    handle.remove()

    h_last = cache["h_last"]

    target_id = tokenizer(target, return_tensors="pt").input_ids[0, -1].item()
    target_emb = model.lm_head.weight[target_id].detach()

    with torch.no_grad():
        delta_W = eta * (h_last.squeeze(0)[:, None] @ target_emb[None, :])
        down_proj.weight += delta_W.to(down_proj.weight.dtype).to(down_proj.weight.device)

    return delta_W

success = 0
layer = model.transformer.h[args.layer]
down_proj = layer.mlp.c_proj
W0 = down_proj.weight.clone()

for dp in tqdm(ds, desc="Edit"):
    prompt, target_new = get_prompt_and_target(dp)

    delta_W = forward_pass_shift_edit(model, tokenizer, prompt, target_new,
                                      layer_idx=args.layer, eta=ETA)

for dp in tqdm(ds, desc="Test"):
    prompt, target_new = get_prompt_and_target(dp)
    ok = check_efficacy_greedy(model, tokenizer, prompt, target_new)
    # print(ok)
    success += int(ok)


eff = success / args.sample * 100
print(f"\nEfficacy: {eff:.2f}%  (success {success}/{args.sample})")
