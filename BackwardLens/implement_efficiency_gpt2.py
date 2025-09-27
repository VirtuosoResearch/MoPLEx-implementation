import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

MODEL = "gpt2-xl"
DATASET = "azhx/counterfact"
N_EXAMPLES = 50 
ETA = 0.24 
EDIT_LAYER = 47
TOPK = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
model.eval()

ds = load_dataset(DATASET, split=f"train[:{N_EXAMPLES}]")

def get_prompt_and_target(ex):
    rr = ex["requested_rewrite"]
    prompt_tmp = rr["prompt"]
    subject = rr["subject"]
    prompt = prompt_tmp.format(subject)
    target_new = rr["target_new"]["str"]
    if not target_new.startswith(" "):
        target_new = " " + target_new
    return prompt, target_new

def forward_pass_shift_edit(model, tokenizer, prompt, target, layer_idx=EDIT_LAYER, eta=ETA):
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

def check_efficacy(model, tokenizer, prompt, target, topk=TOPK):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        probs = logits.softmax(-1)
        topk_ids = probs.topk(topk).indices[0].tolist()
        decoded = [tokenizer.decode(i) for i in topk_ids]
    # print(decoded)
    return target.strip() in [d.strip() for d in decoded]

success = 0
layer = model.transformer.h[EDIT_LAYER]
down_proj = layer.mlp.c_proj
W0 = down_proj.weight.clone()

for ex in tqdm(ds, desc="Editing"):
    prompt, target_new = get_prompt_and_target(ex)

    delta_W = forward_pass_shift_edit(model, tokenizer, prompt, target_new,
                                      layer_idx=EDIT_LAYER, eta=ETA)

    ok = check_efficacy(model, tokenizer, prompt, target_new)
    success += int(ok)
    # print(ok, int(ok))

    with torch.no_grad():
        down_proj.weight.copy_(W0)

eff = success / N_EXAMPLES * 100
print(f"\nEfficacy: {eff:.2f}%  (success {success}/{N_EXAMPLES})")
