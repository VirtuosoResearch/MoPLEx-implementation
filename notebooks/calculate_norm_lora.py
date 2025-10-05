import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

def build_base_policy():
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="cache",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    base.resize_token_embeddings(len(tokenizer))
    base.to("cpu")
    base.eval()
    return base

base_policy = build_base_policy()

def l2_params(model: torch.nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        t = p.detach().to(dtype=torch.float32, device="cpu")
        s += float(t.norm(2).item() ** 2)
    return s ** 0.5

def l2_diff(model_a: torch.nn.Module, model_b: torch.nn.Module) -> float:
    s = 0.0
    sd_a = {k: v.detach().to(dtype=torch.float32, device="cpu") for k, v in model_a.state_dict().items()}
    sd_b = {k: v.detach().to(dtype=torch.float32, device="cpu") for k, v in model_b.state_dict().items()}
    for k in sd_a.keys():
        d = (sd_a[k] - sd_b[k]).norm(2)
        s += float(d.item() ** 2)
    return s ** 0.5

base_norm = l2_params(base_policy)
print("||base||:", base_norm)

for i in range(275, 501, 25):
    ckpt = f"/home/michael/project/Preference-tuning-and-evaluation/outputs/dpo/Qwen3-0.6B-alpaca_farm-31898-SFT_lora/checkpoint-{i}"
    if not os.path.isdir(ckpt):
        print(f"[skip] {ckpt} (not a dir)")
        continue

    policy = build_base_policy()

    peft_model = PeftModel.from_pretrained(policy, ckpt)
    peft_model.to("cpu")
    peft_model.eval()

    merged_policy = peft_model.merge_and_unload()  # -> AutoModelForCausalLM
    merged_policy.to("cpu")
    merged_policy.eval()

    merged_norm = l2_params(merged_policy)
    delta_norm  = l2_diff(merged_policy, base_policy)

    rel = delta_norm / merged_norm if merged_norm > 0 else float("nan")

    print(f"[epoch {i}]  ||merged|| = {merged_norm:.6e} | ||delta|| = {delta_norm:.6e} | ||delta||/||merged|| = {rel:.6e}")
