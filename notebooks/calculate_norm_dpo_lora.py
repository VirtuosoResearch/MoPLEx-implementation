import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "Qwen/Qwen3-0.6B"
ckpt_root = "/home/michael/project/Preference-tuning-and-evaluation/outputs/dpo/dpo_Qwen/Qwen3-0.6B_bs4_lora"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

def build_base():
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="cache",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    base.resize_token_embeddings(len(tokenizer))
    base.to("cpu").eval()
    return base

def l2_params(model: torch.nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        t = p.detach().to(dtype=torch.float32, device="cpu")
        s += float(t.norm(2).item() ** 2)
    return s ** 0.5

def l2_diff(a: torch.nn.Module, b: torch.nn.Module) -> float:
    s = 0.0
    sda = {k: v.detach().to(dtype=torch.float32, device="cpu") for k, v in a.state_dict().items()}
    sdb = {k: v.detach().to(dtype=torch.float32, device="cpu") for k, v in b.state_dict().items()}
    for k in sda.keys():
        d = (sda[k] - sdb[k]).norm(2)
        s += float(d.item() ** 2)
    return s ** 0.5

raw_policy = build_base()
raw_norm = l2_params(raw_policy)
print("raw_policy (||θ_base||):", raw_norm)

for i in range(0, 9):
    ckpt = os.path.join(ckpt_root, f"Qwen3-0.6B_epoch_{i}")
    if not os.path.isdir(ckpt):
        print(f"[skip] {ckpt}")
        continue

    policy = build_base()
    peft = PeftModel.from_pretrained(policy, ckpt)

    merged_policy = peft.merge_and_unload()   # AutoModelForCausalLM
    merged_policy.to("cpu").eval()

    merged_norm = l2_params(merged_policy)
    delta_norm  = l2_diff(merged_policy, raw_policy)

    print(f"[epoch {i}] ||merged||={merged_norm:.6e}  ||delta||={delta_norm:.6e}  ratio={delta_norm/merged_norm:.6e}")
