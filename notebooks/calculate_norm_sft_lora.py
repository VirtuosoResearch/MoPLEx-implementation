import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_NAME = "Qwen/Qwen3-4B-Instruct-2507"
CKPT_DIR = "/home/michael/project/Preference-tuning-and-evaluation/outputs/dpo/Qwen3-4B-Instruct-2507-alpaca_farm-11829-SFT_lora"

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

def build_base(tokenizer_len: int = None):
    base = AutoModelForCausalLM.from_pretrained(
        BASE_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    if tokenizer_len is not None:
        base.resize_token_embeddings(tokenizer_len)
    base.to("cpu").eval()
    return base

tok_src = CKPT_DIR
try:
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(BASE_NAME, trust_remote_code=True)
tokenizer_len = len(tokenizer)

raw_model = build_base(tokenizer_len)
raw_norm = l2_params(raw_model)
print("||base|| =", raw_norm)

for i in range(275, 501, 25):
    ckpt = os.path.join(CKPT_DIR, f"checkpoint-{i}")
    if not os.path.isdir(ckpt):
        print(f"[skip] {ckpt} (not found)")
        continue

    base = build_base(tokenizer_len) 
    peft = PeftModel.from_pretrained(base, ckpt)
    merged = peft.merge_and_unload()
    merged.to("cpu").eval()

    merged_norm = l2_params(merged)
    delta_norm  = l2_diff(merged, raw_model)

    print(f"[checkpoint-{i}] ||merged||={merged_norm:.6e}  ||delta||={delta_norm:.6e}  ratio={delta_norm/merged_norm:.6e}")
