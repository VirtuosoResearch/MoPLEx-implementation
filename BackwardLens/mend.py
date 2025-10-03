import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook

from experiments.py.demo import model_editing
# EleutherAI/gpt-j-6B
MODEL_NAME = "gpt2-xl"

model, tokenizer = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=False,
        torch_dtype=(torch.float16 if "20b" in MODEL_NAME else None),
    ).to("cuda"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
tokenizer.pad_token = tokenizer.eos_token
model.config

request = [{
        "prompt": "{} was the founder of",
        "subject": "Steve Jobs",
        "target_new": {"str": "Microsoft"},
    }
]

generation_prompts = [
    "Steve Jobs was the founder of",
]

ALG_NAME = "MEND"


# Execute rewrite
model_raw = AutoModelForCausalLM.from_pretrained("gpt2-xl").to("cuda")
model_new, orig_weights = model_editing(
    model, tokenizer, request, alg_name=ALG_NAME
)

enc = tokenizer(generation_prompts[0], return_tensors="pt").to(model.device)
out = model_raw.generate(
    **enc,
    max_new_tokens=2,
    do_sample=False, num_beams=1,
    pad_token_id=tokenizer.eos_token_id
)[0]

out_new = model_new.generate(
    **enc,
    max_new_tokens=2,
    do_sample=False, num_beams=1,
    pad_token_id=tokenizer.eos_token_id
)[0]

print("raw output: ", tokenizer.decode(out))
print("new output: ", tokenizer.decode(out_new))
