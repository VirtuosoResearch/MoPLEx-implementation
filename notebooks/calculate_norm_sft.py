import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trainers.network_utils import AutoModelForCausalLMWithValueHead


# "EleutherAI/pythia-1.4b"
raw_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

total2 = 0.0
for p in raw_model.parameters():
    p_norm = p.data.norm(2)
    total2 += p_norm.item()**2
total2 = total2 **0.5
print("raw_model : ", total2)

for i in range(225,501,25):
    model_path = f"/home/michael/project/Preference-tuning-and-evaluation/outputs/dpo/Qwen3-4B-Instruct-2507-alpaca_farm-23186/checkpoint-{i}"

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    total1= 0.0

    for p in model.parameters():
        p_norm = p.data.norm(2)
        total1 += p_norm.item()**2
    total1 = total1 **0.5
    print(f"model {i}: {total1}")


    model = model.to("cpu")
    raw_model = raw_model.to("cpu")
    d_total = 0.0
    for p1, p2 in zip(model.parameters(), raw_model.parameters()):
        diff = p1.data-p2.data
        d_total += diff.norm(2).item()**2
    d_total = d_total ** 0.5
    print("difference: ",d_total)