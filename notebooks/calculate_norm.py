import torch
from transformers import AutoModelForCausalLM

model_path = "/home/michael/project/Preference-tuning-and-evaluation/outputs/01_30_dpo_ablation_all_datasets/dpo_relabeled_alpacafarm_pythiasft_20K_preference_data_minlength_beta0.05_lr1e-7_bs16_gradacc8/01_30_dpo_ablation_all_datasets_dpo_relabeled_alpacafarm_pythiasft_20K_preference_data_minlength_beta0.05_lr1e-7_bs16_gradacc8_num_batches_0"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)

raw_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

total1, total2=0.0, 0.0

for p in model.parameters():
    p_norm = p.data.norm(2)
    total1 += p_norm.item()**2
total1 = total1 **0.5
print(total1)

for p in raw_model.parameters():
    p_norm = p.data.norm(2)
    total2 += p_norm.item()**2
total2 = total2 **0.5
print(total2)

d_total = 0.0
for p1, p2 in zip(model.parameters(), raw_model.parameters()):
    diff = p1.data-p2.data
    d_total += diff.norm(2).item()**2
d_total = d_total ** 0.5
print(d_total)