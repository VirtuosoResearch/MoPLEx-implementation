import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trainers.network_utils import AutoModelForCausalLMWithValueHead


# "EleutherAI/pythia-1.4b"
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
eos = tokenizer.eos_token
policy = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="cache", 
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    device_map='auto',
)
policy.resize_token_embeddings(len(tokenizer))
raw_model = AutoModelForCausalLMWithValueHead(policy)

total2 = 0.0
for p in raw_model.parameters():
    p_norm = p.data.norm(2)
    total2 += p_norm.item()**2
total2 = total2 **0.5
print("raw_model : ", total2)

for i in range(0,15):
    model_path = f"/home/michael/project/Preference-tuning-and-evaluation/outputs/01_30_dpo_ablation_all_datasets/dpo_relabeled_alpacafarm_pythiasft_20K_preference_data_minlength_beta0.05_lr1e-7_bs4_gradacc4/01_30_dpo_ablation_all_datasets_dpo_relabeled_alpacafarm_pythiasft_20K_preference_data_minlength_beta0.05_lr1e-7_bs4_gradacc4_epoch_{i}"

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