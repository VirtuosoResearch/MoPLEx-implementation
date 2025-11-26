# %%
from datasets import load_dataset

ds = load_dataset(
    "tatsu-lab/alpaca_farm",
    "alpaca_instructions",
    split="unlabeled",
    revision="refs/convert/parquet", 
)
print(ds)
# %%
from datasets import get_dataset_config_names, get_dataset_split_names

configs = get_dataset_config_names("tatsu-lab/alpaca_farm", revision="refs/convert/parquet")
print("configs:", configs)
for c in configs:
    splits = get_dataset_split_names("tatsu-lab/alpaca_farm", c, revision="refs/convert/parquet")
    print(c, "->", splits)
