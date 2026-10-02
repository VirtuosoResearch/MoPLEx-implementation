#!/usr/bin/env bash
set -euo pipefail

# Example script: generate synthetic text ranking data for mixture-of-PL experiments.

# Optional: activate your environment first, for example:
# conda activate alignment
k=4
for m in 2 3 4 5 6 7 8 9 10
do
args=(
  # Output directory where train.jsonl, val.jsonl, and metadata.json are written.
  --output_dir mixture_of_pls/synth_data_n100k_m${m}_k${k}
  # Number of training rankings (annotators) to generate.
  --n_train 100000
  # Number of validation rankings (annotators) to generate.
  --n_val 1000
  # Number of items in each slate (ranking length).
  --m_items $m
  # Number of latent preference clusters (and feature dimensions).
  --k_clusters $k
  # Minimum integer value for each feature entry.
  --value_low 1
  # Maximum integer value for each feature entry.
  --value_high 20
  # Mixture weights alpha for latent clusters (must sum to positive value; normalized internally).
  --alpha "0.25,0.25,0.25,0.25"
  # Random seed for reproducibility.
  --seed 42
  # Prompt text shared by all examples.
  --prompt "Rank the following items based on the hidden preference."
)

# /home/ldy/miniconda3/envs/alignment/bin/python 
python generate_text_mixture_dataset.py "${args[@]}"
done
# Notes:
# - Each response is text with integer features like: Item i Features: [ 4 , 9 , 2 ]
# - For a sampled cluster c, utility is the c-th feature, then Gumbel noise is added (PL generation).
