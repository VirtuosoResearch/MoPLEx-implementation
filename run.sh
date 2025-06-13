#!/usr/bin/env sh

python src/train.py \
    --epochs 10 \
    --batch_size 2 \
    --max_length 256 \
    --lr 1e-6 \
    --beta 0.1 \
    --seed 2003 \
    --model_name "gpt2" \
    --dataset_name "jondurbin/truthy-dpo-v0.1" \
    --wandb_project "truthy-dpo" \
    --dpo_type "approx" \
    --ref_cache_path "ref_cache.pt"
