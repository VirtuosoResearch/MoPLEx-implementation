#!/bin/bash

# Script to run the DPO criteria experiment
# This demonstrates that DPO can fit single-criteria preferences but struggles with mixed criteria

# set -e

# echo "=========================================="
# echo "DPO Criteria Experiment"
# echo "=========================================="

# # # Step 1: Create synthetic datasets
# # echo ""
# # echo "Step 1: Creating synthetic datasets..."
# # echo "----------------------------------------"
# # python create_synthetic_datasets.py \
# #     --data_dir data_out \
# #     --output_dir synthetic_datasets \
# #     --seed 42

# # Step 2: Train and evaluate
# echo ""
# echo "Step 2: Training DPO models and evaluating..."
# echo "----------------------------------------"
# python train_dpo_criteria_experiment.py \
#     --datasets_dir synthetic_datasets \
#     --model_name Qwen/Qwen3-0.6B \
#     --output_dir dpo_criteria_results \
#     --num_epochs 10 \
#     --batch_size 8 \
#     --learning_rate 1e-7 \
#     --device cuda \
#     --seed 42 \
#     --downsample_ratio 0.25



python train_dpo_criteria_experiment.py \
    --datasets_dir synthetic_datasets \
    --model_name Qwen/Qwen3-0.6B \
    --output_dir dpo_criteria_results \
    --num_epochs 50 \
    --batch_size 8 \
    --learning_rate 1e-7 \
    --device cuda \
    --seed 42 \
    --downsample_ratio 0.25


# echo ""
# echo "=========================================="
# echo "Experiment completed!"
# echo "Results saved to: dpo_criteria_results/results.json"
# echo "=========================================="

