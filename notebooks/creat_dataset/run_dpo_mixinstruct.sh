#!/bin/bash

# Script to run DPO training on mix-instruct synthetic datasets
# This script trains DPO models on different criteria (single and mixed)

cache_dir="cache"
wandb_project="dpo_mixinstruct"
model_name="Qwen/Qwen3-0.6B"
output_dir='outputs'

export WANDB_PROJECT=$wandb_project
export WANDB_API_KEY="b3ea34bec4058d216f518671f078ed74c5b1dda3"
export WANDB_USERNAME="2462970640"
export WANDB_USER_EMAIL="2462970640@qq.com"
export WANDB_DIR="./cache"
export WANDB_DATA_DIR="./cache"
export WANDB_CACHE_DIR="./cache"
export WANDB_TEMP="./cache/tmp"
export HF_DATASETS_CACHE=$cache_dir
export CUDA_VISIBLE_DEVICES=0

which_exp=${1:--1}
dryrun=false
debug=false
lr=1e-7
beta=0.05
gradient_accumulation_steps=2
batch_size=4
mini_batch_size=1
downsample_ratio=0.25
num_train_epochs=3

ipo_loss=false

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi

# Dataset configuration
# Options:
# - mixinstruct_synthetic_datasets/single_rougeL
# - mixinstruct_synthetic_datasets/single_bleu
# - mixinstruct_synthetic_datasets/single_bertscore
# - mixinstruct_synthetic_datasets/single_bleurt
# - mixinstruct_synthetic_datasets/single_bartscore
# - mixinstruct_synthetic_datasets/mixed_criteria
# Or use ds01 version: mixinstruct_synthetic_datasets_ds01/...

dataset_type=${2:-"mixed_criteria"}  # Default to mixed_criteria
ds_suffix=${3:-""}  # Optional suffix like "ds01"

if [[ -n "$ds_suffix" ]]; then
    preference_dataset_path="creat_dataset/mixinstruct_synthetic_datasets_${ds_suffix}/${dataset_type}"
else
    preference_dataset_path="creat_dataset/mixinstruct_synthetic_datasets/${dataset_type}"
fi

run_name="dpo_mixinstruct_${dataset_type}_${model_name}_bs${batch_size}_beta${beta}"
if [[ -n "$ds_suffix" ]]; then
    run_name="${run_name}_${ds_suffix}"
fi

echo "Running experiment $run_name"
echo "Using dataset: $preference_dataset_path"

command="python -m trainers.dpo \
    --wandb_project $wandb_project --run_name $run_name \
    --inner_iteration_steps 1 \
    --batch_size $batch_size \
    --mini_batch_size $mini_batch_size \
    --pretrained_dir $model_name \
    --preference_dataset_path $preference_dataset_path \
    --temperature $beta \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --cache_dir $cache_dir \
    --learning_rate $lr \
    --output_dir $output_dir \
    --downsample_ratio $downsample_ratio \
    --num_train_epochs $num_train_epochs \
    --use_lora True \
    --seed 42 \
"

if [[ $ipo_loss = true ]]; then
    command+=" --ipo_loss"
fi

echo -e "$command\n"
if [ $dryrun = false ]; then
    eval $command
    sleep 20
fi
