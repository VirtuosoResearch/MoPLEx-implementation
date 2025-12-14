#!/bin/bash

# Script to run approximation error comparison

cache_dir="cache"
preference_dataset_path='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_minlength'

# Models to compare (add more as needed)
models=(
    "meta-llama/Llama-3.2-1B"
    # "meta-llama/Llama-3.2-3B"
    # Add more models here
)

# Parameters
batch_size=4
mini_batch_size=1
temperature=0.05
num_samples=20
use_lora=true

echo "Running approximation error comparison..."
echo "Models: ${models[@]}"
echo "Dataset: $preference_dataset_path"
echo "Num samples: $num_samples"

for model in "${models[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing model: $model"
    echo "=========================================="
    
    python trainers/compare_approx_error.py \
        --model_names "$model" \
        --preference_dataset_path "$preference_dataset_path" \
        --batch_size $batch_size \
        --mini_batch_size $mini_batch_size \
        --temperature $temperature \
        --num_samples $num_samples \
        --cache_dir "$cache_dir" \
        $([ "$use_lora" = true ] && echo "--use_lora" || echo "")
    
    if [ $? -ne 0 ]; then
        echo "Error processing model: $model"
        continue
    fi
done

echo ""
echo "Comparison complete!"
