#!/bin/bash
# Complete workflow to compare trained DPO model vs estimated model for UltraFeedback helpfulness

cache_dir="cache"
model_name="Qwen/Qwen3-0.6B"
output_dir='outputs'
preference_dataset_path="notebooks/ultrafeedback_synthetic/single_helpfulness"
beta=0.05
downsample_ratio=0.05
projection_dim=200
batch_size=2

# echo "=========================================="
# echo "Workflow: Compare Trained vs Estimated Model"
# echo "Dataset: UltraFeedback helpfulness"
# echo "Model: $model_name"
# echo "=========================================="

# Step 1: Precompute gradients and b
# echo ""
# echo "Step 1: Precomputing gradients and b..."
# echo "=========================================="
# use_lora=true  # Should match training setting
# precompute_output_file="${model_name##*/}_${projection_dim}_${downsample_ratio}.pt"
# precompute_output_file=$(echo $precompute_output_file | tr '/' '-')

# if [ "$use_lora" = true ]; then
#     python -m trainers.precompute_gradients_for_approx_dpo \
#         --model_name "$model_name" \
#         --preference_dataset_path "$preference_dataset_path" \
#         --batch_size $batch_size \
#         --temperature $beta \
#         --cache_dir $cache_dir \
#         --projection_dim $projection_dim \
#         --downsample_ratio $downsample_ratio \
#         --seed 42 \
#         --use_lora
# else
#     python -m trainers.precompute_gradients_for_approx_dpo \
#         --model_name "$model_name" \
#         --preference_dataset_path "$preference_dataset_path" \
#         --batch_size $batch_size \
#         --temperature $beta \
#         --cache_dir $cache_dir \
#         --projection_dim $projection_dim \
#         --downsample_ratio $downsample_ratio \
#         --seed 42
# fi

# if [ $? -ne 0 ]; then
#     echo "Error in precomputation step!"
#     exit 1
# fi

# echo ""
# echo "Precomputation complete. Output: pre_compute/$precompute_output_file"

# # Step 2: Train DPO model
# echo ""
# echo "Step 2: Training DPO model..."
# echo "=========================================="
# run_name="dpo_ultrafeedback_helpfulness_${model_name##*/}_bs${batch_size}_beta${beta}"
# use_lora=true
# model_base_name="${model_name##*/}"  # Qwen3-0.6B

# WANDB_MODE=disabled python -m trainers.dpo \
#     --wandb_project "dpo_synthetic" \
#     --run_name "$run_name" \
#     --inner_iteration_steps 1 \
#     --batch_size $batch_size \
#     --mini_batch_size 1 \
#     --pretrained_dir "$model_name" \
#     --preference_dataset_path "$preference_dataset_path" \
#     --temperature $beta \
#     --gradient_accumulation_steps 2 \
#     --cache_dir $cache_dir \
#     --learning_rate 1e-7 \
#     --output_dir $output_dir \
#     --downsample_ratio $downsample_ratio \
#     --num_train_epochs 11 \
#     --use_lora $use_lora \
#     --seed 42

# if [ $? -ne 0 ]; then
#     echo "Error in DPO training step!"
#     exit 1
# fi

# # Construct checkpoint path
# # From dpo.py: output_dir = os.path.join(args.output_dir, args.wandb_project, args.run_name)
# # If use_lora: output_dir += "_lora"
# # Checkpoint saved as: model_name + f"_epoch_{epoch}" where epoch is divisible by 10
# wandb_project="dpo_synthetic"
# if [ "$use_lora" = true ]; then
#     run_name_with_lora="${run_name}_lora"
# else
#     run_name_with_lora="$run_name"
# fi
# base_checkpoint_dir="$output_dir/$wandb_project/$run_name_with_lora"
# # Checkpoints are saved every 10 epochs (epoch % 10 == 0), so for 10 epochs we use epoch_10
# trained_model_path="$base_checkpoint_dir/${model_base_name}_epoch_10"
# echo ""
# echo "DPO training complete. Expected checkpoint path: $trained_model_path"
# echo "Note: Training for 10 epochs, using epoch_10 checkpoint. If the checkpoint doesn't exist, please check the actual saved checkpoints in: $base_checkpoint_dir"

# # Step 3: Compare trained model vs estimated model
# run_name="dpo_ultrafeedback_helpfulness_${model_name##*/}_bs${batch_size}_beta${beta}"
# wandb_project="dpo_synthetic"
# use_lora=true
# model_base_name="${model_name##*/}"  # Qwen3-0.6B

# # Construct checkpoint path (epoch_0)
# if [ "$use_lora" = true ]; then
#     run_name_with_lora="${run_name}_lora"
# else
#     run_name_with_lora="$run_name"
# fi
# base_checkpoint_dir="$output_dir/$wandb_project/$run_name_with_lora"
# trained_model_path="$base_checkpoint_dir/${model_base_name}_epoch_10"

# # Precompute file path
# precompute_output_file="${model_name##*/}_${projection_dim}_${downsample_ratio}.pt"
# precompute_output_file=$(echo $precompute_output_file | tr '/' '-')

# echo ""
# echo "Configuration:"
# echo "  Trained model path: $trained_model_path"
# echo "  Precompute file: pre_compute/$precompute_output_file"
# echo "  Base model: $model_name"
# echo ""

# # Check if files exist
# if [ ! -d "$trained_model_path" ]; then
#     echo "ERROR: Trained model checkpoint not found at: $trained_model_path"
#     echo "Available checkpoints in $base_checkpoint_dir:"
#     ls -la "$base_checkpoint_dir" 2>/dev/null || echo "  Directory does not exist"
#     exit 1
# fi

# precompute_path="pre_compute/$precompute_output_file"
# if [ ! -f "$precompute_path" ]; then
#     echo "ERROR: Precomputed gradients file not found at: $precompute_path"
#     exit 1
# fi

# echo "Files found. Starting comparison..."
# echo ""

# Step 3: Compare trained model vs estimated model
run_name="dpo_ultrafeedback_helpfulness_${model_name##*/}_bs${batch_size}_beta${beta}"
wandb_project="dpo_synthetic"
use_lora=true
model_base_name="${model_name##*/}"  # Qwen3-0.6B

# Construct checkpoint path
if [ "$use_lora" = true ]; then
    run_name_with_lora="${run_name}_lora"
else
    run_name_with_lora="$run_name"
fi
base_checkpoint_dir="$output_dir/$wandb_project/$run_name_with_lora"
trained_model_path="$base_checkpoint_dir/${model_base_name}_epoch_10"

# Precompute file path
precompute_output_file="${model_name##*/}_${projection_dim}_${downsample_ratio}.pt"
precompute_output_file=$(echo $precompute_output_file | tr '/' '-')

echo ""
echo "Configuration:"
echo "  Trained model path: $trained_model_path"
echo "  Precompute file: pre_compute/$precompute_output_file"
echo "  Base model: $model_name"
echo ""

# Check if files exist
if [ ! -d "$trained_model_path" ]; then
    echo "ERROR: Trained model checkpoint not found at: $trained_model_path"
    echo "Available checkpoints in $base_checkpoint_dir:"
    ls -la "$base_checkpoint_dir" 2>/dev/null || echo "  Directory does not exist"
    exit 1
fi

precompute_path="pre_compute/$precompute_output_file"
if [ ! -f "$precompute_path" ]; then
    echo "ERROR: Precomputed gradients file not found at: $precompute_path"
    exit 1
fi

echo "Files found. Starting comparison..."
echo ""

# Step: Compare trained model vs estimated model
echo "=========================================="
echo "Comparing trained model vs estimated model..."
echo "=========================================="

python src/compare_loss_on_testset.py \
    --trained_model_path "$trained_model_path" \
    --precompute_file "$precompute_output_file" \
    --base_model_name "$model_name" \
    --preference_dataset_path "$preference_dataset_path" \
    --beta $beta \
    --cache_dir $cache_dir \
    --verbose

if [ $? -ne 0 ]; then
    echo "Error in comparison step!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "=========================================="
