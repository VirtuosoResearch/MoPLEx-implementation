cache_dir="cache"
wandb_project="rlchf_synthetic"
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
num_train_epochs=60

ipo_loss=false

# RLCHF-specific parameters
num_latent_groups=4
aggregation_type="weighted_avg"  # or "log_sum_exp"
aggregation_temperature=1.0
gating_type="prompt_dependent"  # or "global"
gating_hidden_size=128
entropy_reg_coef=0.1
use_group_embeddings=false
group_embedding_dim=64

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi

# Use mixed_criteria dataset (multi-preference)
preference_dataset_path="notebooks/mixinstruct_synthetic_datasets_ds01/mixed_criteria"

run_name="rlchf_mixinstruct_${model_name}_bs${batch_size}_beta${beta}_K${num_latent_groups}_${aggregation_type}"
echo "Running experiment $run_name"
echo "Using dataset: $preference_dataset_path"

command="python -m trainers.rlchf \
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
    --num_latent_groups $num_latent_groups \
    --aggregation_type $aggregation_type \
    --aggregation_temperature $aggregation_temperature \
    --gating_type $gating_type \
    --gating_hidden_size $gating_hidden_size \
    --entropy_reg_coef $entropy_reg_coef \
    --use_group_embeddings $use_group_embeddings \
    --group_embedding_dim $group_embedding_dim
"

if [[ $ipo_loss = true ]]; then
    command+=" --ipo_loss"
fi

echo -e "$command\n"
if [ $dryrun = false ]; then
    eval $command
    sleep 20
fi
