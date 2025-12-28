cache_dir="cache"
wandb_project="em_dpo"
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
downsample_ratio=0.1
num_train_epochs=100

ipo_loss=false

# EM-DPO specific parameters
num_clusters=2
reward_model_name="distilbert-base-uncased"
em_max_iterations=20
reward_learning_rate=1e-5
reward_num_epochs=3

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi

# Use mixed_criteria dataset (multi-preference)
preference_dataset_path="notebooks/ultrafeedback_synthetic/mixed_criteria"

run_name="em_dpo_ultrafeedback_mixed_criteria_${model_name}_bs${batch_size}_clusters${num_clusters}_beta${beta}"
echo "Running experiment $run_name"
echo "Using dataset: $preference_dataset_path"

command="python -m trainers.em_dpo \
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
    --num_clusters $num_clusters \
    --reward_model_name $reward_model_name \
    --em_max_iterations $em_max_iterations \
    --reward_learning_rate $reward_learning_rate \
    --reward_num_epochs $reward_num_epochs \
    --seed 42
"

if [[ $ipo_loss = true ]]; then
    command+=" --ipo_loss"
fi

echo -e "$command\n"
if [ $dryrun = false ]; then
    eval $command
    sleep 20
fi

