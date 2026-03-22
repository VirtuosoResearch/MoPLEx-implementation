cache_dir="cache"
wandb_project="rlhf_synthetic"
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
gradient_accumulation_steps=2
batch_size=4
mini_batch_size=1
downsample_ratio=0.25
num_train_epochs=60

# RLHF specific parameters
clip_range=0.2
gamma=1.0
gae_lambda=0.95
vf_coef=0.1
inner_iteration_steps=4
max_length=512
max_new_tokens=256

# KL control parameters
adap_kl_ctrl=true
init_kl_coef=0.2
target_kl=6.0
kl_horizon=10000

# Reward model path (required for RLHF)
# Note: RLHF requires a reward model that outputs a single scalar reward
# You need to either:
#   1. Train a reward model first using preference data (chosen/rejected pairs)
#   2. Use a pre-trained reward model compatible with AutoModelForSequenceClassification
# For mix-instruct, you could potentially use a general reward model, but it's recommended
# to train a proper reward model on the preference data first
reward_model_path="lvwerra/distilbert-imdb"  # Replace with your trained reward model path

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi

# Use mixed_criteria dataset (multi-preference)
preference_dataset_path="notebooks/mixinstruct_synthetic_datasets_ds01/mixed_criteria"

run_name="rlhf_mixinstruct_mixed_criteria_${model_name}_bs${batch_size}_lr${lr}"
echo "Running experiment $run_name"
echo "Using dataset: $preference_dataset_path"
echo "Using reward model: $reward_model_path"

command="python -m trainers.rlhf \
    --wandb_project $wandb_project --run_name $run_name \
    --pretrained_dir $model_name \
    --reward_model_path $reward_model_path \
    --preference_dataset_path $preference_dataset_path \
    --batch_size $batch_size \
    --mini_batch_size $mini_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --cache_dir $cache_dir \
    --learning_rate $lr \
    --output_dir $output_dir \
    --downsample_ratio $downsample_ratio \
    --num_train_epochs $num_train_epochs \
    --inner_iteration_steps $inner_iteration_steps \
    --clip_range $clip_range \
    --gamma $gamma \
    --gae_lambda $gae_lambda \
    --vf_coef $vf_coef \
    --max_length $max_length \
    --max_new_tokens $max_new_tokens \
    --init_kl_coef $init_kl_coef \
    --target_kl $target_kl \
    --kl_horizon $kl_horizon \
    --use_lora \
    --seed 42
"

if [[ $adap_kl_ctrl = true ]]; then
    command+=" --adap_kl_ctrl"
fi

echo -e "$command\n"
if [ $dryrun = false ]; then
    eval $command
    sleep 20
fi
