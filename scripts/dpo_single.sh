cache_dir="cache"
wandb_project="dpo"
model_name="Qwen/Qwen3-4B"
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
export CUDA_VISIBLE_DEVICES=0,1

which_exp=${1:--1}
dryrun=false
debug=false
lr=1e-7
beta=0.05
gradient_accumulation_steps=2
batch_size=2
mini_batch_size=1
downsample_ratio=0.01
epoch=500

ipo_loss=false

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi

# 'Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_minlength'
preference_dataset_path="ZHZisZZ/imdb_preference"

run_name="dpo_${model_name}_bs${batch_size}"
echo "Running experiment $run_name"

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
    --num_train_epochs $epoch \
"

if [[ $ipo_loss = true ]]; then
    command+="--ipo_loss "
fi

echo -e "$command\n"
if [ $dryrun = false ]; then
    eval $command
    sleep 20
fi

