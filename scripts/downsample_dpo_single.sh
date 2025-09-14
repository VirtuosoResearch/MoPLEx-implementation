cache_dir="cache"
wandb_project="dpo"
model_name="meta-llama/Llama-3.2-1B"
output_dir='outputs'

export WANDB_PROJECT=$wandb_project
export WANDB_API_KEY="b3ea34bec4058d216f518671f078ed74c5b1dda3"
export WANDB_USERNAME="2462970640"
export WANDB_USER_EMAIL="2462970640@qq.com"
export HF_DATASETS_CACHE=$cache_dir

wandb_project="all_data"
which_exp=${1:--1}
dryrun=false
debug=false
lr=1e-7
beta=0.05
gradient_accumulation_steps=4
batch_size=4
mini_batch_size=1
downsample_ratio=0.01
epoch=20
approx_dpo=false
seed=3

data_min='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_minlength'
data_max='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_maxlength'
data_mode='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data_modelength'
skew_data_merge='Asap7772/alpaca_skewexp_minlength_merged'
alpaca_farm='Asap7772/relabeled_alpacafarm_pythiasft_20K_preference_data'
ipo_loss=false

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi

preference_dataset_path=$data_max

dataset_basename=$(basename -- $preference_dataset_path)

if [[ $preference_dataset_path = $data_max ]]; then
    dataset_basename="max"
fi

if [[ $preference_dataset_path = $data_min ]]; then
    dataset_basename="min"
fi

if [[ $preference_dataset_path = $data_mode ]]; then
    dataset_basename="mode"
fi

run_name="${dataset_basename}_beta${beta}_lr${lr}_bs${batch_size}_ga${gradient_accumulation_steps}_sd${seed}"
echo "Running experiment $run_name"

command="python -m trainers.dpo_sample_subset \
    --wandb_project $wandb_project \
    --run_name $run_name \
    --tokenizer_type $model_name \
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
    --seed $seed \
"

if [[ $ipo_loss = true ]]; then
    command+="--ipo_loss "
fi

echo -e "$command\n"
if [ $dryrun = false ]; then
    eval $command
    sleep 20
fi