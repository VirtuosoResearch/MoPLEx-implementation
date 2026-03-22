# TODO: Fill in the following variables
cache_dir="cache"
wandb_project="dpo"
output_dir='outputs'

export WANDB_PROJECT=$wandb_project
export WANDB_API_KEY="b3ea34bec4058d216f518671f078ed74c5b1dda3"
export WANDB_USERNAME="2462970640"
export WANDB_USER_EMAIL="2462970640@qq.com"
export HF_DATASETS_CACHE=$cache_dir
env_name=''

exp_num=0
which_exp=${1:--1}
dryrun=false
debug=false
use_tpu=false

wandb_dir=/tmp/wandb/$RANDOM
mkdir -p $wandb_dir
export WANDB_DIR=$wandb_dir

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi


dataset_path="tatsu-lab/alpaca_farm"

model="meta-llama/Llama-3.2-1B-Instruct"

mnsimple=$(basename -- "$model")
dsimple=$(basename -- "$dataset_path")
output_dir="$output_dir/$wandb_project/$mnsimple-$dsimple-$RANDOM-SFT"

echo "wandb_project: $wandb_project"
echo "model: $model"
echo "dataset_path: $dataset_path"
echo "output_dir: $output_dir"

command="python ./trainers/sft.py \
    --dataset_path \"$dataset_path\" \
    --pretrained_dir=\"$model\" \
    --output_dir=\"$output_dir\" \
    --num_train_epochs 20
"


echo -e "$command\n"
if [ $dryrun = false ]; then
    eval $command
    sleep 20
fi


cache_dir="cache"
wandb_project="dpo"
output_dir='outputs'

export WANDB_PROJECT=$wandb_project
export WANDB_API_KEY="b3ea34bec4058d216f518671f078ed74c5b1dda3"
export WANDB_USERNAME="2462970640"
export WANDB_USER_EMAIL="2462970640@qq.com"
export HF_DATASETS_CACHE=$cache_dir
env_name=''

exp_num=0
which_exp=${1:--1}
dryrun=false
debug=false
use_tpu=false

wandb_dir=/tmp/wandb/$RANDOM
mkdir -p $wandb_dir
export WANDB_DIR=$wandb_dir

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi


dataset_path="tatsu-lab/alpaca_farm"

model="Qwen/Qwen3-0.6B"

mnsimple=$(basename -- "$model")
dsimple=$(basename -- "$dataset_path")
output_dir="$output_dir/$wandb_project/$mnsimple-$dsimple-$RANDOM"

echo "wandb_project: $wandb_project"
echo "model: $model"
echo "dataset_path: $dataset_path"
echo "output_dir: $output_dir"

command="python ./trainers/sft.py \
    --dataset_path \"$dataset_path\" \
    --pretrained_dir=\"$model\" \
    --output_dir=\"$output_dir\" \
    --num_train_epochs 20
"


echo -e "$command\n"
if [ $dryrun = false ]; then
    eval $command
    sleep 20
fi


cache_dir="cache"
wandb_project="dpo"
output_dir='outputs'

export WANDB_PROJECT=$wandb_project
export WANDB_API_KEY="b3ea34bec4058d216f518671f078ed74c5b1dda3"
export WANDB_USERNAME="2462970640"
export WANDB_USER_EMAIL="2462970640@qq.com"
export HF_DATASETS_CACHE=$cache_dir
env_name=''

exp_num=0
which_exp=${1:--1}
dryrun=false
debug=false
use_tpu=false

wandb_dir=/tmp/wandb/$RANDOM
mkdir -p $wandb_dir
export WANDB_DIR=$wandb_dir

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi


dataset_path="tatsu-lab/alpaca_farm"

model="Qwen/Qwen3-4B-Instruct-2507"

mnsimple=$(basename -- "$model")
dsimple=$(basename -- "$dataset_path")
output_dir="$output_dir/$wandb_project/$mnsimple-$dsimple-$RANDOM"

echo "wandb_project: $wandb_project"
echo "model: $model"
echo "dataset_path: $dataset_path"
echo "output_dir: $output_dir"

command="python ./trainers/sft.py \
    --dataset_path \"$dataset_path\" \
    --pretrained_dir=\"$model\" \
    --output_dir=\"$output_dir\" \
    --num_train_epochs 20
"


echo -e "$command\n"
if [ $dryrun = false ]; then
    eval $command
    sleep 20
fi
