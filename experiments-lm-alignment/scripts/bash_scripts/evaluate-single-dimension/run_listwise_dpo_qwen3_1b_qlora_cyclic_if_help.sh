#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-VirtuosoResearch}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_NAME="${WANDB_NAME:-qwen3-0.6b-listwise-dpo-qlora-cyclic}"
export WANDB_MODE="${WANDB_MODE:-online}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3-1b-listwise-dpo-qlora-cyclic}"

DATASET_DIR="${REPO_ROOT}/data/cyclic_ultrafeedback_m2_instruction_following_helpfulness"
CONFIG_PATH="recipes/qwen3-1b/dpo/config_qlora_listwise_cyclic_instruction_following_helpfulness.yaml"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing cyclic dataset directory: ${DATASET_DIR}"
  echo "Generate it first, for example:"
  echo "  PYTHONPATH=src /home/ldy/miniconda3/envs/alignment/bin/python scripts/create_cyclic_ultrafeedback_dataset.py --dataset_name openbmb/UltraFeedback --source_split train --dimensions instruction_following helpfulness --output_dir data/cyclic_ultrafeedback_m2_instruction_following_helpfulness"
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

# ACCELERATE_LOG_LEVEL=info accelerate launch \
#   --config_file recipes/accelerate_configs/ddp.yaml \
#   --num_processes=1 \
#   scripts/dpo.py \
#   --config "${CONFIG_PATH}" \
#   --dataset_name "./data/cyclic_ultrafeedback_m2_instruction_following_helpfulness" \
#   --output_dir "${OUTPUT_DIR}_combined_instruction_following" \
#   --report_to wandb \
#   --run_name "${WANDB_NAME}_combined_instruction_following" \
#   --listwise_num_responses 2 \
#   --preference_dimensions instruction_following helpfulness \
#   --do_eval False --eval_strategy no \
#   --num_train_epochs 10 \
#   --per_device_train_batch_size 4

for dim in "instruction_following" "helpfulness"
do
if [[ "${dim}" == "helpfulness" ]]; then
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/ddp.yaml \
  --num_processes=1 \
  scripts/dpo.py \
  --config "${CONFIG_PATH}" \
  --dataset_name "./data/cyclic_ultrafeedback_m2_instruction_following_helpfulness" \
  --output_dir "${OUTPUT_DIR}_instruction_following_helpfulness_single_${dim}_m2" \
  --report_to wandb \
  --run_name "${WANDB_NAME}_instruction_following_helpfulness_single_${dim}_m2" \
  --listwise_num_responses 2 \
  --preference_dimensions ${dim} \
  --do_eval False --eval_strategy no \
  --num_train_epochs 10 \
  --per_device_train_batch_size 2
fi

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/ddp.yaml \
  --num_processes=1 \
  scripts/dpo.py \
  --config "${CONFIG_PATH}" \
  --dataset_name "./data/cyclic_ultrafeedback_m2_instruction_following_helpfulness" \
  --output_dir "${OUTPUT_DIR}_instruction_following_helpfulness_single_${dim}_m3" \
  --report_to wandb \
  --run_name "${WANDB_NAME}_instruction_following_helpfulness_single_${dim}_m3" \
  --listwise_num_responses 3 \
  --preference_dimensions ${dim} \
  --do_eval False --eval_strategy no \
  --num_train_epochs 15 \
  --per_device_train_batch_size 2

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/ddp.yaml \
  --num_processes=1 \
  scripts/dpo.py \
  --config "${CONFIG_PATH}" \
  --dataset_name "./data/cyclic_ultrafeedback_m2_instruction_following_helpfulness" \
  --output_dir "${OUTPUT_DIR}_instruction_following_helpfulness_single_${dim}_m4" \
  --report_to wandb \
  --run_name "${WANDB_NAME}_instruction_following_helpfulness_single_${dim}_m4" \
  --listwise_num_responses 4 \
  --preference_dimensions ${dim} \
  --do_eval False --eval_strategy no \
  --num_train_epochs 60 \
  --per_device_train_batch_size 2
done 


ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/ddp.yaml \
  --num_processes=1 \
  scripts/dpo.py \
  --config "${CONFIG_PATH}" \
  --dataset_name "./data/cyclic_ultrafeedback_m2_helpfulness_honesty" \
  --output_dir "${OUTPUT_DIR}_combined_helpfulness_honesty" \
  --report_to wandb \
  --run_name "${WANDB_NAME}_combined_helpfulness_honesty" \
  --listwise_num_responses 2 \
  --preference_dimensions helpfulness honesty \
  --do_eval False --eval_strategy no \
  --num_train_epochs 10 \
  --per_device_train_batch_size 2

for dim in "helpfulness" "honesty"
do
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/ddp.yaml \
  --num_processes=1 \
  scripts/dpo.py \
  --config "${CONFIG_PATH}" \
  --dataset_name "./data/cyclic_ultrafeedback_m2_helpfulness_honesty" \
  --output_dir "${OUTPUT_DIR}_helpfulness_honesty_single_${dim}_m2" \
  --report_to wandb \
  --run_name "${WANDB_NAME}_helpfulness_honesty_single_${dim}_m2" \
  --listwise_num_responses 2 \
  --preference_dimensions ${dim} \
  --do_eval False --eval_strategy no \
  --num_train_epochs 10 \
  --per_device_train_batch_size 2

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/ddp.yaml \
  --num_processes=1 \
  scripts/dpo.py \
  --config "${CONFIG_PATH}" \
  --dataset_name "./data/cyclic_ultrafeedback_m2_helpfulness_honesty" \
  --output_dir "${OUTPUT_DIR}_helpfulness_honesty_single_${dim}_m3" \
  --report_to wandb \
  --run_name "${WANDB_NAME}_helpfulness_honesty_single_${dim}_m3" \
  --listwise_num_responses 3 \
  --preference_dimensions ${dim} \
  --do_eval False --eval_strategy no \
  --num_train_epochs 15 \
  --per_device_train_batch_size 2

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/ddp.yaml \
  --num_processes=1 \
  scripts/dpo.py \
  --config "${CONFIG_PATH}" \
  --dataset_name "./data/cyclic_ultrafeedback_m2_helpfulness_honesty" \
  --output_dir "${OUTPUT_DIR}_helpfulness_honesty_single_${dim}_m4" \
  --report_to wandb \
  --run_name "${WANDB_NAME}_helpfulness_honesty_single_${dim}_m4" \
  --listwise_num_responses 4 \
  --preference_dimensions ${dim} \
  --do_eval False --eval_strategy no \
  --num_train_epochs 60 \
  --per_device_train_batch_size 2
done 
# --max_train_steps 1000 \
#   --dataset_dir "${DATASET_DIR}" \
#   --overwrite_output_dir