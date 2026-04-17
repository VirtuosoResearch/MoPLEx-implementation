#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

for responses in 4 3
do
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-VirtuosoResearch}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_NAME="${WANDB_NAME:-qwen3-small-1b-listwise-dpo-qlora-helpfulness-m${responses}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3-small-1b-listwise-dpo-qlora-helpfulness-m${responses}}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/ddp.yaml \
  --num_processes=1 \
  scripts/dpo.py \
  --config recipes/qwen3-1b/dpo/config_qlora_listwise_helpfulness.yaml \
  --output_dir "${OUTPUT_DIR}" \
  --report_to wandb \
  --run_name "${WANDB_NAME}" \
  --max_steps 4000 \
  --listwise_num_responses "${responses}" --per_device_train_batch_size 2  
done