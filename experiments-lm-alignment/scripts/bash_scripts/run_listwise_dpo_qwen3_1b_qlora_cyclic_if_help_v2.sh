#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-VirtuosoResearch}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_NAME="${WANDB_NAME:-qwen3-0.6b-listwise-dpo-qlora-cyclic-m2-if-help}"
export WANDB_MODE="${WANDB_MODE:-online}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3-1b-listwise-dpo-qlora-cyclic-m2-if-help}"

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

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/ddp.yaml \
  --num_processes=1 \
  scripts/dpo.py \
  --config "${CONFIG_PATH}" \
  --output_dir "${OUTPUT_DIR}_m4_single_instruction_following" \
  --report_to wandb \
  --run_name "${WANDB_NAME}_m4_single_instruction_following" \
  --listwise_num_responses 4 \
  --preference_dimensions instruction_following \
  --do_eval False --eval_strategy no \
  --num_train_epochs 10 \
  --per_device_train_batch_size 2

# --max_train_steps 1000 \
#   --dataset_dir "${DATASET_DIR}" \
#   --overwrite_output_dir