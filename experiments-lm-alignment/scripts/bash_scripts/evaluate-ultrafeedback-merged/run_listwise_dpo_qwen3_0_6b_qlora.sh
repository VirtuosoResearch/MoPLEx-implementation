#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/cyclic_ultrafeedback_merged}"
CONFIG_PATH="${CONFIG_PATH:-recipes/qwen3-1b/dpo/ultrafeedback_merged/config_listwise_qlora.yaml}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-1000}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_ENTITY="${WANDB_ENTITY:-VirtuosoResearch}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_NAME="${WANDB_NAME:-qwen3-0.6b-listwise-dpo-qlora-ultrafeedback-merged-s${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ultrafeedback-merged/listwise-dpo/qwen3-0.6b-s${SEED}}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing merged dataset directory: ${DATASET_DIR}" >&2
  echo "Run scripts/bash_scripts/merge_dataset.sh or scripts/merge_cyclic_ultrafeedback_datasets.py first." >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/single.yaml \
  --num_processes="${NUM_PROCESSES:-1}" \
  scripts/dpo.py \
  --config "${CONFIG_PATH}" \
  --dataset_name "./data/cyclic_ultrafeedback_merged" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${WANDB_NAME}" \
  --report_to wandb \
  --seed "${SEED}" \
  --max_steps "${MAX_STEPS}" \
  "$@"
