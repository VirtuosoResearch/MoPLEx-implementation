#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-VirtuosoResearch}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_NAME="${WANDB_NAME:-zephyr-7b-listwise-dpo-helpfulness-qlora}"
export WANDB_MODE="${WANDB_MODE:-online}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/zephyr-7b-listwise-dpo-helpfulness-qlora-gpu1}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/ddp.yaml \
  --num_processes=1 \
  scripts/dpo.py \
  --config recipes/zephyr-7b-beta/dpo/config_qlora_listwise_helpfulness.yaml \
  --output_dir "${OUTPUT_DIR}" \
  --report_to wandb \
  --run_name "${WANDB_NAME}" \
  "$@"
