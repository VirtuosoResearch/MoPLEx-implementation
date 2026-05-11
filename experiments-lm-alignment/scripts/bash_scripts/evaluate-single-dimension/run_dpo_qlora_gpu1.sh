#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-VirtuosoResearch}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_NAME="${WANDB_NAME:-zephyr-7b-dpo-qlora}"
export WANDB_MODE="${WANDB_MODE:-online}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/zephyr-7b-dpo-qlora-gpu1}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/ddp.yaml \
  --num_processes=1 \
  scripts/dpo.py \
  --config recipes/zephyr-7b-beta/dpo/config_qlora.yaml \
  --output_dir "${OUTPUT_DIR}" \
  --report_to wandb \
  --run_name "${WANDB_NAME}"

# CUDA_VISIBLE_DEVICES=0 python -m alpaca_eval.main evaluate_from_model zephyr-7b-dpo-qlora-gpu1-ckpt1600 \
#   --annotators_config=alpaca_eval_gpt4_0613 \
#   --max_instances=2 \
#   --output_path=/home/ldy/Scalable-preference-optimization-and-evaluation/evaluations/alpaca_eval/results/zephyr-7b-dpo-qlora-gpu1-ckpt1600-smoke

# setsid nohup bash scripts/bash_scripts/run_dpo_qlora_gpu1.sh > outputs/run_dpo_qlora_gpu1.log 2>&1 < /dev/null &