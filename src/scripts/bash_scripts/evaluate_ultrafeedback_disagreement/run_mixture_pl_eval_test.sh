#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-recipes/qwen3-1b/dpo/ultrafeedback_merged/config_mixture_qlora.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/ultrafeedback_disagreement}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
RANKING_SPLIT="${RANKING_SPLIT:-test}"
DIMENSIONS="${DIMENSIONS:-instruction_following honesty truthfulness helpfulness}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Missing CHECKPOINT_PATH. Example:" >&2
  echo "  CHECKPOINT_PATH=outputs/.../checkpoint-2000 bash $0" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

read -r -a dimension_args <<< "${DIMENSIONS}"

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/single.yaml \
  --num_processes "${NUM_PROCESSES}" \
  scripts/dpo.py \
  --config "${CONFIG_PATH}" \
  --dataset_name "${DATASET_DIR}" \
  --dataset_format listwise \
  --dataset_train_split train \
  --dataset_test_split validation \
  --preference_dimensions "${dimension_args[@]}" \
  --run_ranking_eval true \
  --ranking_eval_during_training false \
  --eval_strategy no \
  --report_to none \
  --eval_only_ranking true \
  --eval_checkpoint_path "${CHECKPOINT_PATH}" \
  --eval_ranking_split "${RANKING_SPLIT}"
