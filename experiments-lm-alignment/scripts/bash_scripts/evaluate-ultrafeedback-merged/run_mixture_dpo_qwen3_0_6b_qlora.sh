#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/cyclic_ultrafeedback_merged}"
CONFIG_PATH="${CONFIG_PATH:-recipes/qwen3-1b/dpo/ultrafeedback_merged/config_mixture_qlora.yaml}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-1000}"
MIXTURE_TRAINING_MODE="${MIXTURE_TRAINING_MODE:-em_only}"
MIXTURE_REWARD_BACKEND="${MIXTURE_REWARD_BACKEND:-head}"
EM_TEMPERATURES="${EM_TEMPERATURES:-0.5 0.2}"
M_STEP_UPDATES="${M_STEP_UPDATES:-3 2 1}"
LEARNING_RATES="${LEARNING_RATES:-1e-6}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-VirtuosoResearch}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_MODE="${WANDB_MODE:-online}"
BASE_WANDB_NAME="${WANDB_NAME:-qwen3-0.6b-mixture-dpo-${MIXTURE_TRAINING_MODE}-${MIXTURE_REWARD_BACKEND}-ultrafeedback-merged-s${SEED}}"
BASE_OUTPUT_DIR="${OUTPUT_DIR:-outputs/ultrafeedback-merged/mixture-dpo-${MIXTURE_TRAINING_MODE}-${MIXTURE_REWARD_BACKEND}/qwen3-0.6b-s${SEED}}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing merged dataset directory: ${DATASET_DIR}" >&2
  echo "Run scripts/bash_scripts/merge_dataset.sh or scripts/merge_cyclic_ultrafeedback_datasets.py first." >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

for temperature in ${EM_TEMPERATURES}; do
for m_step_updates in ${M_STEP_UPDATES}; do
for learning_rate in ${LEARNING_RATES}; do
temp_tag="${temperature//./p}"
lr_tag="${learning_rate//./p}"
lr_tag="${lr_tag//-/m}"
m_tag="${m_step_updates//./p}"
run_name="${BASE_WANDB_NAME}-temp${temp_tag}-m${m_tag}-lr${lr_tag}"
output_dir="${BASE_OUTPUT_DIR}-temp${temp_tag}-m${m_tag}-lr${lr_tag}"

echo "Launching mixture DPO: temperature=${temperature}, m_step_updates=${m_step_updates}, learning_rate=${learning_rate}"

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/single.yaml \
  --num_processes="${NUM_PROCESSES:-1}" \
  scripts/dpo.py \
  --config "${CONFIG_PATH}" \
  --dataset_name "./data/cyclic_ultrafeedback_merged" \
  --output_dir "${output_dir}" \
  --run_name "${run_name}" \
  --report_to wandb \
  --seed "${SEED}" \
  --max_steps "${MAX_STEPS}" \
  --mixture_training_mode "${MIXTURE_TRAINING_MODE}" \
  --mixture_reward_backend "${MIXTURE_REWARD_BACKEND}" \
  --em_temperature "${temperature}" \
  --m_step_updates "${m_step_updates}" \
  --learning_rate "${learning_rate}" \
  "$@"
done
done
done
