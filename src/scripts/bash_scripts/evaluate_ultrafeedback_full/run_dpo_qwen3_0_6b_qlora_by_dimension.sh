#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-recipes/qwen3-1b/dpo/ultrafeedback_merged/config_dpo_qlora.yaml}"
SOURCE_DATASET="${SOURCE_DATASET:-openbmb/UltraFeedback}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/ultrafeedback_full_98_1_1}"
DIMENSIONS="${DIMENSIONS:-instruction_following helpfulness honesty truthfulness}"
SEEDS="${SEEDS:-42}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.98}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.01}"
TEST_RATIO="${TEST_RATIO:-0.01}"
MAX_STEPS="${MAX_STEPS:-2000}"
LISTWISE_NUM_RESPONSES="${LISTWISE_NUM_RESPONSES:-4}"
PAIRWISE_STRATEGY="${PAIRWISE_STRATEGY:-all_pairs}"
PYTHON="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-VirtuosoResearch}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_MODE="${WANDB_MODE:-online}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

for dimension in ${DIMENSIONS}; do
  dim_tag="${dimension//[^a-zA-Z0-9]/_}"
  dataset_dir="${DATA_ROOT}/pairwise-${dim_tag}-split-s${SPLIT_SEED}-${PAIRWISE_STRATEGY}"

  if [[ ! -d "${dataset_dir}" ]]; then
    echo "Preparing full UltraFeedback pairwise dataset for preference_dimension=${dimension}"
    "${PYTHON}" scripts/prepare_ultrafeedback_full_dimension_dataset.py \
      --dataset_name "${SOURCE_DATASET}" \
      --source_split "${SOURCE_SPLIT}" \
      --dimension "${dimension}" \
      --output_dir "${dataset_dir}" \
      --format pairwise \
      --seed "${SPLIT_SEED}" \
      --train_ratio "${TRAIN_RATIO}" \
      --validation_ratio "${VALIDATION_RATIO}" \
      --test_ratio "${TEST_RATIO}" \
      --listwise_num_responses "${LISTWISE_NUM_RESPONSES}" \
      --pairwise_strategy "${PAIRWISE_STRATEGY}"
  fi

  for seed in ${SEEDS}; do
    run_name="${WANDB_NAME:-qwen3-0.6b-dpo-qlora-ultrafeedback-full}-${dim_tag}-s${seed}"
    output_dir="${OUTPUT_DIR:-outputs/ultrafeedback-full/dpo-by-dimension/qwen3-0.6b}-${dim_tag}-s${seed}"

    echo "Launching pairwise DPO for preference_dimension=${dimension}, seed=${seed}"

    ACCELERATE_LOG_LEVEL=info accelerate launch \
      --config_file recipes/accelerate_configs/single.yaml \
      --num_processes="${NUM_PROCESSES:-1}" \
      scripts/dpo.py \
      --config "${CONFIG_PATH}" \
      --dataset_name "${dataset_dir}" \
      --dataset_format pairwise \
      --dataset_train_split train \
      --dataset_test_split validation \
      --preference_dimensions "${dimension}" \
      --metric_for_best_model rewards/accuracies \
      --output_dir "${output_dir}" \
      --run_name "${run_name}" \
      --report_to wandb \
      --seed "${seed}" \
      --max_steps "${MAX_STEPS}" \
      "$@"
  done
done
