#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASH_SCRIPTS_DIR="$(cd "${EVAL_SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-recipes/qwen3-1b/dpo/ultrafeedback_merged/config_listwise_qlora.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/ultrafeedback_disagreement}"
SEEDS="${SEEDS:-42 43}"
RANKING_SIZES="${RANKING_SIZES:-4}"
MAX_STEPS="${MAX_STEPS:-3000}"
LEARNING_RATES="${LEARNING_RATES:-${LEARNING_RATE:-1e-5}}"
LISTWISE_BETAS="${LISTWISE_BETAS:-0.2}"
LORA_RANKS="${LORA_RANKS:-32}"
DOWNSAMPLE_RATIO="${DOWNSAMPLE_RATIO:-0.25}"
DOWNSAMPLE_GROUP_KEY="${DOWNSAMPLE_GROUP_KEY:-source_index}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-ranking_validation/ranking/pairwise_acc}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-VirtuosoResearch}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_MODE="${WANDB_MODE:-online}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing disagreement dataset directory: ${DATASET_DIR}" >&2
  echo "Run scripts/bash_scripts/generate_disagreement_ultrafeedback_dataset.sh first." >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DIMENSIONS="${DIMENSIONS:-instruction_following honesty truthfulness helpfulness}" 
if [[ -z "${DIMENSIONS}" && -f "${DATASET_DIR}/stats.json" ]]; then
  DIMENSIONS="$(
    /home/ldy/miniconda3/envs/alignment/bin/python -c \
      'import json, sys; stats=json.load(open(sys.argv[1])); print(" ".join(stats.get("dimensions", [])))' \
      "${DATASET_DIR}/stats.json"
  )"
fi

downsample_split_args=()
if [[ -n "${DOWNSAMPLE_SPLITS:-}" ]]; then
  read -r -a downsample_split_args <<< "${DOWNSAMPLE_SPLITS}"
  downsample_split_args=(--dataset_downsample_splits "${downsample_split_args[@]}")
fi

dataset_tag="$(basename "${DATASET_DIR}")"
ratio_tag="${DOWNSAMPLE_RATIO//./p}"
BASE_OUTPUT_DIR="${OUTPUT_DIR:-outputs/ultrafeedback-disagreement/ablations/single-pl-lora-by-criterion/qwen3-0.6b-${dataset_tag}}"
BASE_WANDB_NAME="${WANDB_NAME:-qwen3-0.6b-single-pl-lora-by-criterion-${dataset_tag}}"

for seed in ${SEEDS}; do
for learning_rate in ${LEARNING_RATES}; do
for listwise_beta in ${LISTWISE_BETAS}; do
for lora_rank in ${LORA_RANKS}; do
for ranking_size in ${RANKING_SIZES}; do
for dimension in ${DIMENSIONS}; do
  lora_alpha=$((2 * lora_rank))
  dim_tag="${dimension//[^a-zA-Z0-9]/_}"
  lr_tag="${learning_rate//./p}"
  lr_tag="${lr_tag//-/m}"
  beta_tag="${listwise_beta//./p}"
  beta_tag="${beta_tag//-/m}"

  run_name="${BASE_WANDB_NAME}-${dim_tag}-m${ranking_size}-ds${ratio_tag}-beta${beta_tag}-r${lora_rank}-a${lora_alpha}-lr${lr_tag}-s${seed}-$(date +%Y%m%d-%H%M%S)"
  output_dir="${BASE_OUTPUT_DIR}/${dim_tag}/m${ranking_size}-ds${ratio_tag}-beta${beta_tag}-r${lora_rank}-a${lora_alpha}-lr${lr_tag}-s${seed}-$(date +%Y%m%d-%H%M%S)"

  echo "Launching single PL LoRA ablation: dataset=${DATASET_DIR}; criterion=${dimension}; m=${ranking_size}; seed=${seed}; downsample=${DOWNSAMPLE_RATIO}; lr=${learning_rate}; beta=${listwise_beta}; lora_r=${lora_rank}; lora_alpha=${lora_alpha}; gradient_checkpointing=${GRADIENT_CHECKPOINTING}"

  ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "${ACCELERATE_CONFIG:-recipes/accelerate_configs/single.yaml}" \
    --num_processes="${NUM_PROCESSES:-1}" \
    scripts/dpo.py \
    --config "${CONFIG_PATH}" \
    --dataset_name "${DATASET_DIR}" \
    --dataset_format listwise \
    --dataset_train_split train \
    --dataset_test_split validation \
    --preference_dimensions "${dimension}" \
    --listwise true \
    --listwise_num_responses "${ranking_size}" \
    --listwise_min_responses 2 \
    --run_ranking_eval true \
    --ranking_eval_during_training true \
    --dataset_downsample_ratio "${DOWNSAMPLE_RATIO}" \
    --dataset_downsample_seed "${seed}" \
    --dataset_downsample_group_key "${DOWNSAMPLE_GROUP_KEY}" \
    "${downsample_split_args[@]}" \
    --use_mixture false \
    --lora_r "${lora_rank}" \
    --lora_alpha "${lora_alpha}" \
    --beta "${listwise_beta}" \
    --listwise_beta "${listwise_beta}" \
    --learning_rate "${learning_rate}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-2}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --eval_steps "${EVAL_STEPS:-200}" \
    --save_steps "${SAVE_STEPS:-200}" \
    --metric_for_best_model "${METRIC_FOR_BEST_MODEL}" \
    --output_dir "${output_dir}" \
    --run_name "${run_name}" \
    --report_to "${REPORT_TO:-wandb}" \
    --seed "${seed}" \
    "$@"
done
done
done
done
done
done
