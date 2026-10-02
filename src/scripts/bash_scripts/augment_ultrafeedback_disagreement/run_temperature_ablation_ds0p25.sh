#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

PYTHON="${PYTHON:-/home/ldy/miniconda3/envs/alignment/bin/python}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/ultrafeedback_disagreement}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-0.6B}"
BASE_NUM_RESPONSES="${BASE_NUM_RESPONSES:-4}"
NUM_NEW_RESPONSES="${NUM_NEW_RESPONSES:-4}"
TOTAL_RESPONSES_TAG="$((BASE_NUM_RESPONSES + NUM_NEW_RESPONSES))"
SEED="${SEED:-42}"
DOWNSAMPLE_RATIO="${DOWNSAMPLE_RATIO:-0.25}"
DOWNSAMPLE_GROUP_KEY="${DOWNSAMPLE_GROUP_KEY:-source_index}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TOP_P="${TOP_P:-0.95}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DEDUPLICATE="${DEDUPLICATE:-false}"
GENERATION_TEMPERATURES="${GENERATION_TEMPERATURES:-0.5 1.0 4.0}"
SCORE_TEMPERATURES="${SCORE_TEMPERATURES:-0.5 1.0 2.0 4.0}"
EXISTING_TEMPERATURE="${EXISTING_TEMPERATURE:-2.0}"
SUMMARY_OUTPUT_DIR="${SUMMARY_OUTPUT_DIR:-${REPO_ROOT}/outputs/ultrafeedback-disagreement/augmented-response-ranking-eval/qwen3-0.6b-seed${SEED}-temperature-ablation-summary}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

dataset_tag="$(basename "${DATASET_DIR}")"
ratio_tag="${DOWNSAMPLE_RATIO//./p}"
model_tag="$(basename "${MODEL_NAME_OR_PATH}")"
model_tag="${model_tag//./p}"
model_tag="${model_tag//-/_}"
base_augmented_dir="${REPO_ROOT}/data/${dataset_tag}_train_augmented_${model_tag}_ds${ratio_tag}_k${TOTAL_RESPONSES_TAG}_s${SEED}"
eval_root="${REPO_ROOT}/outputs/ultrafeedback-disagreement/augmented-response-ranking-eval"

temperature_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "${value}"
}

dataset_dir_for_temperature() {
  local temperature="$1"
  if [[ "${temperature}" == "${EXISTING_TEMPERATURE}" ]]; then
    printf '%s' "${base_augmented_dir}"
  else
    printf '%s_temp%s' "${base_augmented_dir}" "$(temperature_tag "${temperature}")"
  fi
}

eval_dir_for_temperature() {
  local temperature="$1"
  printf '%s/qwen3-0.6b-seed%s-temp%s' "${eval_root}" "${SEED}" "$(temperature_tag "${temperature}")"
}

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Missing ranking dataset directory: ${DATASET_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo "UltraFeedback response-generation temperature ablation"
echo "  GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  generation temperatures: ${GENERATION_TEMPERATURES}"
echo "  score temperatures:      ${SCORE_TEMPERATURES}"
echo "  deduplicate generations: ${DEDUPLICATE}"
echo "  existing ${EXISTING_TEMPERATURE} dataset: $(dataset_dir_for_temperature "${EXISTING_TEMPERATURE}")"
echo

for temperature in ${GENERATION_TEMPERATURES}; do
  output_dir="$(dataset_dir_for_temperature "${temperature}")"
  if [[ -d "${output_dir}" && "${FORCE_GENERATE:-false}" != "true" ]]; then
    echo "Skipping generation for temperature=${temperature}; dataset already exists: ${output_dir}"
    continue
  fi

  overwrite_args=()
  if [[ "${FORCE_GENERATE:-false}" == "true" ]]; then
    overwrite_args=(--overwrite)
  fi
  disable_tqdm_args=()
  if [[ "${DISABLE_TQDM:-false}" == "true" ]]; then
    disable_tqdm_args=(--disable_tqdm)
  fi
  deduplicate_args=()
  if [[ "${DEDUPLICATE}" == "false" ]]; then
    deduplicate_args=(--no-deduplicate)
  fi

  echo "Generating responses for temperature=${temperature}"
  echo "  output: ${output_dir}"
  "${PYTHON}" scripts/augment_ranking_dataset_with_generations.py \
    --dataset_name "${DATASET_DIR}" \
    --output_dir "${output_dir}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --num_new_responses "${NUM_NEW_RESPONSES}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${temperature}" \
    --top_p "${TOP_P}" \
    --batch_size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --device_map "${DEVICE_MAP}" \
    --prompt_format "${PROMPT_FORMAT}" \
    --splits train validation test \
    --augment_splits train \
    --dataset_downsample_ratio "${DOWNSAMPLE_RATIO}" \
    --dataset_downsample_seed "${SEED}" \
    --dataset_downsample_group_key "${DOWNSAMPLE_GROUP_KEY}" \
    --dataset_downsample_splits train \
    "${overwrite_args[@]}" \
    "${disable_tqdm_args[@]}" \
    "${deduplicate_args[@]}"
done

summary_inputs=()
for temperature in ${SCORE_TEMPERATURES}; do
  scored_dataset_dir="$(dataset_dir_for_temperature "${temperature}")"
  output_dir="$(eval_dir_for_temperature "${temperature}")"
  response_scores="${output_dir}/response_scores.csv"

  if [[ ! -d "${scored_dataset_dir}" ]]; then
    echo "Missing generated dataset for scoring temperature=${temperature}: ${scored_dataset_dir}" >&2
    exit 1
  fi

  if [[ -f "${response_scores}" && "${OVERWRITE_EVAL:-false}" != "true" ]]; then
    echo "Skipping scoring for temperature=${temperature}; response scores already exist: ${response_scores}"
  else
    if [[ -d "${output_dir}" && -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" && "${OVERWRITE_EVAL:-false}" != "true" ]]; then
      echo "Evaluation output exists but response_scores.csv is missing: ${output_dir}" >&2
      echo "Set OVERWRITE_EVAL=true to replace it." >&2
      exit 1
    fi

    eval_overwrite_args=()
    if [[ "${OVERWRITE_EVAL:-false}" == "true" ]]; then
      eval_overwrite_args=(--overwrite)
    fi

    echo "Scoring generated responses for temperature=${temperature}"
    echo "  dataset: ${scored_dataset_dir}"
    echo "  output:  ${output_dir}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    DATASET_DIR="${scored_dataset_dir}" \
    OUTPUT_DIR="${output_dir}" \
    PYTHON="${PYTHON}" \
    bash scripts/bash_scripts/evaluate_ultrafeedback_disagreement/ablations/run_evaluate_augmented_response_rankings.sh \
      "${eval_overwrite_args[@]}"
  fi

  summary_inputs+=(--input "${temperature}=${response_scores}")
done

echo "Summarizing generated-response utility gaps"
echo "  output: ${SUMMARY_OUTPUT_DIR}"
"${PYTHON}" scripts/summarize_augmented_response_utility_gaps.py \
  "${summary_inputs[@]}" \
  --output_dir "${SUMMARY_OUTPUT_DIR}"

echo "Done."
echo "  Summary CSV:  ${SUMMARY_OUTPUT_DIR}/generated_utility_gap_summary.csv"
echo "  Prompt CSV:   ${SUMMARY_OUTPUT_DIR}/generated_utility_gap_rows.csv"
echo "  Summary JSON: ${SUMMARY_OUTPUT_DIR}/generated_utility_gap_summary.json"
