#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_UF_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASH_SCRIPTS_DIR="$(cd "${EVAL_UF_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
HELPER="${SCRIPT_DIR}/summarize_generation_win_rate.py"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/ultrafeedback_disagreement}"
SPLIT="${SPLIT:-validation}"
MAX_EXAMPLES="${MAX_EXAMPLES:-100}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-balanced_by_dimension}"
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-Qwen/Qwen3-0.6B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ultrafeedback-disagreement/generation-win-rate-vs-top-ranked}"
REFERENCE_RESPONSE_SOURCE="${REFERENCE_RESPONSE_SOURCE:-dataset_top_ranked}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
REWARD_BETA="${REWARD_BETA:-0.01}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-true}"
OVERWRITE="${OVERWRITE:-false}"
DRY_RUN="${DRY_RUN:-false}"
DISABLE_TQDM="${DISABLE_TQDM:-false}"
PRINT_SUMMARY="${PRINT_SUMMARY:-true}"

DPO_ADAPTER_PATH="${DPO_ADAPTER_PATH:-outputs/ultrafeedback-disagreement/dpo/qwen3-0.6b-all-s42/checkpoint-1500}"
LISTDPO_RUN_DIR="${LISTDPO_RUN_DIR:-outputs/ultrafeedback-disagreement/listpo-m4-linear-approx/qwen3-0.6b/approx-a2-instruction_following-ds0p25-s42}"
MIXTURE_MP4_RUN_DIR="${MIXTURE_MP4_RUN_DIR:-outputs/ultrafeedback-disagreement/mixture-pl-lora-linear-approx/augmented/qwen3-0.6b-ultrafeedback_disagreement_train_augmented_Qwen3_0p6B_ds0p25_k8_s42/approx-a2-m4-mp4-ms3-ds1p0-temp1p0-lr2em6-s42}"
MIXTURE_MP2_RUN_DIR="${MIXTURE_MP2_RUN_DIR:-outputs/ultrafeedback-disagreement/mixture-pl-lora-linear-approx/augmented/qwen3-0.6b-ultrafeedback_disagreement_train_augmented_Qwen3_0p6B_ds0p25_k8_s42/approx-a2-m4-mp2-ms3-ds1p0-temp1p0-lr2em6-s42}"

REWARD_INSTRUCTION_FOLLOWING="${REWARD_INSTRUCTION_FOLLOWING:-outputs/ultrafeedback-disagreement/dpo-by-dimension/qwen3-0.6b-ds0p25-instruction_following-s42}"
REWARD_HELPFULNESS="${REWARD_HELPFULNESS:-outputs/ultrafeedback-disagreement/dpo-by-dimension/qwen3-0.6b-ds0p25-helpfulness-s42}"
REWARD_HONESTY="${REWARD_HONESTY:-outputs/ultrafeedback-disagreement/dpo-by-dimension/qwen3-0.6b-ds0p25-honesty-s42}"
REWARD_TRUTHFULNESS="${REWARD_TRUTHFULNESS:-outputs/ultrafeedback-disagreement/dpo-by-dimension/qwen3-0.6b-ds0p25-truthfulness-s42}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

require_dir() {
  local path="$1"
  local description="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Missing ${description}: ${path}" >&2
    exit 1
  fi
}

require_adapter() {
  local path="$1"
  local description="$2"
  require_dir "${path}" "${description}"
  if [[ ! -f "${path}/adapter_config.json" ]]; then
    echo "Missing adapter_config.json for ${description}: ${path}" >&2
    exit 1
  fi
}

resolve_checkpoint() {
  "${PYTHON}" "${HELPER}" resolve-checkpoint "$1"
}

require_dir "${DATASET_DIR}" "UltraFeedback disagreement dataset"
require_adapter "${DPO_ADAPTER_PATH}" "DPO adapter"
require_adapter "${REWARD_INSTRUCTION_FOLLOWING}" "instruction_following reward adapter"
require_adapter "${REWARD_HELPFULNESS}" "helpfulness reward adapter"
require_adapter "${REWARD_HONESTY}" "honesty reward adapter"
require_adapter "${REWARD_TRUTHFULNESS}" "truthfulness reward adapter"

LISTDPO_ADAPTER_PATH="${LISTDPO_ADAPTER_PATH:-$(resolve_checkpoint "${LISTDPO_RUN_DIR}")}"
MIXTURE_MP4_CHECKPOINT="${MIXTURE_MP4_CHECKPOINT:-$(resolve_checkpoint "${MIXTURE_MP4_RUN_DIR}")}"
MIXTURE_MP2_CHECKPOINT="${MIXTURE_MP2_CHECKPOINT:-$(resolve_checkpoint "${MIXTURE_MP2_RUN_DIR}")}"

require_adapter "${LISTDPO_ADAPTER_PATH}" "ListDPO adapter"
require_dir "${MIXTURE_MP4_CHECKPOINT}" "mp4 mixture checkpoint"
require_dir "${MIXTURE_MP2_CHECKPOINT}" "mp2 mixture checkpoint"

reward_args=(
  --reward_adapter "instruction_following=${REWARD_INSTRUCTION_FOLLOWING}"
  --reward_adapter "helpfulness=${REWARD_HELPFULNESS}"
  --reward_adapter "honesty=${REWARD_HONESTY}"
  --reward_adapter "truthfulness=${REWARD_TRUTHFULNESS}"
)

flag_args=()
if [[ "${LOAD_IN_4BIT}" == "true" ]]; then
  flag_args+=(--load_in_4bit)
fi
if [[ "${OVERWRITE}" == "true" ]]; then
  flag_args+=(--overwrite)
fi
if [[ "${DISABLE_TQDM}" == "true" ]]; then
  flag_args+=(--disable_tqdm)
fi

run_eval() {
  local model_tag="$1"
  local adapter_path="$2"
  local output_dir="${OUTPUT_ROOT}/${model_tag}"
  require_adapter "${adapter_path}" "${model_tag} policy adapter"

  cmd=(
    "${PYTHON}" scripts/evaluate_generation_win_rate.py
    --dataset_name "${DATASET_DIR}"
    --split "${SPLIT}"
    --max_examples "${MAX_EXAMPLES}"
    --sample_strategy "${SAMPLE_STRATEGY}"
    --base_model_name_or_path "${BASE_MODEL_NAME_OR_PATH}"
    --policy_adapter_path "${adapter_path}"
    --policy_adapter_mode single
    --reference_response_source "${REFERENCE_RESPONSE_SOURCE}"
    "${reward_args[@]}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --seed "${SEED}"
    --batch_size "${BATCH_SIZE}"
    --reward_beta "${REWARD_BETA}"
    --max_length "${MAX_LENGTH}"
    --max_prompt_length "${MAX_PROMPT_LENGTH}"
    --output_dir "${output_dir}"
    --torch_dtype "${TORCH_DTYPE}"
    --device_map "${DEVICE_MAP}"
    "${flag_args[@]}"
  )

  echo "Evaluating ${model_tag}: adapter=${adapter_path}; output=${output_dir}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '  %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

echo "Resolved ListDPO adapter: ${LISTDPO_ADAPTER_PATH}"
echo "Resolved mp4 mixture checkpoint: ${MIXTURE_MP4_CHECKPOINT}"
echo "Resolved mp2 mixture checkpoint: ${MIXTURE_MP2_CHECKPOINT}"

run_eval "dpo" "${DPO_ADAPTER_PATH}"
run_eval "listdpo" "${LISTDPO_ADAPTER_PATH}"

for cluster_idx in 0 1 2 3; do
  run_eval "mixture_mp4_cluster_${cluster_idx}" "${MIXTURE_MP4_CHECKPOINT}/mixture_cluster_${cluster_idx}"
done

for cluster_idx in 0 1 2 3; do
  run_eval "mixture_mp2_cluster_${cluster_idx}" "${MIXTURE_MP2_CHECKPOINT}/mixture_cluster_${cluster_idx}"
done

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "DRY_RUN=true; skipping summary generation."
  exit 0
fi

"${PYTHON}" "${HELPER}" summarize \
  --output_root "${OUTPUT_ROOT}" \
  --output_json "${OUTPUT_ROOT}/summary.json" \
  --output_csv "${OUTPUT_ROOT}/summary.csv"

if [[ "${PRINT_SUMMARY}" == "true" ]]; then
  echo
  echo "Generation win-rate summary:"
  cat "${OUTPUT_ROOT}/summary.csv"
fi
