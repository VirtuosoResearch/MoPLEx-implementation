#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python}"
SOURCE_DATASET="${SOURCE_DATASET:-SynthLabsAI/PERSONA}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/persona_baselines}"
PROMPT_MODE="${PROMPT_MODE:-instruction_only}"
NUM_PERSONAS="${NUM_PERSONAS:-10}"
NUM_RESPONSES_PER_PROMPT="${NUM_RESPONSES_PER_PROMPT:-4}"
HARD_NEGATIVE_PERSONA_POOL="${HARD_NEGATIVE_PERSONA_POOL:-all}"
NEGATIVE_POOL="${NEGATIVE_POOL:-same_split}"
NEGATIVE_RESPONSE_SOURCE="${NEGATIVE_RESPONSE_SOURCE:-other_persona}" # key
if [[ -z "${EVAL_NUM_RESPONSES_PER_PROMPT+x}" ]]; then
  if [[ "${NEGATIVE_RESPONSE_SOURCE}" == "other_persona" ]]; then
    EVAL_NUM_RESPONSES_PER_PROMPT="2"
  else
    EVAL_NUM_RESPONSES_PER_PROMPT=""
  fi
fi
SPLIT_MODE="${SPLIT_MODE:-prompt}"
NUM_CLUSTERS="${NUM_CLUSTERS:-${NUM_PERSONAS}}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-1000}"
METHODS="${METHODS:-single_dpo single_listdpo mixture_bt mixture_pl}"
FORCE_PREPARE="${FORCE_PREPARE:-false}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-VirtuosoResearch}"
export WANDB_PROJECT="${WANDB_PROJECT:-multimodal-preference-optimization}"
export WANDB_MODE="${WANDB_MODE:-online}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ "${NEGATIVE_RESPONSE_SOURCE}" == "original" ]]; then
  DATASET_NAME_PREFIX="${DATASET_NAME_PREFIX:-persona_${NUM_PERSONAS}}"
else
  DATASET_NAME_PREFIX="${DATASET_NAME_PREFIX:-persona_${NUM_PERSONAS}_${NEGATIVE_RESPONSE_SOURCE}}"
fi
RUN_TAG="n${NUM_PERSONAS}-${PROMPT_MODE}-${NEGATIVE_RESPONSE_SOURCE}-s${SEED}"
LISTWISE_K_SUFFIX="k${NUM_RESPONSES_PER_PROMPT}"
eval_num_responses_args=()
if [[ -n "${EVAL_NUM_RESPONSES_PER_PROMPT}" ]]; then
  eval_num_responses_args=(--eval_num_responses_per_prompt "${EVAL_NUM_RESPONSES_PER_PROMPT}")
  if [[ "${EVAL_NUM_RESPONSES_PER_PROMPT}" != "${NUM_RESPONSES_PER_PROMPT}" ]]; then
    LISTWISE_K_SUFFIX="${LISTWISE_K_SUFFIX}_evalk${EVAL_NUM_RESPONSES_PER_PROMPT}"
  fi
fi

PAIRWISE_DATASET_DIR="${DATA_ROOT}/${DATASET_NAME_PREFIX}_pairwise_${PROMPT_MODE}"
LISTWISE_DATASET_DIR="${DATA_ROOT}/${DATASET_NAME_PREFIX}_top1_listwise_${PROMPT_MODE}_${LISTWISE_K_SUFFIX}"

if [[ "${FORCE_PREPARE}" == "true" || ! -d "${PAIRWISE_DATASET_DIR}" || ! -d "${LISTWISE_DATASET_DIR}" ]]; then
  echo "Preparing PERSONA datasets under ${DATA_ROOT}"
  overwrite_args=()
  if [[ "${FORCE_PREPARE}" == "true" || -d "${PAIRWISE_DATASET_DIR}" || -d "${LISTWISE_DATASET_DIR}" ]]; then
    overwrite_args=(--overwrite)
  fi
  "${PYTHON}" scripts/prepare_persona_baselines_dataset.py \
    --dataset_name "${SOURCE_DATASET}" \
    --output_dir "${DATA_ROOT}" \
    --output_name_prefix "${DATASET_NAME_PREFIX}" \
    --num_personas "${NUM_PERSONAS}" \
    --num_responses_per_prompt "${NUM_RESPONSES_PER_PROMPT}" \
    --prompt_mode "${PROMPT_MODE}" \
    --hard_negative_persona_pool "${HARD_NEGATIVE_PERSONA_POOL}" \
    --negative_pool "${NEGATIVE_POOL}" \
    --negative_response_source "${NEGATIVE_RESPONSE_SOURCE}" \
    --split_mode "${SPLIT_MODE}" \
    --persona_subset_seed "${SEED}" \
    --split_seed "${SEED}" \
    --hard_negative_seed "${SEED}" \
    "${eval_num_responses_args[@]}" \
    "${overwrite_args[@]}"
fi

run_dpo() {
  local config_path="$1"
  local dataset_name="$2"
  local run_name="$3"
  local output_dir="$4"
  shift 4

  ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/single.yaml \
    --num_processes="${NUM_PROCESSES:-1}" \
    scripts/dpo.py \
    --config "${config_path}" \
    --dataset_name "${dataset_name}" \
    --output_dir "${output_dir}" \
    --run_name "${run_name}" \
    --seed "${SEED}" \
    --max_steps "${MAX_STEPS}" \
    "$@"
}

for method in ${METHODS}; do
  case "${method}" in
    single_dpo)
      run_dpo \
        recipes/qwen3-1b/dpo/persona/config_dpo_qlora.yaml \
        "${PAIRWISE_DATASET_DIR}" \
        "qwen3-0.6b-persona-single-dpo-${RUN_TAG}" \
        "outputs/persona/single-dpo-${RUN_TAG}"
      ;;
    single_listdpo)
      run_dpo \
        recipes/qwen3-1b/dpo/persona/config_top1_listwise_qlora.yaml \
        "${LISTWISE_DATASET_DIR}" \
        "qwen3-0.6b-persona-single-top1-listdpo-${RUN_TAG}-k${NUM_RESPONSES_PER_PROMPT}" \
        "outputs/persona/single-top1-listdpo-${RUN_TAG}-k${NUM_RESPONSES_PER_PROMPT}" \
        --listwise_num_responses "${NUM_RESPONSES_PER_PROMPT}"
      ;;
    mixture_bt)
      run_dpo \
        recipes/qwen3-1b/dpo/persona/config_mixture_bt_qlora.yaml \
        "${PAIRWISE_DATASET_DIR}" \
        "qwen3-0.6b-persona-mixture-bt-${RUN_TAG}-c${NUM_CLUSTERS}" \
        "outputs/persona/mixture-bt-${RUN_TAG}-c${NUM_CLUSTERS}" \
        --num_clusters "${NUM_CLUSTERS}"
      ;;
    mixture_pl)
      run_dpo \
        recipes/qwen3-1b/dpo/persona/config_mixture_pl_top1_qlora.yaml \
        "${LISTWISE_DATASET_DIR}" \
        "qwen3-0.6b-persona-mixture-pl-top1-${RUN_TAG}-k${NUM_RESPONSES_PER_PROMPT}-c${NUM_CLUSTERS}" \
        "outputs/persona/mixture-pl-top1-${RUN_TAG}-k${NUM_RESPONSES_PER_PROMPT}-c${NUM_CLUSTERS}" \
        --listwise_num_responses "${NUM_RESPONSES_PER_PROMPT}" \
        --num_clusters "${NUM_CLUSTERS}"
      ;;
    *)
      echo "Unknown METHOD '${method}'" >&2
      exit 1
      ;;
  esac
done
