#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/ultrafeedback_disagreement_train_augmented_Qwen3_0p6B_ds0p25_k8_s42}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/ultrafeedback-disagreement/ablations/single-pl-lora-by-criterion/qwen3-0.6b-ultrafeedback_disagreement}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/ultrafeedback-disagreement/augmented-response-ranking-eval/qwen3-0.6b-seed42}"

INSTRUCTION_FOLLOWING_CHECKPOINT="${INSTRUCTION_FOLLOWING_CHECKPOINT:-${CHECKPOINT_ROOT}/instruction_following/m4-ds0p25-beta0p2-r32-a64-lr1em5-s42-20260711-214224}"
HONESTY_CHECKPOINT="${HONESTY_CHECKPOINT:-${CHECKPOINT_ROOT}/honesty/m4-ds0p25-beta0p2-r32-a64-lr1em5-s42-20260711-225820}"
TRUTHFULNESS_CHECKPOINT="${TRUTHFULNESS_CHECKPOINT:-${CHECKPOINT_ROOT}/truthfulness/m4-ds0p25-beta0p2-r32-a64-lr1em5-s42-20260711-234818}"
HELPFULNESS_CHECKPOINT="${HELPFULNESS_CHECKPOINT:-${CHECKPOINT_ROOT}/helpfulness/m4-ds0p25-beta0p2-r32-a64-lr1em5-s42-20260712-003819}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

args=(
  --dataset_name "${DATASET_DIR}"
  --split train
  --criterion_checkpoint "instruction_following=${INSTRUCTION_FOLLOWING_CHECKPOINT}"
  --criterion_checkpoint "honesty=${HONESTY_CHECKPOINT}"
  --criterion_checkpoint "truthfulness=${TRUTHFULNESS_CHECKPOINT}"
  --criterion_checkpoint "helpfulness=${HELPFULNESS_CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE:-4}"
  --beta "${BETA:-0.2}"
  --max_length "${MAX_LENGTH:-1024}"
  --max_prompt_length "${MAX_PROMPT_LENGTH:-512}"
  --torch_dtype "${TORCH_DTYPE:-bfloat16}"
  --device_map "${DEVICE_MAP:-auto}"
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}"
)

if [[ -n "${MAX_ROWS:-}" ]]; then
  args+=(--max_rows "${MAX_ROWS}")
fi
if [[ "${OVERWRITE:-false}" == "true" ]]; then
  args+=(--overwrite)
fi
if [[ "${DISABLE_TQDM:-false}" == "true" ]]; then
  args+=(--disable_tqdm)
fi

cd "${REPO_ROOT}"
"${PYTHON:-/home/ldy/miniconda3/envs/alignment/bin/python}" \
  scripts/evaluate_augmented_response_rankings.py \
  "${args[@]}" \
  "$@"
