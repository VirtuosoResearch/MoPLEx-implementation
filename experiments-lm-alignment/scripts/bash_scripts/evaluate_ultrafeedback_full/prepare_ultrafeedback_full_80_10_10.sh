#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

SOURCE_DATASET="${SOURCE_DATASET:-openbmb/UltraFeedback}"
SOURCE_SPLIT="${SOURCE_SPLIT:-train}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/ultrafeedback_full_80_10_10}"
DIMENSIONS="${DIMENSIONS:-instruction_following helpfulness honesty truthfulness}"
FORMATS="${FORMATS:-listwise pairwise}"
SPLIT_SEED="${SPLIT_SEED:-42}"
LISTWISE_NUM_RESPONSES="${LISTWISE_NUM_RESPONSES:-4}"
LISTWISE_MIN_RESPONSES="${LISTWISE_MIN_RESPONSES:-2}"
PAIRWISE_STRATEGY="${PAIRWISE_STRATEGY:-all_pairs}"
FORCE_PREPARE="${FORCE_PREPARE:-false}"
PYTHON="${PYTHON:-python}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

for dimension in ${DIMENSIONS}; do
  dim_tag="${dimension//[^a-zA-Z0-9]/_}"

  for format in ${FORMATS}; do
    case "${format}" in
      listwise)
        dataset_dir="${DATA_ROOT}/listwise-${dim_tag}-split-s${SPLIT_SEED}-k${LISTWISE_NUM_RESPONSES}"
        extra_args=()
        ;;
      pairwise)
        dataset_dir="${DATA_ROOT}/pairwise-${dim_tag}-split-s${SPLIT_SEED}-${PAIRWISE_STRATEGY}"
        extra_args=(--pairwise_strategy "${PAIRWISE_STRATEGY}")
        ;;
      *)
        echo "Unsupported format '${format}'. Use FORMATS='listwise pairwise' or one of those values." >&2
        exit 1
        ;;
    esac

    if [[ -d "${dataset_dir}" && "${FORCE_PREPARE}" != "true" ]]; then
      echo "Skipping existing ${format} dataset for ${dimension}: ${dataset_dir}"
      continue
    fi

    echo "Preparing ${format} dataset for preference_dimension=${dimension}"
    "${PYTHON}" scripts/prepare_ultrafeedback_full_dimension_dataset.py \
      --dataset_name "${SOURCE_DATASET}" \
      --source_split "${SOURCE_SPLIT}" \
      --dimension "${dimension}" \
      --output_dir "${dataset_dir}" \
      --format "${format}" \
      --seed "${SPLIT_SEED}" \
      --listwise_num_responses "${LISTWISE_NUM_RESPONSES}" \
      --listwise_min_responses "${LISTWISE_MIN_RESPONSES}" \
      "${extra_args[@]}"
  done
done
