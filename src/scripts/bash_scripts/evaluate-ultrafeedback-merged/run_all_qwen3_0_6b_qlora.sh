#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${SEEDS:-42}"

for seed in ${SEEDS}; do
  echo "Launching DPO baseline with seed ${seed}"
  SEED="${seed}" "${SCRIPT_DIR}/run_dpo_qwen3_0_6b_qlora.sh" "$@"

  echo "Launching single ListDPO baseline with seed ${seed}"
  SEED="${seed}" "${SCRIPT_DIR}/run_listwise_dpo_qwen3_0_6b_qlora.sh" "$@"

  # echo "Launching Mixture DPO with seed ${seed}"
  # SEED="${seed}" "${SCRIPT_DIR}/run_mixture_dpo_qwen3_0_6b_qlora.sh" "$@"
done
