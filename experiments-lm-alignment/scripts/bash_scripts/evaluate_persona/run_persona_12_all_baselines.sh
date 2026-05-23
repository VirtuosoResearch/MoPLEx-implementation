#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASELINES="${BASELINES:-dpo listpo emdpo_cluster maxmin_cluster micro_cluster}"
export SEEDS="${SEEDS:-42 43}"
export MAX_STEPS="${MAX_STEPS:-2000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

for baseline in ${BASELINES}; do
  case "${baseline}" in
    dpo)
      echo "Launching PERSONA-12 DPO: seeds=${SEEDS}"
      bash "${SCRIPT_DIR}/run_persona_12_dpo.sh" "$@"
      ;;
    listpo|listdpo)
      echo "Launching PERSONA-12 ListPO: seeds=${SEEDS}"
      bash "${SCRIPT_DIR}/run_persona_12_listpo.sh" "$@"
      ;;
    emdpo|emdpo_cluster)
      echo "Launching PERSONA-12 EM-DPO cluster baseline: seeds=${SEEDS}"
      bash "${SCRIPT_DIR}/run_persona_12_emdpo_cluster.sh" "$@"
      ;;
    maxmin|maxmin_cluster)
      echo "Launching PERSONA-12 MaxMin cluster baseline: seeds=${SEEDS}"
      bash "${SCRIPT_DIR}/run_persona_12_maxmin_cluster.sh" "$@"
      ;;
    micro|micro_cluster|mixture_bt)
      echo "Launching PERSONA-12 MiCRo mixture-BT cluster baseline: seeds=${SEEDS}"
      bash "${SCRIPT_DIR}/run_persona_12_micro_cluster.sh" "$@"
      ;;
    *)
      echo "Unknown PERSONA-12 baseline '${baseline}'" >&2
      echo "Valid baselines: dpo listpo emdpo_cluster maxmin_cluster micro_cluster" >&2
      exit 1
      ;;
  esac
done
