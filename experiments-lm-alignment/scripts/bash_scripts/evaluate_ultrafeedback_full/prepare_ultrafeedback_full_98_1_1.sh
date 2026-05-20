#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS_DIR="$(cd "${BASH_SCRIPTS_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/ultrafeedback_full_98_1_1}"
export TRAIN_RATIO="${TRAIN_RATIO:-0.98}"
export VALIDATION_RATIO="${VALIDATION_RATIO:-0.01}"
export TEST_RATIO="${TEST_RATIO:-0.01}"

exec "${SCRIPT_DIR}/prepare_ultrafeedback_full_80_10_10.sh" "$@"
