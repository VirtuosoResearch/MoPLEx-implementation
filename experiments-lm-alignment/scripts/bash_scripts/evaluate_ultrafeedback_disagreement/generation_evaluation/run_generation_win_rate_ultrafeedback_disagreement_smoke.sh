#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -x "/home/ldy/miniconda3/envs/alignment/bin/python" ]]; then
  DEFAULT_PYTHON="/home/ldy/miniconda3/envs/alignment/bin/python"
else
  DEFAULT_PYTHON="python"
fi

export PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"
export MAX_EXAMPLES="${MAX_EXAMPLES:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export OVERWRITE="${OVERWRITE:-true}"
export DISABLE_TQDM="${DISABLE_TQDM:-true}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ultrafeedback-disagreement/generation-win-rate-smoke}"

"${SCRIPT_DIR}/run_generation_win_rate_ultrafeedback_disagreement.sh"
