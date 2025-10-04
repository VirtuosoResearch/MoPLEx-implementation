#!/usr/bin/env bash
set -euo pipefail

SOURCES=(sentiment conciseness)

default_device=${CUDA_VISIBLE_DEVICES:-0}
default_log_level=${ACCELERATE_LOG_LEVEL:-info}

for source in "${SOURCES[@]}"; do
    CUDA_VISIBLE_DEVICES=${default_device} ACCELERATE_LOG_LEVEL=${default_log_level} \
    accelerate launch run_simpo.py training_configs/gpt2-imdb-simpo.yaml \
        training_method="simpo" \
        load_multi_preference=True \
        load_multi_preference_dataset=imdb_preference_with_source \
        model_name_or_path=gpt2 \
        preference_sources="${source}" \
        output_dir="outputs/gpt2-imdb-${source}" \
        run_name="gpt2-imdb-${source}" \
        seed=42 "$@"

done
