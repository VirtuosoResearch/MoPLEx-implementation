#!/usr/bin/env bash
set -euo pipefail

dimensions=("instruction_following" "helpfulness" "truthfulness" "honesty")
# iterate over two

for dim1 in "${dimensions[@]}"
do
    for dim2 in "${dimensions[@]}"
    do
        if [[ "${dim1}" != "${dim2}" ]]
        then
        echo "Generating dataset for dimensions: ${dim1}, ${dim2}"
        PYTHONPATH=src /home/ldy/miniconda3/envs/alignment/bin/python scripts/create_cyclic_ultrafeedback_dataset.py \
        --dataset_name openbmb/UltraFeedback \
        --source_split train \
        --dimensions ${dim1} ${dim2} \
        --output_dir data/cyclic_ultrafeedback_m2_${dim1}_${dim2}
        else
        echo "Skipping identical dimensions: ${dim1}, ${dim2}"
        continue
        fi 
done 
done