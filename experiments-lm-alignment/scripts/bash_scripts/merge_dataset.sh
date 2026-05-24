PYTHONPATH=src /home/michael/anaconda3/envs/dpo/bin/python scripts/merge_cyclic_ultrafeedback_datasets.py \
  --input_root data \
  --pattern 'cyclic_ultrafeedback_m2_*' \
  --output_dir data/cyclic_ultrafeedback_m2_all_pairs_merged \
  --validation_size 0.1 \
  --test_size 0.1 \
  --seed 42