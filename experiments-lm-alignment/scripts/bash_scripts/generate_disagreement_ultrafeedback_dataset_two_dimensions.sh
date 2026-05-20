PYTHONPATH=src /home/ldy/miniconda3/envs/alignment/bin/python \
  scripts/create_disagreement_ultrafeedback_dataset.py \
  --dimensions instruction_following helpfulness \
  --allow_ties \
  --max_ties_per_dimension 1 \
  --min_distinct_rankings 2 \
  --output_dir data/ultrafeedback_disagreement_if_help_allow_one_tie \
  --create_splits

PYTHONPATH=src /home/ldy/miniconda3/envs/alignment/bin/python \
  scripts/create_disagreement_ultrafeedback_dataset.py \
  --dimensions instruction_following honesty \
  --allow_ties \
  --max_ties_per_dimension 1 \
  --min_distinct_rankings 2 \
  --output_dir data/ultrafeedback_disagreement_if_honesty_allow_one_tie \
  --create_splits

PYTHONPATH=src /home/ldy/miniconda3/envs/alignment/bin/python \
  scripts/create_disagreement_ultrafeedback_dataset.py \
  --dimensions instruction_following truthfulness \
  --allow_ties \
  --max_ties_per_dimension 1 \
  --min_distinct_rankings 2 \
  --output_dir data/ultrafeedback_disagreement_if_truthfulness_allow_one_tie \
  --create_splits

PYTHONPATH=src /home/ldy/miniconda3/envs/alignment/bin/python \
  scripts/create_disagreement_ultrafeedback_dataset.py \
  --dimensions helpfulness honesty \
  --allow_ties \
  --max_ties_per_dimension 1 \
  --min_distinct_rankings 2 \
  --output_dir data/ultrafeedback_disagreement_help_honesty_allow_one_tie \
  --create_splits

