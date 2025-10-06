1. Prepare for the score matrix

estimation: 
python -m preference_optimization.attribution.prepare_score_matrix results/imdb_qwen/dpo_evaluation_results.json score_estimated.npz --format logistic
real: 
python -m attribution.prepare_score_matrix \
  --prefix outputs/subset-simpo-qwen-imdb-lora- \
  --format evaluation \
  --output-dir scores

2. Construct affnity
original score:
python -m preference_optimization.attribution.annotator_surrogate_pipeline --score-matrix scores/combined_score_matrix.npz --skip-surrogate
surrogate model:
python -m preference_optimization.attribution.annotator_surrogate_pipeline --score-matrix score_estimated.npz

3. Cluster-weighted accuracy (train per cluster and aggregate)
python -m preference_optimization.attribution.run_simpo_cluster \
  training_configs/base-simpo.yaml \
  results/annotator_affinity/clusters_actual.json \
  --output-root outputs/cluster_runs \
  --overrides beta=2.0 training_method="simpo" load_multi_preference_dataset=imdb_preference_with_source \
  --summary-output results/annotator_affinity/cluster_metrics.json
