# Merged UltraFeedback Ranking Comparison

These scripts compare three methods on `data/cyclic_ultrafeedback_merged`:

- Pairwise DPO: listwise rows are converted to top-vs-bottom pairwise examples.
- Single ListDPO/ListPO: one shared listwise preference model across all dimensions.
- Mixture DPO: listwise DPO plus a 4-component mixture of PL models.

All configs use the original 4-response ranking rows with `listwise_num_responses=4`.
The default per-device train batch size is matched across methods.

Run one method:

```bash
scripts/bash_scripts/evaluate-ultrafeedback-merged/run_dpo_qwen3_0_6b_qlora.sh
scripts/bash_scripts/evaluate-ultrafeedback-merged/run_listwise_dpo_qwen3_0_6b_qlora.sh
scripts/bash_scripts/evaluate-ultrafeedback-merged/run_mixture_dpo_qwen3_0_6b_qlora.sh
```

Run all three:

```bash
scripts/bash_scripts/evaluate-ultrafeedback-merged/run_all_qwen3_0_6b_qlora.sh
```

Useful overrides:

```bash
SEEDS="42 43 44" MAX_STEPS=2000 CUDA_VISIBLE_DEVICES=0 \
  scripts/bash_scripts/evaluate-ultrafeedback-merged/run_all_qwen3_0_6b_qlora.sh
```

For an EM-only mixture diagnostic:

```bash
MIXTURE_TRAINING_MODE=em_only \
  scripts/bash_scripts/evaluate-ultrafeedback-merged/run_mixture_dpo_qwen3_0_6b_qlora.sh
```

Main validation metrics:

- `ranking_validation/ranking/pairwise_acc`: shared policy ranking accuracy.
- `ranking_validation/ranking/by_dimension/<dimension>/pairwise_acc`: per-dimension policy accuracy.
- `ranking_validation/mixture_posterior/pairwise_acc`: posterior-selected mixture component accuracy.
- `ranking_validation/mixture/cluster_acc`, `mixture/nmi`, `mixture/ari`: clustering diagnostics.

Final offline metrics are saved for `train`, `validation`, and `test` after training.
